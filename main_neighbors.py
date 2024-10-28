"""
PyTorch DDP integrated with PyGeom for multi-node training
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging

from typing import Optional, Union, Callable, Tuple, Dict

import numpy as np

import hydra
import time
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
import torch.distributions as tdist 

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
Tensor = torch.Tensor

# PyTorch Geometric
import torch_geometric

import models.gnn as gnn

import dataprep.nekrs_graph_setup as ngs

log = logging.getLogger(__name__)

# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'CPU'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)



def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)

    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup():
    dist.destroy_process_group()

def metric_average(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val

class Trainer:
    def __init__(self, cfg: DictConfig, scaler: Optional[GradScaler] = None):
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.backend
        if WITH_DDP:
            init_process_group(RANK, SIZE, backend=self.backend)
        
        # ~~~~ Init torch stuff 
        self.setup_torch()

        # ~~~~ Init training and testing loss history 
        self.loss_hist_train = np.zeros(self.cfg.epochs)
        self.loss_hist_test = np.zeros(self.cfg.epochs)
        self.lr_hist = np.zeros(self.cfg.epochs)

        # ~~~~ Init datasets
        self.data = self.setup_data()
        # if WITH_CUDA: 
        #     self.data['train']['stats'][0][0] = self.data['train']['stats'][0][0].cuda()
        #     self.data['train']['stats'][0][1] = self.data['train']['stats'][0][1].cuda()
        #     self.data['train']['stats'][1][0] = self.data['train']['stats'][1][0].cuda()
        #     self.data['train']['stats'][1][1] = self.data['train']['stats'][1][1].cuda()

        # ~~~~ Init model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        # ~~~~ Set model and checkpoint savepaths:
        try:
            self.ckpt_path = cfg.ckpt_dir + self.model.get_save_header() + '.tar'
            self.model_path = cfg.model_dir + self.model.get_save_header() + '.tar'
        except (AttributeError) as e:
            self.ckpt_path = cfg.ckpt_dir + 'checkpoint.tar'
            self.model_path = cfg.model_dir + 'model.tar'

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.epoch = 0
        self.epoch_start = 1
        self.training_iter = 0
        if self.cfg.restart:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.epoch_start = ckpt['epoch'] + 1
            self.epoch = self.epoch_start
            self.training_iter = ckpt['training_iter']
            self.loss_hist_train = ckpt['loss_hist_train']
            self.loss_hist_test = ckpt['loss_hist_test']
            self.lr_hist = ckpt['lr_hist']

            if len(self.loss_hist_train) < self.cfg.epochs:
                loss_hist_train_new = np.zeros(self.cfg.epochs)
                loss_hist_test_new = np.zeros(self.cfg.epochs)
                lr_hist_new = np.zeros(self.cfg.epochs)
                loss_hist_train_new[:len(self.loss_hist_train)] = self.loss_hist_train
                loss_hist_test_new[:len(self.loss_hist_test)] = self.loss_hist_test
                lr_hist_new[:len(self.lr_hist)] = self.lr_hist
                self.loss_hist_train = loss_hist_train_new
                self.loss_hist_test = loss_hist_test_new
                self.lr_hist = lr_hist_new 
            

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

        # ~~~~ Set loss function 
        self.loss_fn = nn.MSELoss()

        # ~~~~ Set optimizer 
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Set scheduler 
        self.scheduler = self.build_scheduler(self.optimizer)

        # ~~~~ Load optimizer+scheduler parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if RANK == 0: 
                astr = 'RESTARTING FROM CHECKPOINT -- STATE AT EPOCH %d/%d' %(self.epoch_start-1, self.cfg.epochs)
                sepstr = '-' * len(astr)
                log.info(sepstr)
                log.info(astr)
                log.info(sepstr)

    def build_model(self) -> nn.Module:
         
        sample = self.data['train']['example']
        input_node_channels = sample.x.shape[1]
        input_edge_channels = sample.pos.shape[1] + sample.x.shape[1] + 1 
        hidden_channels = self.cfg.hidden_channels 
        output_node_channels = sample.y.shape[1] 
        n_mlp_hidden_layers = self.cfg.n_mlp_hidden_layers 
        n_messagePassing_layers = self.cfg.n_messagePassing_layers 

        name = self.cfg.model_name
        model = gnn.GNN_Element_Neighbor(
                           input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)

                
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        optimizer = optim.Adam(model.parameters(),
                               lr=SIZE * self.cfg.lr_init)
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
        return scheduler

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def setup_data(self):
        kwargs = {}
        
        # single snapshot - oneshot  
        #train_dataset = torch.load(self.cfg.data_dir + "Single_Snapshot_Re_1600_T_10.0_Interp_1to7/train_dataset.pt")
        #test_dataset = torch.load(self.cfg.data_dir + "Single_Snapshot_Re_1600_T_10.0_Interp_1to7/valid_dataset.pt")

        # multi snapshot - oneshot  
        n_element_neighbors = self.cfg.n_element_neighbors
        #train_dataset = torch.load(self.cfg.data_dir + f"/train_dataset.pt")
        #test_dataset = torch.load(self.cfg.data_dir + f"/valid_dataset.pt")
        train_dataset = torch.load(self.cfg.data_dir + f"Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_1to7_Neighbors_{n_element_neighbors}/train_dataset.pt")
        test_dataset = torch.load(self.cfg.data_dir + f"Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_1to7_Neighbors_{n_element_neighbors}/valid_dataset.pt")

        # # multi snapshot - incr 
        # train_dataset = [] 
        # test_dataset = [] 
        # train_dataset += torch.load(self.cfg.data_dir + "Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_1to3/train_dataset_p3.pt")
        # train_dataset += torch.load(self.cfg.data_dir + "Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_3to5/train_dataset_p5.pt")
        # train_dataset += torch.load(self.cfg.data_dir + "Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_5to7/train_dataset_p7.pt")
        # test_dataset += torch.load(self.cfg.data_dir + "Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_1to3/valid_dataset_p3.pt")
        # test_dataset += torch.load(self.cfg.data_dir + "Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_3to5/valid_dataset_p5.pt")
        # test_dataset += torch.load(self.cfg.data_dir + "Multi_Snapshot_Re_1600_T_8.0_9.0_10.0_Interp_5to7/valid_dataset_p7.pt")

        if RANK == 0:
            log.info('train dataset: %d elements' %(len(train_dataset)))
            log.info('valid dataset: %d elements' %(len(test_dataset)))

        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=SIZE, rank=RANK, shuffle=True,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=SIZE, rank=RANK, shuffle=False,
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset, 
            batch_size=self.cfg.test_batch_size,
            sampler=test_sampler,
        )

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
                'example': train_dataset[0]
                # 'stats': [data_mean, data_std] 
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
            }
        }

    def train_step(
        self,
        data: DataBatch
    ) -> Tensor:
        #t_total = time.time()
        try: 
            _ = data.node_weight
        except AttributeError: 
            data.node_weight = data.x.new_ones(data.x.shape[0], 1)
        if WITH_CUDA:
            data.x = data.x.cuda()
            data.x_mean = data.x_mean.cuda()
            data.x_std = data.x_std.cuda()
            data.node_weight = data.node_weight.cuda()
            data.y = data.y.cuda()
            data.edge_index = data.edge_index.cuda()
            data.pos_norm = data.pos_norm.cuda()
            data.batch = data.batch.cuda()
            data.central_element_mask = data.central_element_mask.cuda()

        #t_1 = time.time()
        if self.cfg.n_element_neighbors > 0:
            edge_index_coin = ngs.get_edge_index_coincident(data.batch, data.pos_norm, data.edge_index)
            degree = torch_geometric.utils.degree(edge_index_coin[1,:], num_nodes = data.pos.shape[0])
            degree += 1.
        else:
            edge_index_coin = None
            degree = None
        #t_1 = time.time() - t_1

        self.optimizer.zero_grad()

        # 1) Preprocessing: scale input  
        eps = 1e-10
        x_scaled = (data.x - data.x_mean)/(data.x_std + eps) 
          
        # 2) evaluate model 
        #t_2 = time.time()
        out_gnn = self.model(
                x = x_scaled,
                edge_index = data.edge_index,
                pos = data.pos_norm,
                batch = data.batch,
                edge_index_coin = edge_index_coin,
                degree = degree)
        #t_2 = time.time() - t_2
        
        # 3) set the target -- target = data.x + GNN(x_scaled)  
        mask = data.central_element_mask
        target = (data.y - data.x[mask])/(data.x_std[mask] + eps)
        #target = data.y - data.x

        # 4) evaluate loss 
        # loss = self.loss_fn(out_gnn, target) # vanilla mse 
        loss = torch.mean( data.node_weight * (out_gnn[mask] - target)**2 ) # weighted mse 

        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        #t_total = time.time() - t_total

        #if RANK == 0:
        #    if self.training_iter < 500:
        #        log.info(f"t_1: {t_1}s \t t_2: {t_2}s \t t_total: {t_total}s")

        return loss

    def train_epoch(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        running_loss = torch.tensor(0.)

        count = torch.tensor(0.)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, data in enumerate(train_loader):
            #print('Rank %d, bid %d, data:' %(RANK, bidx), data.y[1].shape)
            loss = self.train_step(data)
            running_loss += loss.item()

            count += 1 # accumulate current batch count 
            self.training_iter += 1 # accumulate total training iteration
            
            # Log on Rank 0:
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - start,
                    'batch_loss': loss.item(),
                    'running_loss': running_loss,
                }
                pre = [
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        #f' {bidx+1}/{len(train_sampler)}'
                        f' Batch {bidx+1}'
                        f' ({100. * (bidx+1) / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))

        # divide running loss by number of batches
        running_loss = running_loss / count

        # Allreduce, mean
        loss_avg = metric_average(running_loss)
        return {'loss': loss_avg}

    def test(self) -> dict:
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()
        self.model.eval()
        test_loader = self.data['test']['loader']
        with torch.no_grad():
            for data in test_loader:
                try: 
                    _ = data.node_weight
                except AttributeError: 
                    data.node_weight = data.x.new_ones(data.x.shape[0], 1)
                if WITH_CUDA:
                    data.x = data.x.cuda()
                    data.x_mean = data.x_mean.cuda()
                    data.x_std = data.x_std.cuda()
                    data.node_weight = data.node_weight.cuda()
                    data.y = data.y.cuda()
                    data.edge_index = data.edge_index.cuda()
                    data.pos_norm = data.pos_norm.cuda()
                    data.batch = data.batch.cuda()

                if self.cfg.n_element_neighbors > 0:
                    edge_index_coin = ngs.get_edge_index_coincident(data.batch, data.pos_norm, data.edge_index)
                    degree = torch_geometric.utils.degree(edge_index_coin[1,:], num_nodes = data.pos.shape[0])
                    degree += 1.
                else:
                    edge_index_coin = None
                    degree = None

                # 1) Preprocessing: scale input  
                eps = 1e-10
                x_scaled = (data.x - data.x_mean)/(data.x_std + eps) 
                  
                # 2) evaluate model 
                out_gnn = self.model(
                        x = x_scaled,
                        edge_index = data.edge_index,
                        pos = data.pos_norm,
                        batch = data.batch,
                        edge_index_coin = edge_index_coin,
                        degree = degree)
                
                # 3) set the target -- target = data.x + GNN(x_scaled)  
                mask = data.central_element_mask
                target = (data.y - data.x[mask])/(data.x_std[mask] + eps)
                #target = data.y - data.x

                # 4) evaluate loss 
                # loss = self.loss_fn(out_gnn, target) # vanilla mse 
                loss = torch.mean( data.node_weight * (out_gnn[mask] - target)**2 ) # weighted mse 
                
                running_loss += loss.item()
                count += 1

            running_loss = running_loss / count
            loss_avg = metric_average(running_loss)

        return {'loss': loss_avg}


def run_demo(demo_fn: Callable, world_size: int | str) -> None:
    mp.spawn(demo_fn,  # type: ignore
             args=(world_size,),
             nprocs=int(world_size),
             join=True)

def train(cfg: DictConfig):
    start = time.time()
    trainer = Trainer(cfg)
    epoch_times = []
    valid_times = []

    for epoch in range(trainer.epoch_start, cfg.epochs+1):
        # ~~~~ Training step 
        t0 = time.time()
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch(epoch)

        trainer.loss_hist_train[epoch-1] = train_metrics["loss"]

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ~~~~ Validation step
        t0 = time.time()
        test_metrics = trainer.test()
        trainer.loss_hist_test[epoch-1] = test_metrics["loss"]
        valid_time = time.time() - t0
        valid_times.append(valid_time)

        # ~~~~ Learning rate 
        lr = trainer.optimizer.param_groups[0]['lr']
        trainer.lr_hist[epoch-1] = lr

        if RANK == 0:
            astr = f'[TEST] loss={test_metrics["loss"]:.4e}'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={train_metrics["loss"]:.4e}',
                f'epoch_time={epoch_time:.4g} sec',
                f' valid_time={valid_time:.4g} sec',
                f' learning_rate={lr:.6g}'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        # ~~~~ Step scheduler based on validation loss
        trainer.scheduler.step(test_metrics["loss"])

        # ~~~~ Checkpointing step 
        if epoch % cfg.ckptfreq == 0 and RANK == 0:
            astr = 'Checkpointing on root processor, epoch = %d' %(epoch)
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)

            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
           
            if WITH_DDP and SIZE > 1:
                ckpt = {'epoch' : epoch, 
                        'training_iter' : trainer.training_iter,
                        'model_state_dict' : trainer.model.module.state_dict(), 
                        'optimizer_state_dict' : trainer.optimizer.state_dict(), 
                        'scheduler_state_dict' : trainer.scheduler.state_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test,
                        'lr_hist' : trainer.lr_hist} 
            else:
                ckpt = {'epoch' : epoch, 
                        'training_iter' : trainer.training_iter,
                        'model_state_dict' : trainer.model.state_dict(), 
                        'optimizer_state_dict' : trainer.optimizer.state_dict(), 
                        'scheduler_state_dict' : trainer.scheduler.state_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test,
                        'lr_hist' : trainer.lr_hist}

            torch.save(ckpt, trainer.ckpt_path)
        dist.barrier()

    rstr = f'[{RANK}] ::'
    log.info(' '.join([
        rstr,
        f'Total training time: {time.time() - start} seconds'
    ]))

    if RANK == 0:
        if WITH_CUDA:  
            trainer.model.to('cpu')
        if not os.path.exists(cfg.model_dir):
            os.makedirs(cfg.model_dir)
        if WITH_DDP and SIZE > 1:
            save_dict = {
                        'state_dict' : trainer.model.module.state_dict(), 
                        'input_dict' : trainer.model.module.input_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test,
                        'lr_hist' : trainer.lr_hist,
                        'training_iter' : trainer.training_iter
                        }
        else:
            save_dict = {   
                        'state_dict' : trainer.model.state_dict(), 
                        'input_dict' : trainer.model.input_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test,
                        'lr_hist' : trainer.lr_hist,
                        'training_iter' : trainer.training_iter
                        }

        torch.save(save_dict, trainer.model_path)


def write_full_dataset(cfg: DictConfig):
    return 



@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices. Seed = %d' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count(), cfg.seed))

    if RANK == 0:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('INPUTS:')
        print(OmegaConf.to_yaml(cfg)) 
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    train(cfg)
    cleanup()


if __name__ == '__main__':
    main()