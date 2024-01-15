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

def write_full_dataset(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device_for_loading = 'cpu'
    fraction_valid = 0.1 

    train_dataset = []
    test_dataset = [] 

    mean_x_i = [] 
    mean_y_i = []
    var_x_i = []
    var_y_i = []
    n_i = []

    data_read_world_size = 4
    snapshot_time_list = ['8.0', '9.0', '10.0']
    Re_list = ['1600', '2000', '2400']
    #snapshot_time_list = ['9.0']

    for Re_id in range(len(Re_list)):
        for snap_id in range(len(snapshot_time_list)):
            for i in range(data_read_world_size):
                Re = Re_list[Re_id]
                snapshot_time = snapshot_time_list[snap_id]
                data_x_path = cfg.case_path + '/Re_%s_poly_7/gnn_outputs_recon_poly_7' %(Re) + '/fld_u_time_%s_rank_%d_size_%d' %(snapshot_time,i,data_read_world_size) # input  
                data_y_path = cfg.case_path + '/Re_%s_poly_7/gnn_outputs_target_poly_7' %(Re) + '/fld_u_time_%s_rank_%d_size_%d' %(snapshot_time,i,data_read_world_size) # target 
                edge_index_path = cfg.case_path + '/Re_%s_poly_7/gnn_outputs_recon_poly_7' %(Re) + '/edge_index_element_local_rank_%d_size_%d' %(i,data_read_world_size) 
                node_element_ids_path = cfg.case_path + '/Re_%s_poly_7/gnn_outputs_recon_poly_7' %(Re) + '/node_element_ids_rank_%d_size_%d' %(i,data_read_world_size)
                global_ids_path = cfg.case_path + '/Re_%s_poly_7/gnn_outputs_recon_poly_7' %(Re) + '/global_ids_rank_%d_size_%d' %(i,data_read_world_size) 
                pos_path = cfg.case_path + '/Re_%s_poly_7/gnn_outputs_recon_poly_7' %(Re) + '/pos_node_rank_%d_size_%d' %(i,data_read_world_size) 
                        
                # data_x_path = cfg.case_path + '/gnn_outputs_recon_poly_7' + '/fld_u_rank_%d_size_%d' %(i,data_read_world_size) # input  
                # data_y_path = cfg.case_path + '/gnn_outputs_original_poly_7' + '/fld_u_rank_%d_size_%d' %(i,data_read_world_size) # target 
                # edge_index_path = cfg.case_path + '/Re_1600_poly_7/gnn_outputs_recon_poly_7' + '/edge_index_element_local_rank_%d_size_%d' %(i,data_read_world_size) 
                # node_element_ids_path = cfg.case_path + '/Re_1600_poly_7/gnn_outputs_recon_poly_7' + '/node_element_ids_rank_%d_size_%d' %(i,data_read_world_size)
                # global_ids_path = cfg.case_path + '/Re_1600_poly_7/gnn_outputs_recon_poly_7' + '/global_ids_rank_%d_size_%d' %(i,data_read_world_size) 
                # pos_path = cfg.case_path + '/Re_1600_poly_7/gnn_outputs_recon_poly_7' + '/pos_node_rank_%d_size_%d' %(i,data_read_world_size) 

                # # FOR TESTING ONLY -- poly 1 data 
                # data_x_path = cfg.case_path + '/gnn_outputs_distributed_gnn/gnn_outputs_poly_1' + '/fld_u_rank_%d_size_%d' %(i,data_read_world_size) # input  
                # data_y_path = cfg.case_path + '/gnn_outputs_distributed_gnn/gnn_outputs_poly_1' + '/fld_u_rank_%d_size_%d' %(i,data_read_world_size) # target 
                # edge_index_path = cfg.case_path + '/gnn_outputs_distributed_gnn/gnn_outputs_poly_1' + '/edge_index_element_local_rank_%d_size_%d' %(i,data_read_world_size) 
                # node_element_ids_path = cfg.case_path + '/gnn_outputs_distributed_gnn/gnn_outputs_poly_1' + '/node_element_ids_rank_%d_size_%d' %(i,data_read_world_size)
                # global_ids_path = cfg.case_path + '/gnn_outputs_distributed_gnn/gnn_outputs_poly_1' + '/global_ids_rank_%d_size_%d' %(i,data_read_world_size) 
                # pos_path = cfg.case_path + '/gnn_outputs_distributed_gnn/gnn_outputs_poly_1' + '/pos_node_rank_%d_size_%d' %(i,data_read_world_size) 

                if RANK == 0:
                    log.info('in get_pygeom_dataset...')

                train_dataset_temp, test_dataset_temp, data_mean, data_var, n_samples = ngs.get_pygeom_dataset_updated_statistics(
                                                                     data_x_path, 
                                                                     data_y_path,
                                                                     edge_index_path,
                                                                     node_element_ids_path,
                                                                     global_ids_path,
                                                                     pos_path,
                                                                     device_for_loading,
                                                                     fraction_valid)

                train_dataset += train_dataset_temp
                test_dataset += test_dataset_temp

                mean_x_i.append(data_mean[0])
                mean_y_i.append(data_mean[1])
                var_x_i.append(data_var[0])
                var_y_i.append(data_var[1])
                n_i.append(n_samples)

                # print('data_mean[0].shape: ', data_mean[0].shape)
                # print('data_var[0].shape: ', data_var[0].shape)


    # ~~~~ Get global training statistics 
    mean_x_i = np.stack(mean_x_i, axis=0)
    mean_y_i = np.stack(mean_y_i, axis=0)
    var_x_i = np.stack(var_x_i, axis=0)
    var_y_i = np.stack(var_y_i, axis=0)
    n_i = np.stack(n_i, axis=0).reshape(-1,1)

    # mean_x, std_x
    mean_x = np.sum(n_i * mean_x_i, axis=0) / np.sum(n_i)
    num_1 = np.sum(n_i * var_x_i, axis=0) # n_i * var_i
    num_2 = np.sum(n_i * (mean_x_i - mean_x)**2, axis=0) # n_i * (mean_i - global_mean)**2
    var_x = (num_1 + num_2)/np.sum(n_i)
    std_x = np.sqrt(var_x)

    # mean_y, std_y
    mean_y = np.sum(n_i * mean_y_i, axis=0) / np.sum(n_i)
    num_1 = np.sum(n_i * var_y_i, axis=0) # n_i * var_i
    num_2 = np.sum(n_i * (mean_y_i - mean_y)**2, axis=0) # n_i * (mean_i - global_mean)**2
    var_y = (num_1 + num_2)/np.sum(n_i)
    std_y = np.sqrt(var_y)

    # put into original format 
    data_mean = [torch.tensor(mean_x), torch.tensor(mean_y)]
    data_std = [torch.tensor(std_x), torch.tensor(std_y)]
   

    if RANK == 0:
        log.info('train dataset: %d elements' %(len(train_dataset)))
        log.info('valid dataset: %d elements' %(len(test_dataset)))
 
        mean_x = data_mean[0]
        mean_y = data_mean[1]
        std_x = data_std[0]
        std_y = data_std[1]
       
        print('\n\nmean_x: ', mean_x)
        print('\n\nmean_x.shape: ', mean_x.shape)
        print('\n\nmean_x type: ', type(mean_x))
        print('\n\nstd_x: ', std_x)
        print('\n\nstd_x.shape: ', std_x.shape)
        print('\n\nstd_x type: ', type(std_x))
        print('\n\nn_samples: ', np.sum(n_i))

        # print('\n\nmean_y: ', mean_y)
        # print('\n\nmean_y.shape: ', mean_y.shape)
        # print('\n\nmean_y type: ', type(mean_y))
        # print('\n\nstd_y: ', std_y)
        # print('\n\nstd_y.shape: ', std_y.shape)
        # print('\n\nstd_y type: ', type(std_y))

        # print('data_mean[0].shape: ', data_mean[0].shape)
        # print('data_std[0].shape: ', data_std[0].shape)

    # try torch.save 
    t_save = time.time()
    torch.save(train_dataset, cfg.data_dir + "train_dataset.pt")
    torch.save(test_dataset, cfg.data_dir + "valid_dataset.pt")
    torch.save(data_mean, cfg.data_dir + "data_mean.pt")
    torch.save(data_std, cfg.data_dir + "data_std.pt")
    t_save = time.time() - t_save 
    
    # load the dataset 
    t_load = time.time()
    train_dataset = torch.load(cfg.data_dir + "train_dataset.pt")
    test_dataset = torch.load(cfg.data_dir + "valid_dataset.pt")
    data_mean = torch.load(cfg.data_dir + "data_mean.pt")
    data_std = torch.load(cfg.data_dir + "data_std.pt")
    t_load = time.time() - t_load

    if RANK == 0:
        log.info('t_save: %g sec ' %(t_save))
        log.info('t_load: %g sec ' %(t_load))

    return 



@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices. Seed = %d' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count(), cfg.seed))

    if RANK == 0:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('INPUTS:')
        print(OmegaConf.to_yaml(cfg)) 
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    write_full_dataset(cfg)
    #train(cfg)
    #cleanup()


if __name__ == '__main__':
    main()