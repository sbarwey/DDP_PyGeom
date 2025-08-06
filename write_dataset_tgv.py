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

import dataprep.nekrs_graph_setup_tgv as ngs

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

    SRGNN_HOME = "/lus/eagle/projects/datascience/sbarwey/codes/aets_material/sr_gnn"
    case_path = SRGNN_HOME + "/nekrs_cases/tgv"
    Re = '1600'
    snap_list = ['newtgv0.f00016', 'newtgv0.f00018', 'newtgv0.f00020']
    n_element_neighbors = 6
    for snap_id in range(len(snap_list)):
        snap = snap_list[snap_id]
        input_path = f"{case_path}/Re_{Re}_poly_7_dataset/snapshots_coarse_7to1/{snap}"
        target_path = f"{case_path}/Re_{Re}_poly_7_dataset/snapshots_target/{snap}"

        # element-local edge index 
        edge_index_path_lo = f"{case_path}/gnn_outputs_poly_1/edge_index_element_local_rank_0_size_4"
        edge_index_path_hi = f"{case_path}/gnn_outputs_poly_7/edge_index_element_local_rank_0_size_4"

        if RANK == 0:
                log.info('in get_pygeom_dataset...')

        train_dataset_temp, test_dataset_temp = ngs.get_pygeom_dataset_lo_hi_pymech(
                                data_xlo_path = input_path, 
                                data_xhi_path = target_path,
                                edge_index_path_lo = edge_index_path_lo,
                                edge_index_path_hi = edge_index_path_hi,
                                device_for_loading = device_for_loading,
                                fraction_valid = fraction_valid,
                                n_element_neighbors = n_element_neighbors)

        train_dataset += train_dataset_temp
        test_dataset += test_dataset_temp

    # Create data_dir if it doesn't exist
    data_dir = SRGNN_HOME + f"/python_codes/DDP_PyGeom_SR/datasets/tgv_nei_{n_element_neighbors}_re_{Re}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # try torch.save 
    t_save = time.time()
    torch.save(train_dataset, data_dir + f"/train_dataset.pt")
    torch.save(test_dataset, data_dir + f"/valid_dataset.pt")
    t_save = time.time() - t_save 
    
    # load the dataset 
    t_load = time.time()
    train_dataset = torch.load(data_dir + f"/train_dataset.pt")
    test_dataset = torch.load(data_dir + f"/valid_dataset.pt")
    t_load = time.time() - t_load

    if RANK == 0:
        log.info('t_save: %g sec ' %(t_save))
        log.info('t_load: %g sec ' %(t_load))

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    write_full_dataset(cfg)

if __name__ == '__main__':
    main()
