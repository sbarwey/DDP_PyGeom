"""
Prepares PyGeom data from BFS VTK files obtained from foamToVTK
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import os,time,sys 
import numpy as np
import pyvista as pv 

import torch 
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn
import torch_geometric.transforms as transforms


def get_pygeom_dataset_cell_data_radius(
        path_to_vtk : str, 
        path_to_ei : str, 
        path_to_ea : str, 
        path_to_pos : str, 
        device_for_loading : str, 
        features_to_keep : Optional[list] = None, 
        fraction_valid : Optional[float] = 0.1, 
        multiple_cases : Optional[bool] = False ) -> tuple[list,list]:
    print('Reading vtk: %s' %(path_to_vtk))
    mesh = pv.read(path_to_vtk)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Extracting data from vtk...')
    if multiple_cases:
        print('\tmultiple cases...')
        data_full_temp = []
        time_vec = []

        # Get the case file list 
        case_path_list = mesh.field_data['case_path_list']
        n_cases = len(case_path_list)
        for c in range(n_cases):
            data_full_c = np.array(mesh.cell_data['x_%d' %(c)]) # [N_nodes x (N_features x N_snaps)]
            time_vec_c = np.array(mesh.field_data['time_%d' %(c)])
            field_names = np.array(mesh.field_data['field_list'])
            n_cells = mesh.n_cells
            n_features = len(field_names)
            n_snaps = len(time_vec_c)
            data_full_c = np.reshape(data_full_c, (n_cells, n_features, n_snaps), order='F')
            data_full_temp.append(data_full_c)
            time_vec.append(time_vec_c)
            
        # Concatenate data_full_temp and time_vec 
        data_full_temp = np.concatenate(data_full_temp, axis=2)
        time_vec = np.concatenate(time_vec)
        n_snaps = len(time_vec)
    else:
        print('\tsingle case...')
        # Node features 
        data_full_temp = np.array(mesh.cell_data['x']) # [N_nodes x (N_features x N_snaps)]
        field_names = np.array(mesh.field_data['field_list'])
        time_vec = np.array(mesh.field_data['time'])
        n_cells = mesh.n_cells
        n_features = len(field_names)
        n_snaps = len(time_vec)
        data_full_temp = np.reshape(data_full_temp, (n_cells, n_features, n_snaps), order='F')

    # Do a dumb reshape
    data_full = np.zeros((n_snaps, n_cells, n_features), dtype=np.float32)
    for i in range(n_snaps):
        data_full[i,:,:] = data_full_temp[:,:,i]

    # Edge attributes and index, and node positions 
    print('Reading edge index and node positions...')
    edge_index = np.loadtxt(path_to_ei, dtype=np.long).T
    #edge_attr = np.loadtxt(path_to_ea, dtype=np.float32)
    pos = np.loadtxt(path_to_pos, dtype=np.float32)
  
    # Distance field
    distance = np.array(mesh.cell_data['distance'], dtype=np.float32)
    distance = np.reshape(distance, (n_cells, 1), order='F')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create radius graph
    # -- outputs are edge_index and edge_attr
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Creating radius graph...')
    # Create initial Knn
    pos = torch.tensor(pos)
    radius = 0.001 # m
    self_loops = False
    max_num_neighbors = 30 
    edge_index_rad = tgnn.radius_graph(pos, r=radius, max_num_neighbors=max_num_neighbors)

    # Concatenate
    edge_index = torch.concat((edge_index_rad, torch.tensor(edge_index)), axis=1)

    data_ref = Data( pos = pos, edge_index = edge_index )
    cart = transforms.Cartesian(norm=False, max_value = None, cat = False)
    dist = transforms.Distance(norm = False, max_value = None, cat = True)

    # populate edge_attr
    cart(data_ref) # adds cartesian/component-wise distance
    dist(data_ref) # adds euclidean distance

    # extract edge_attr
    edge_attr = data_ref.edge_attr

    # Eliminate duplicate edges
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # change back to numpy 
    pos = np.array(pos)
    edge_attr = np.array(edge_attr)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shuffle -- train/valid split
    # set aside 10% of the data for validation  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('arranging data for train/valid split...')
    print('\tvalidation size is %g * n_data' %(fraction_valid))
    np.random.seed(12345)

    # Shuffle 
    n_snaps = data_full.shape[0]
 
    # How many total snapshots to extract 
    n_full = n_snaps
    n_valid = int(fraction_valid * n_full)

    # Get validation set indices 
    idx_valid = np.sort(np.random.choice(n_full, n_valid, replace=False))

    # Get training set indices 
    idx_train = np.array(list(set(list(range(n_full))) - set(list(idx_valid))))

    # Train/test split 
    data_train = data_full[idx_train]
    data_valid = data_full[idx_valid]

    time_vec_train = time_vec[idx_train]
    time_vec_valid = time_vec[idx_valid]

    n_train = n_full - n_valid
    
    print('\tn_train: ', n_train)
    print('\tn_valid: ', n_valid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize node data: mean/std standardization  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    eps = 1e-10
    data_train_mean = data_train.mean(axis=(0,1), keepdims=True)
    data_train_std = data_train.std(axis=(0,1), keepdims=True)

    data_train = (data_train - data_train_mean)/(data_train_std + eps)
    data_valid = (data_valid - data_train_mean)/(data_train_std + eps)
  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize edge attributes 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    edge_attr_mean = edge_attr.mean()
    edge_attr_std = edge_attr.std()
    edge_attr = (edge_attr - edge_attr_mean)/(edge_attr_std + eps) 

    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Normalize distance field  
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # distance_mean = distance.mean()
    # distance_std = distance.std()
    # distance = (distance - distance_mean)/(distance_std + eps)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make pyGeom dataset 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data_train = torch.tensor(data_train)
    data_valid = torch.tensor(data_valid)
    time_vec_train = torch.tensor(time_vec_train)
    time_vec_valid = torch.tensor(time_vec_valid)
    
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    pos = torch.tensor(pos)
    distance = torch.tensor(distance)
    
    # Re-sort:  
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # Add self loops:
    #edge_index, edge_attr = utils.add_self_loops(edge_index, edge_attr, fill_value='mean')

    # Restrict data based on features_to_keep: 
    if features_to_keep == None:
        n_features = data_train[0].shape[-1]
        features_to_keep = list(range(n_features))
    data_train = data_train[:,:,features_to_keep]
    data_valid = data_valid[:,:,features_to_keep]
    data_train_mean = data_train_mean[:,:,features_to_keep]
    data_train_std = data_train_std[:,:,features_to_keep]

    data_train_mean = torch.tensor(data_train_mean)
    data_train_std = torch.tensor(data_train_std)


    # Training 
    data_train_list = []
    for i in range(n_train):
        #print('Train %d/%d' %(i+1, n_train))
        data_temp = Data(   x = data_train[i],
                            y = time_vec_train[i],
                            distance = distance, 
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos = pos,
                            data_scale = (data_train_mean, data_train_std), 
                            edge_scale = (edge_attr_mean, edge_attr_std), 
                            # distance_scale = (distance_mean, distance_std),
                            t = time_vec_train[i],
                            field_names = field_names)
        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)


    # Testing: 
    data_valid_list = []
    for i in range(n_valid):
        #print('Test %d/%d' %(i+1, n_valid))
        data_temp = Data(   x = data_valid[i],
                            y = time_vec_valid[i],
                            distance = distance,
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos = pos,
                            data_scale = (data_train_mean, data_train_std),
                            edge_scale = (edge_attr_mean, edge_attr_std), 
                            #distance_scale = (distance_mean, distance_std),
                            t = time_vec_valid[i],
                            field_names = field_names)
        data_temp = data_temp.to(device_for_loading)
        data_valid_list.append(data_temp)
    

    print('\n\tTraining samples: ', len(data_train_list))
    print('\tValidation samples: ', len(data_valid_list))
    print('\tN_nodes: ', data_train_list[0].x.shape[0])
    print('\tN_edges: ', data_train_list[0].edge_index.shape[1])
    print('\tN_features: ', data_train_list[0].x.shape[1])
    print('\n')

    return data_train_list, data_valid_list





