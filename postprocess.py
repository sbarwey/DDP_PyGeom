import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
import matplotlib.pyplot as plt
import dataprep.nekrs_graph_setup_tgv as ngs
import models.gnn as gnn
from pymech.neksuite import readnek,writenek
from pymech.dataset import open_dataset
from typing import Optional, Union, Callable, List, Tuple

seed = 122
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_grad_enabled(False)

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def get_edge_index(edge_index_path: str,
                   edge_index_vertex_path: Optional[str] = None) -> torch.Tensor:
    print('Loading edge index')
    edge_index = np.loadtxt(edge_index_path, dtype=np.int64).T
    if edge_index_vertex_path:
        print('Adding p1 connectivity...')
        print('\tEdge index shape before: ', edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=np.int64).T
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        print('\tEdge index shape after: ', edge_index.shape)
    edge_index = torch.tensor(edge_index)
    return edge_index 

if __name__ == "__main__":
    """
    Inference script: save predicted flowfield into .f file.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
    SRGNN_HOME = "/lus/eagle/projects/datascience/sbarwey/codes/aets_material/sr_gnn" # SRGNN_HOME path 
    model_path = f"{SRGNN_HOME}/python_codes/DDP_PyGeom_SR/saved_models/tgv_re_1600/GNN_TGV_3_7_132_128_3_2_2_True.tar" # model path
    use_residual = True # residual mode for inference. 
    n_element_neighbors = 6 # number of element neighbors to use for inference. 
    nrs_snap_dir = f"{SRGNN_HOME}/nekrs_cases/tgv/Re_1600_poly_7_dataset" # inference input/target snapshot directory
    t_str_list = ['00016', '00017'] # inference snapshot IDs to process 
    case_name = "newtgv" # inference case name header for the output .f files 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 

    # ~~~~ Create model 
    a = torch.load(model_path)
    input_dict = a['input_dict'] 
    input_node_channels = input_dict['input_node_channels']
    input_edge_channels_coarse = input_dict['input_edge_channels_coarse'] 
    input_edge_channels_fine = input_dict['input_edge_channels_fine'] 
    hidden_channels = input_dict['hidden_channels']
    output_node_channels = input_dict['output_node_channels']
    n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
    n_messagePassing_layers = input_dict['n_messagePassing_layers']
    use_fine_messagePassing = input_dict['use_fine_messagePassing']
    name = input_dict['name']

    model = gnn.GNN_Element_Neighbor_Lo_Hi(
            input_node_channels             = input_dict['input_node_channels'],
            input_edge_channels_coarse      = input_dict['input_edge_channels_coarse'],
            input_edge_channels_fine        = input_dict['input_edge_channels_fine'],
            hidden_channels                 = input_dict['hidden_channels'],
            output_node_channels            = input_dict['output_node_channels'],
            n_mlp_hidden_layers             = input_dict['n_mlp_hidden_layers'],
            n_messagePassing_layers         = input_dict['n_messagePassing_layers'],
            use_fine_messagePassing         = input_dict['use_fine_messagePassing'],
            name                            = input_dict['name'])

    def count_parameters(mdl):
        return sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"number of parameters in the model: {count_parameters(model)}")

    model.load_state_dict(a['state_dict'])
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model.to(device)
    model.eval()

    # ~~~~ Load eval and target snapshot 
    TORCH_FLOAT = torch.float32
    
    # Load in edge index 
    poly_lo = 1
    poly_hi = 7
    edge_index_path_lo = f"{nrs_snap_dir}/../gnn_outputs_poly_{poly_lo}/edge_index_element_local_rank_0_size_4"
    edge_index_path_hi = f"{nrs_snap_dir}/../gnn_outputs_poly_{poly_hi}/edge_index_element_local_rank_0_size_4"
    edge_index_lo = get_edge_index(edge_index_path_lo)
    edge_index_hi = get_edge_index(edge_index_path_hi)

    # Get full edge index 
    edge_index = edge_index_lo
    n_nodes_per_element = edge_index.max() + 1
    if n_element_neighbors > 0:
        node_max_per_element = edge_index.max()
        n_edges_per_element = edge_index.shape[1]
        edge_index_full = torch.zeros((2, n_edges_per_element*(n_element_neighbors+1)), dtype=edge_index.dtype)
        edge_index_full[:, :n_edges_per_element] = edge_index
        for i in range(1,n_element_neighbors+1):
            start = n_edges_per_element*i
            end = n_edges_per_element*(i+1)
            edge_index_full[:, start:end] = edge_index + (node_max_per_element+1)*i
        edge_index = edge_index_full

    for t_str in t_str_list:
        directory_path = nrs_snap_dir + f"/predictions/{model.get_save_header()}"
        if os.path.exists(directory_path + f"/{case_name}_pred0.f{t_str}"):
            print(directory_path + f"/{case_name}_pred0.f{t_str} \n already exists!!! Skipping...") 
            continue

        # One-shot
        xlo_field = readnek(nrs_snap_dir + f'/snapshots_coarse_{poly_hi}to{poly_lo}/{case_name}0.f{t_str}')
        xhi_field = readnek(nrs_snap_dir + f'/snapshots_target/{case_name}0.f{t_str}')
        #xhi_field = readnek(nrs_snap_dir + f'/snapshots_interp_{poly_lo}to{poly_hi}/{case_name}0.f{t_str}')
        xhi_field_pred = readnek(nrs_snap_dir + f'/snapshots_target/{case_name}0.f{t_str}')
        xhi_field_error = readnek(nrs_snap_dir + f'/snapshots_target/{case_name}0.f{t_str}')

        n_snaps = len(xlo_field.elem)

        # Get the element neighborhoods
        if n_element_neighbors > 0:
            Nelements = len(xlo_field.elem)
            pos_c = torch.zeros((Nelements, 3)) 
            for i in range(Nelements):
                pos_c[i] = torch.tensor(xlo_field.elem[i].centroid)
            edge_index_c = tgnn.knn_graph(x = pos_c, k = n_element_neighbors)

        # Get the element masks
        # Get the element masks
        central_element_mask = torch.concat(
                (torch.ones((n_nodes_per_element), dtype=torch.int64),
                    torch.zeros((n_nodes_per_element * n_element_neighbors), dtype=torch.int64))
                )
        central_element_mask = central_element_mask.to(torch.bool)

        with torch.no_grad():
            for i in range(n_snaps):
                print(f"Evaluating element {i}/{n_snaps}")
                
                pos_xlo_i = torch.tensor(xlo_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3]
                vel_xlo_i = torch.tensor(xlo_field.elem[i].vel).reshape((3, -1)).T
                pos_xhi_i = torch.tensor(xhi_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3]
                vel_xhi_i = torch.tensor(xhi_field.elem[i].vel).reshape((3, -1)).T

                x_gll = xhi_field.elem[i].pos[0,0,0,:]
                dx_min = x_gll[1] - x_gll[0]

                error_max = (pos_xlo_i.max(dim=0)[0] - pos_xhi_i.max(dim=0)[0]).max()
                error_min = (pos_xlo_i.min(dim=0)[0] - pos_xhi_i.min(dim=0)[0]).max()
                rel_error_max = torch.abs(error_max / dx_min)*100
                rel_error_min = torch.abs(error_min / dx_min)*100

                # Check positions 
                if (rel_error_max > 1e-2) or (rel_error_min > 1e-2):
                    print(f"Relative error in positions exceeds 0.01% in element i={i}.")
                    sys.exit()
                if (pos_xlo_i.max() == 0. and pos_xlo_i.min() == 0.):
                    print(f"Node positions are not stored in {data_xlo_path}.")
                    sys.exit()
                if (pos_xhi_i.max() == 0. and pos_xhi_i.min() == 0.):
                    print(f"Node positions are not stored in {data_xhi_path}.")
                    sys.exit()
                
                # get x_mean and x_std 
                x_mean_element_lo = torch.mean(vel_xlo_i, dim=0).unsqueeze(0).repeat(central_element_mask.shape[0], 1)
                x_std_element_lo = torch.std(vel_xlo_i, dim=0).unsqueeze(0).repeat(central_element_mask.shape[0], 1)
                x_mean_element_hi = torch.mean(vel_xlo_i, dim=0).unsqueeze(0).repeat(vel_xhi_i.shape[0], 1)
                x_std_element_hi = torch.std(vel_xlo_i, dim=0).unsqueeze(0).repeat(vel_xhi_i.shape[0], 1)

                # element lengthscale 
                lengthscale_element = torch.norm(pos_xlo_i.max(dim=0)[0] - pos_xlo_i.min(dim=0)[0], p=2)

                # node weight
                # nw = torch.ones((vel_xhi_i.shape[0], 1)) * node_weight

                # Get the element neighbors for the input
                if n_element_neighbors > 0:
                    send = edge_index_c[0,:]
                    recv = edge_index_c[1,:]
                    nbrs = send[recv == i]

                    pos_x_full = [pos_xlo_i]
                    vel_x_full = [vel_xlo_i]
                    for j in nbrs:
                        pos_x_full.append( torch.tensor(xlo_field.elem[j].pos).reshape((3, -1)).T )
                        vel_x_full.append( torch.tensor(xlo_field.elem[j].vel).reshape((3, -1)).T )
                    pos_x_full = torch.concat(pos_x_full)
                    vel_x_full = torch.concat(vel_x_full)

                    # reset pos
                    pos_xlo_i = pos_x_full
                    vel_xlo_i = vel_x_full

                # create data 
                data = ngs.DataLoHi( x = vel_xlo_i.to(dtype=TORCH_FLOAT),
                        y = vel_xhi_i.to(dtype=TORCH_FLOAT),
                        x_mean_lo = x_mean_element_lo.to(dtype=TORCH_FLOAT),
                        x_std_lo = x_std_element_lo.to(dtype=TORCH_FLOAT),
                        x_mean_hi = x_mean_element_hi.to(dtype=TORCH_FLOAT),
                        x_std_hi = x_std_element_hi.to(dtype=TORCH_FLOAT),
                        # node_weight = nw.to(dtype=TORCH_FLOAT),
                        L = lengthscale_element.to(dtype=TORCH_FLOAT),
                        pos_norm_lo = (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                        pos_norm_hi = (pos_xhi_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                        edge_index_lo = edge_index,
                        edge_index_hi = edge_index_hi,
                        central_element_mask = central_element_mask,
                        eid = torch.tensor(i))

                # for synchronizing across element boundaries
                if n_element_neighbors > 0:
                    batch = None
                    edge_index_coin = ngs.get_edge_index_coincident(
                            batch, data.pos_norm_lo, data.edge_index_lo)
                    degree = utils.degree(edge_index_coin[1,:], num_nodes = data.pos_norm_lo.shape[0])
                    degree += 1.
                    data.edge_index_coin = edge_index_coin
                    data.degree = degree
                else:
                    data.edge_index_coin = None
                    data.degree = None

                data = data.to(device)

                # ~~~~ Model evaluation ~~~~ # 
                with torch.no_grad():
                    # 1) Preprocessing: scale input  
                    eps = 1e-10
                    x_scaled = (data.x - data.x_mean_lo)/(data.x_std_lo + eps)

                    # 2) Evaluate model 
                    out_gnn = model(
                    x = x_scaled,
                    mask = data.central_element_mask,
                    edge_index_lo = data.edge_index_lo,
                    edge_index_hi = data.edge_index_hi,
                    pos_lo = data.pos_norm_lo,
                    pos_hi = data.pos_norm_hi,
                    #batch_lo = data.x_batch,
                    #batch_hi = data.y_batch,
                    edge_index_coin = data.edge_index_coin if n_element_neighbors>0 else None,
                    degree = data.degree if n_element_neighbors>0 else None)

                    # 3) set the target
                    if use_residual:
                        mask = data.central_element_mask
                        data.x_batch = data.edge_index_lo.new_zeros(data.pos_norm_lo.size(0))
                        data.y_batch = data.edge_index_hi.new_zeros(data.pos_norm_hi.size(0))
                        x_interp = tgnn.unpool.knn_interpolate(
                                x = data.x[mask,:],
                                pos_x = data.pos_norm_lo[mask,:],
                                pos_y = data.pos_norm_hi,
                                batch_x = data.x_batch[mask],
                                batch_y = data.y_batch,
                                k = 8)
                        # target = (data.y - x_interp)/(data.x_std_hi + eps)
                        # gnn = (data.y - x_interp)/(data.x_std_hi + eps)
                        # gnn * (data.x_std_hi + eps) = (data.y - x_interp)
                        # data.y = x_interp + gnn * (data.x_std_hi + eps)
                        y_pred = x_interp + out_gnn * (data.x_std_hi + eps)
                    else:
                        # target = (data.y - data.x_mean_hi)/(data.x_std_hi + eps)
                        # gnn = (data.y - data.x_mean_hi)/(data.x_std_hi + eps)
                        # gnn * (data.x_std_hi + eps) = data.y - data.x_mean_hi
                        # data.y = data.x_mean_hi + gnn * (data.x_std_hi + eps)
                        y_pred = data.x_mean_hi + out_gnn * (data.x_std_hi + eps)

                
                # ~~~~ Making the .f file ~~~~ # 
                # Re-shape the prediction, convert back to fp64 numpy 
                y_pred = y_pred.cpu()
                orig_shape = xhi_field.elem[i].vel.shape
                y_pred_rs = torch.reshape(y_pred.T, orig_shape).to(dtype=torch.float64).numpy()
                target = data.y.cpu()
                target_rs = torch.reshape(target.T, orig_shape).to(dtype=torch.float64).numpy()

                # Place prediction back in the snapshot data 
                xhi_field_pred.elem[i].vel[:,:,:,:] = y_pred_rs

                # Place error back in snapshot data 
                xhi_field_error.elem[i].vel[:,:,:,:] = target_rs - y_pred_rs 

                # Sanity check to make sure reshape is correct.
                # target_orig = xhi_field.elem[i].vel
                # err_sanity = target_orig - target_rs 
                
            # Write 
            print(f'Writing {t_str} prediction to {directory_path}...')
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created.")
            writenek(directory_path +  f"/{case_name}_pred0.f{t_str}", xhi_field_pred)
            writenek(directory_path +  f"/{case_name}_error0.f{t_str}", xhi_field_error)
            print(f'finished writing {t_str}')