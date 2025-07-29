import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
import matplotlib.pyplot as plt
import dataprep.nekrs_graph_setup_bfs as ngs
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
    # ~~~~ Change model names: add "resid" to re3200 models 
    if 1 == 0:
        # Change re_3200 models to add residual 
        Re_str = "_re3200"
        Re_data = '1600'

        model_dir = "./saved_models/single_scale"

        model_list = os.listdir(model_dir)
        model_list = [s for s in model_list if "3200" in s]

        for model_path in model_list:
            a = torch.load(f"./saved_models/single_scale/{model_path}")
            input_dict = a['input_dict'] 
            name = input_dict['name']
            
            name_modified = name.split('re3200')[0]
            name_modified = name_modified + "resid_re3200"
            a['input_dict']['name'] = name_modified


            # Create model, and re-save it  
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

            model.load_state_dict(a['state_dict'])


            # save model 
            model_path = model_dir + "/"+ model.get_save_header() + ".tar"
            save_dict = {
                        'state_dict' : a['state_dict'],
                        'input_dict' : a['input_dict'],
                        'loss_hist_train' : a['loss_hist_train'], 
                        'loss_hist_test' : a['loss_hist_test'],
                        'lr_hist' : a['lr_hist'],
                        'training_iter' : a['training_iter']
                        }
            torch.save(save_dict, model_path)
            print(model_path)

    # ~~~~ [REVISION] POD Spectrum plots for Cavity
    if 1 == 0:
        snap = "newcavity0.f00011-PODSPECTRUM.npz"
        snap_pred = "newcavity_pred0.f00011-PODSPECTRUM.npz"
        case_path = "/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/cavity_rampup/Re_1600_p_7/one_shot"

        data_1600_7 = np.load(f"{case_path}/snapshots_target/{snap}")
        data_1600_1 = np.load(f"{case_path}/snapshots_coarse_7to1/{snap}")
        data_1600_7_nekrs = np.load(f"{case_path}/snapshots_interp_1to7/{snap}")

        # load models 
        cavity_gnn = np.load(f"{case_path}/predictions/cavity_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re1600_3_7_132_128_3_2_6_True/{snap_pred}")
        tgv_gnn = np.load(f"{case_path}/predictions/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_3_7_132_128_3_2_6_True/{snap_pred}")
        bfs_gnn = np.load(f"{case_path}/predictions/bfs_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re1600_3_7_132_128_3_2_6_True/{snap_pred}")
        tgv_ft_gnn = np.load(f"{case_path}/predictions/finetune_tgv2cav_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re1600_3_7_132_128_3_2_6_True/{snap_pred}")
        bfs_ft_gnn = np.load(f"{case_path}/predictions/finetune_bfs2cav_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re1600_3_7_132_128_3_2_6_True/{snap_pred}")

        plt.rcParams.update({'font.size': 16})
        lw = 2
        fig, ax = plt.subplots(2,1,figsize=(8,8),sharex=True)

        # Spectrum plot 
        ax[0].plot(data_1600_7['spectrum'], color='black', lw=lw, label='Target (P=7)')
        ax[0].plot(data_1600_1['spectrum'], color='blue', lw=lw, label='Coarse (P=1)')
        ax[0].plot(data_1600_7_nekrs['spectrum'], color='gray', lw=lw, label='SE Interp. (P=7)')
        ax[0].plot(cavity_gnn['spectrum'], color='lime', lw=lw, ls='--', label='GNN - Cavity')
        #ax[0].plot(bfs_gnn['spectrum'], color='red', lw=lw, ls='--', label='GNN - BFS')
        #ax[0].plot(bfs_ft_gnn['spectrum'], color='red', lw=lw, ls=':', label='GNN - BFS, FT')
        ax[0].plot(tgv_gnn['spectrum'], color='red', lw=lw, ls='--', label='GNN - TGV')
        ax[0].plot(tgv_ft_gnn['spectrum'], color='magenta', lw=lw, ls='--', label='GNN - TGV, FT')

        ax[0].set_yscale('log')
        ax[0].set_ylabel('$E_i$')
        ax[0].set_xlabel('Mode Index, i')
        #ax[0].set_title(snap)
        ax[0].grid(which='minor', alpha=0.1)
        ax[0].grid(which='major', alpha=0.4)
        ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 12})

        # Spectrum error plot 
        err_nekrs = np.abs(data_1600_7_nekrs['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_cavity_gnn = np.abs(cavity_gnn['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_bfs_gnn = np.abs(bfs_gnn['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_bfs_ft_gnn = np.abs(bfs_ft_gnn['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_tgv_gnn = np.abs(tgv_gnn['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_tgv_ft_gnn = np.abs(tgv_ft_gnn['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        ax[1].plot(err_nekrs, color='gray', lw=lw)
        ax[1].plot(err_cavity_gnn, color='lime', lw=lw, ls='--')
        #ax[1].plot(err_bfs_gnn, color='red', lw=lw, ls='--')
        #ax[1].plot(err_bfs_ft_gnn, color='red', lw=lw, ls=':')
        ax[1].plot(err_tgv_gnn, color='red', lw=lw, ls='--')
        ax[1].plot(err_tgv_ft_gnn, color='magenta', lw=lw, ls='--')

        #ax[1].set_xscale('log')
        #ax[1].set_yscale('log')
        ax[1].set_ylim([0, 1])
        ax[1].set_ylabel('Relative Error')
        ax[1].set_xlabel('Mode Index, i')
        ax[1].grid(which='minor', alpha=0.1)
        ax[1].grid(which='major', alpha=0.4)
        #ax[1].legend(fancybox=False, framealpha=1, edgecolor='black')

        x_positions = np.arange(0, len(data_1600_7['spectrum']), 1)  # Adjust as needed
        x_labels = [str(i) for i in x_positions]

        # Spectrum plot
        ax[0].set_xticks(x_positions)
        ax[0].set_xticklabels(x_labels)

        # Spectrum error plot
        ax[1].set_xticks(x_positions)
        ax[1].set_xticklabels(x_labels)

        plt.show(block=False)

    # ~~~~ [REVISION] POD Spectrum plots for BFS 
    if 1 == 0:
        snap = "newbfs0.f00011-PODSPECTRUM.npz"
        snap_pred = "newbfs_pred0.f00011-PODSPECTRUM.npz"
        case_path = "/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/bfs_2_rampup/Re_1600_p_7/one_shot"

        data_1600_7 = np.load(f"{case_path}/snapshots_target/{snap}")
        data_1600_1 = np.load(f"{case_path}/snapshots_coarse_7to1/{snap}")
        data_1600_7_nekrs = np.load(f"{case_path}/snapshots_interp_1to7/{snap}")

        # load models 
        resid = True
        n_mp = 6 
        data_1600_7_gnn_2_nei0 = np.load(f"{case_path}/predictions/bfs_sampled_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re1600_3_7_132_128_3_2_{n_mp}_True/{snap_pred}")
        data_1600_7_gnn_2_nei26 = np.load(f"{case_path}/predictions/bfs_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re1600_3_7_132_128_3_2_{n_mp}_True/{snap_pred}")
        
        plt.rcParams.update({'font.size': 16})
        lw = 2
        fig, ax = plt.subplots(2,1,figsize=(8,8),sharex=True)

        # Spectrum plot 
        ax[0].plot(data_1600_7['spectrum'], color='black', lw=lw, label='Target (P=7)')
        ax[0].plot(data_1600_1['spectrum'], color='blue', lw=lw, label='Coarse (P=1)')
        ax[0].plot(data_1600_7_nekrs['spectrum'], color='gray', lw=lw, label='SE Interp. (P=7)')
        ax[0].plot(data_1600_7_gnn_2_nei0['spectrum'], color='lime', lw=lw, ls='--', label='Model 2: Multi-Scale, 0 Neighbors (P=7)')
        ax[0].plot(data_1600_7_gnn_2_nei26['spectrum'], color='red', lw=lw, ls='--', label='Model 2: Multi-Scale, 26 Neighbors (P=7)')

        ax[0].set_yscale('log')
        ax[0].set_ylabel('$E_i$')
        ax[0].set_xlabel('Mode Index, i')
        #ax[0].set_title(snap)
        ax[0].grid(which='minor', alpha=0.1)
        ax[0].grid(which='major', alpha=0.4)
        ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 12})

        # Spectrum error plot 
        err_nekrs = np.abs(data_1600_7_nekrs['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_gnn_2_nei0 = np.abs(data_1600_7_gnn_2_nei0['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_gnn_2_nei26 = np.abs(data_1600_7_gnn_2_nei26['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        ax[1].plot(err_nekrs[:], color='gray', lw=lw, label='Spectral (P=7)')
        ax[1].plot(err_gnn_2_nei0[:], color='lime', lw=lw, ls='--', label='Model 2, 0 Neighbors (P=7)')
        ax[1].plot(err_gnn_2_nei26[:], color='red', lw=lw, ls='--', label='Model 2, 26 Neighbors (P=7)')

        #ax[1].set_xscale('log')
        #ax[1].set_yscale('log')
        ax[1].set_ylim([0, 1])
        ax[1].set_ylabel('Relative Error')
        ax[1].set_xlabel('Mode Index, i')
        ax[1].grid(which='minor', alpha=0.1)
        ax[1].grid(which='major', alpha=0.4)
        #ax[1].legend(fancybox=False, framealpha=1, edgecolor='black')

        x_positions = np.arange(0, len(data_1600_7['spectrum']), 1)  # Adjust as needed
        x_labels = [str(i) for i in x_positions]

        # Spectrum plot
        ax[0].set_xticks(x_positions)
        ax[0].set_xticklabels(x_labels)

        # Spectrum error plot
        ax[1].set_xticks(x_positions)
        ax[1].set_xticklabels(x_labels)

        plt.show(block=False)

    # ~~~~ Spectrum plots 
    if 1 == 0:
        
        # ~~~~ [REVISION] EFFECT OF INTERPOLATION 
        # nekrs interp : 
        t_snap = "21"
        data_nrs_1to7 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_interp_1to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_knn_1to7 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_knninterp_1to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_tgt_7 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_target/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_crs_7to1 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_coarse_7to1/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")

        # incr 
        data_nrs_5to7 = np.load(f"./outputs/Re_1600_poly_7_testset/incr/snapshots_interp_full_5to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_knn_5to7 = np.load(f"./outputs/Re_1600_poly_7_testset/incr/snapshots_knninterp_5to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_crs_3to1 = np.load(f"./outputs/Re_1600_poly_7_testset/incr/snapshots_coarse_3to1/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")

        # knn interp :
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(data_tgt_7['kspec'], data_tgt_7['spectrum'], color='black', lw=lw, label='Target')
        ax.plot(data_crs_7to1['kspec'], data_crs_7to1['spectrum'], color='blue', lw=lw, ls='-', label='Coarse (P=1)')
        #ax.plot(data_crs_3to1['kspec'], data_crs_3to1['spectrum'], color='gray', lw=lw, ls='-.', label='P=1 (from 3)')
        ax.plot(data_nrs_1to7['kspec'], data_nrs_1to7['spectrum'], color='gray', lw=lw, ls='-', label='SE Interp. (P=7)')
        #ax.plot(data_nrs_5to7['kspec'], data_nrs_5to7['spectrum'], color='cyan', lw=lw, ls='--', label='NekRS-incr')
        ax.plot(data_knn_1to7['kspec'], data_knn_1to7['spectrum'], color='magenta', lw=lw, ls='-', label='kNN Interp. (P=7)')
        #ax.plot(data_knn_5to7['kspec'], data_knn_5to7['spectrum'], color='magenta', lw=lw, ls='--', label='kNN-incr')

        # plot vlines: p = 1 
        ax.vlines(data_crs_7to1['nyq_size'],  1e-9, 1e-1, lw=lw, color='blue', zorder=-1, alpha=0.3)
        ax.vlines(data_tgt_7['nyq_size'],  1e-9, 1e-1, lw=lw, color='gray', zorder=-1, alpha=0.3)

        plt.show(block=False)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1,300])
        ax.set_ylim([1e-9, 1e-1])
        ax.set_ylabel('E(k)')
        ax.set_xlabel('k')
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.4)
        ax.legend(fancybox=False, framealpha=1, edgecolor='black')
        plt.show(block=False)

        asdf
        
        # EFFECT OF COARSENING -- FOR PAPER 
        data_1600_7 = np.load("./outputs/snapshots_for_plotting/snapshots_target/Re_1600/regtgv_reg0.f00002-SPECTRUM.npz")
        data_1600_1 = np.load("./outputs/snapshots_for_plotting/snapshots_coarse_7to1/Re_1600/regtgv_reg0.f00002-SPECTRUM.npz") 
        data_3200_7 = np.load("./outputs/snapshots_for_plotting/snapshots_target/Re_3200/regtgv_reg0.f00002-SPECTRUM.npz")
        data_3200_1 = np.load("./outputs/snapshots_for_plotting/snapshots_coarse_7to1/Re_3200/regtgv_reg0.f00002-SPECTRUM.npz") 

        plt.rcParams.update({'font.size': 16})
        lw = 2
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(data_1600_7['kspec'], data_1600_7['spectrum'], color='black', lw=lw, label='P=7 (DNS)')
        ax.plot(data_1600_1['kspec'], data_1600_1['spectrum'], color='blue', lw=lw, label='P=1 (Coarse)')

        ax.plot(data_3200_7['kspec'], data_3200_7['spectrum'], color='black', lw=lw, ls='--')
        ax.plot(data_3200_1['kspec'], data_3200_1['spectrum'], color='blue', lw=lw, ls='--')

        ax.vlines(data_1600_1['nyq_size'],  1e-9, 1e-1, lw=lw, color='blue', alpha=0.3, zorder=-1)
        ax.vlines(data_1600_7['nyq_size'],  1e-9, 1e-1, lw=lw, color='black', alpha=0.3, zorder=-1)

        plt.show(block=False)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1,300])
        ax.set_ylim([1e-9, 1e-1])
        ax.set_ylabel('E(k)')
        ax.set_xlabel('k')
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.4)
        #ax.legend(fancybox=False, framealpha=1, edgecolor='black')
        plt.show(block=False)


        # PREDICTIONS -- FOR PAPER 
        snap = "regtgv_reg0.f00021-SPECTRUM.npz"

        # # Trained on 1600, tested on 1600
        # Re_str_model = ""
        # Re_str = "1600"
        
        # Trained on 3200, tested on 3200 
        #Re_str_model = "_re3200"
        #Re_str = "3200"

        # # Trained on 1600, tested on 3200
        # Re_str_model = ""
        # Re_str = "3200"

        # Trained on 3200, tested on 1600
        Re_str_model = "_re3200"
        Re_str = "1600"


        data_1600_7 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_target/{snap}")
        data_1600_1 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_coarse_7to1/{snap}")
        data_1600_7_nekrs = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_interp_1to7/{snap}") 

        nei = 26
        resid = True
        
        # Load Model 1 (coarse-scale)
        n_mp = 12
        fine_mp = 'False'
        if not resid:
            modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        else:
            modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        print('coarse modelname: ', modelname)
        data_1600_7_gnn_1 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap}")

        # Load Model 2 (multiscale)
        n_mp = 6
        fine_mp = 'True'
        if not resid:
            modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        else:
            modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        print('multiscale modelname: ', modelname)
        data_1600_7_gnn_2 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap}")

        plt.rcParams.update({'font.size': 16})
        lw = 2
        fig, ax = plt.subplots(2,1,figsize=(6,8),sharex=True)

        # Spectrum plot 
        ax[0].plot(data_1600_7['kspec'], data_1600_7['spectrum'], color='black', lw=lw, label='DNS (P=7)')
        ax[0].plot(data_1600_1['kspec'], data_1600_1['spectrum'], color='blue', lw=lw, label='Coarse (P=1)')
        ax[0].plot(data_1600_7_nekrs['kspec'], data_1600_7_nekrs['spectrum'], color='gray', lw=lw, label='SE Interp. (P=7)')
        ax[0].plot(data_1600_7_gnn_1['kspec'], data_1600_7_gnn_1['spectrum'], color='red', lw=lw, ls='--', label='Model 1: Coarse-Scale (P=7)')
        ax[0].plot(data_1600_7_gnn_2['kspec'], data_1600_7_gnn_2['spectrum'], color='lime', lw=lw, ls='--', label='Model 2: Multi-Scale (P=7)')
        ax[0].vlines(data_1600_1['nyq_size'],  1e-9, 1e-1, lw=lw, color='blue', alpha=0.3, zorder=-1)
        ax[0].vlines(data_1600_7['nyq_size'],  1e-9, 1e-1, lw=lw, color='black', alpha=0.3, zorder=-1)

        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlim([1,300])
        #ax[0].set_xlim([2,data_1600_7['nyq_size'].item()])
        ax[0].set_ylim([1e-9, 1e-1])
        ax[0].set_ylabel('E(k)')
        ax[0].set_xlabel('k')
        #ax[0].set_title(snap)
        ax[0].grid(which='minor', alpha=0.1)
        ax[0].grid(which='major', alpha=0.4)
        #ax[0].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 12})

        # Spectrum error plot 
        err_nekrs = np.abs(data_1600_7_nekrs['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_gnn_1 = np.abs(data_1600_7_gnn_1['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        err_gnn_2 = np.abs(data_1600_7_gnn_2['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
        ax[1].plot(data_1600_7_nekrs['kspec'][1:], err_nekrs[1:], color='gray', lw=lw, label='Spectral (P=7)')
        ax[1].plot(data_1600_7_gnn_1['kspec'][1:], err_gnn_1[1:], color='red', lw=lw, ls='--', label='Model 1 (P=7)')
        ax[1].plot(data_1600_7_gnn_2['kspec'][1:], err_gnn_2[1:], color='lime', lw=lw, ls='--', label='Model 2 (P=7)')
        ax[1].vlines(data_1600_1['nyq_size'],  0, 1e2, lw=lw, color='blue', alpha=0.3, zorder=-1)
        ax[1].vlines(data_1600_7['nyq_size'],  0, 1e2, lw=lw, color='black', alpha=0.3, zorder=-1)

        #ax[1].set_xscale('log')
        #ax[1].set_yscale('log')
        ax[1].set_ylim([0, 1])
        ax[1].set_ylabel('Relative Error')
        ax[1].set_xlabel('k')
        ax[1].grid(which='minor', alpha=0.1)
        ax[1].grid(which='major', alpha=0.4)
        #ax[1].legend(fancybox=False, framealpha=1, edgecolor='black')

        plt.show(block=False)

    # ~~~~ Error versus number of neighbors plot 
    if 1 == 0:
        # PREDICTIONS -- FOR PAPER 
        snap = "regtgv_reg0.f00021-SPECTRUM.npz"

        #Re_str = "1600"
        #Re_str_model = ""
         
        Re_str = "3200"
        Re_str_model = "_re3200"

        data_1600_7 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_target/{snap}")
        data_1600_1 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_coarse_7to1/{snap}")
        data_1600_7_nekrs = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_interp_1to7/{snap}") 
        
        lim_1 = data_1600_1['nyq_size']
        lim_7 = data_1600_7['nyq_size']
        nei_list = [0, 6, 26]
        resid = True
        
        # Error curves
        model_1_error_0_1_mean = []
        model_1_error_0_1_min = []
        model_1_error_0_1_max = []
        model_1_error_1_7_mean = []
        model_1_error_1_7_min = []
        model_1_error_1_7_max = []

        model_2_error_0_1_mean = []
        model_2_error_0_1_min = []
        model_2_error_0_1_max = []
        model_2_error_1_7_mean = []
        model_2_error_1_7_min = []
        model_2_error_1_7_max = []

        for nei in nei_list:

            # Load Model 1 (coarse-scale)
            n_mp = 12
            fine_mp = 'False'
            if not resid:
                modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
            else:
                modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
            data_1600_7_gnn_1 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap}")

            # Load Model 2 (multiscale)
            n_mp = 6
            fine_mp = 'True'
            if not resid:
                modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
            else:
                modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
            data_1600_7_gnn_2 = np.load(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap}")

            err_gnn_1 = np.abs(data_1600_7_gnn_1['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']
            err_gnn_2 = np.abs(data_1600_7_gnn_2['spectrum'] - data_1600_7['spectrum'])/data_1600_7['spectrum']

            k_full = data_1600_7['kspec']
            k_0_1 = k_full[1:int(lim_1 + 1)]
            k_1_7 = k_full[int(lim_1 + 1):int(lim_7 + 1)]

            m1_err_0_1 = err_gnn_1[1:int(lim_1 + 1)]
            m1_err_1_7 = err_gnn_1[int(lim_1 + 1):int(lim_7 + 1)]

            m2_err_0_1 = err_gnn_2[1:int(lim_1 + 1)]
            m2_err_1_7 = err_gnn_2[int(lim_1 + 1):int(lim_7 + 1)]

            model_1_error_0_1_mean.append(m1_err_0_1.mean()) 
            model_1_error_0_1_max.append(m1_err_0_1.max()) 
            model_1_error_0_1_min.append(m1_err_0_1.min()) 

            model_1_error_1_7_mean.append(m1_err_1_7.mean()) 
            model_1_error_1_7_max.append(m1_err_1_7.max()) 
            model_1_error_1_7_min.append(m1_err_1_7.min()) 

            model_2_error_0_1_mean.append(m2_err_0_1.mean()) 
            model_2_error_0_1_max.append(m2_err_0_1.max()) 
            model_2_error_0_1_min.append(m2_err_0_1.min()) 

            model_2_error_1_7_mean.append(m2_err_1_7.mean()) 
            model_2_error_1_7_max.append(m2_err_1_7.max()) 
            model_2_error_1_7_min.append(m2_err_1_7.min()) 
        
        ms = 8
        lw = 1.
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))
        ax[0].plot(nei_list, model_1_error_0_1_mean, color='black', lw=lw, marker='o', ms=ms)
        ax[0].plot(nei_list, model_1_error_0_1_min, color='black', lw=lw, ls='--', marker='o', ms=ms)
        ax[0].plot(nei_list, model_1_error_0_1_max, color='black', lw=lw, ls='-.', marker='o', ms=ms)
        ax[0].plot(nei_list, model_2_error_0_1_mean, color='blue', lw=lw, marker='s', ms=ms)
        ax[0].plot(nei_list, model_2_error_0_1_min, color='blue', lw=lw, ls='--', marker='s', ms=ms)
        ax[0].plot(nei_list, model_2_error_0_1_max, color='blue', lw=lw, ls='-.', marker='s', ms=ms)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Relative Error in Spectrum')
        ax[0].set_xlabel('Coarse Element Neighbors')

        ax[1].plot(nei_list, model_1_error_1_7_mean, color='black', lw=lw, marker='o', ms=ms)
        ax[1].plot(nei_list, model_1_error_1_7_min, color='black', lw=lw, ls='--', marker='o', ms=ms)
        ax[1].plot(nei_list, model_1_error_1_7_max, color='black', lw=lw, ls='-.', marker='o', ms=ms)
        ax[1].plot(nei_list, model_2_error_1_7_mean, color='blue', lw=lw, marker='s', ms=ms)
        ax[1].plot(nei_list, model_2_error_1_7_min, color='blue', lw=lw, ls='--', marker='s', ms=ms)
        ax[1].plot(nei_list, model_2_error_1_7_max, color='blue', lw=lw, ls='-.', marker='s', ms=ms)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Relative Error in Spectrum')
        ax[1].set_xlabel('Coarse Element Neighbors')

        plt.show(block=False)


    # ~~~~ MSE over all elements (FOR PAPER) 
    if 1 == 0:
        # WRITE: 
        # ~~~~ # snap_list = ["newtgv0.f00016", "newtgv0.f00017", "newtgv0.f00018",
        # ~~~~ #              "newtgv0.f00019", "newtgv0.f00020", "newtgv0.f00021"]
        # ~~~~ # snap_gnn_list = ["newtgv_pred0.f00016", "newtgv_pred0.f00017", "newtgv_pred0.f00018",
        # ~~~~ #                  "newtgv_pred0.f00019", "newtgv_pred0.f00020", "newtgv_pred0.f00021"]


        # ~~~~ # n_snaps = len(snap_list)

        # ~~~~ # # Trained on Re=1600, eval on Re=1600
        # ~~~~ # #Re_str_model = ""
        # ~~~~ # #Re_str = "1600"
        # ~~~~ # 
        # ~~~~ # ## Trained on Re=3200, eval on Re=3200
        # ~~~~ # #Re_str_model = "_re3200"
        # ~~~~ # #Re_str = "3200"
        # ~~~~ # 
        # ~~~~ # # # Trained on Re=1600, eval on Re=3200
        # ~~~~ # # Re_str_model = ""
        # ~~~~ # # Re_str = "3200"

        # ~~~~ # # Trained on Re=3200, eval on Re=1600
        # ~~~~ # Re_str_model = "_re3200"
        # ~~~~ # Re_str = "1600"

        # ~~~~ # # ~~~~ GNNs 
        # ~~~~ # nei = 26
        # ~~~~ # resid = True
    
        # ~~~~ # mse_spectral = np.zeros(n_snaps)
        # ~~~~ # mse_gnn_1 = np.zeros(n_snaps)
        # ~~~~ # mse_gnn_2 = np.zeros(n_snaps)

        # ~~~~ # for s in range(n_snaps):

        # ~~~~ #     snap = snap_list[s]
        # ~~~~ #     snap_gnn = snap_gnn_list[s]

        # ~~~~ #     print(f"Snap: {snap}")

        # ~~~~ #     # Target 
        # ~~~~ #     y_7 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_target/{snap}")
        # ~~~~ #     n_snaps = len(y_7.elem)

        # ~~~~ #     # Input
        # ~~~~ #     y_1 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_coarse_7to1/{snap}")

        # ~~~~ #     # nekrs spectral interp
        # ~~~~ #     y_spectral = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_interp_1to7/{snap}") 

        # ~~~~ #     # Load Model 1 (coarse-scale)
        # ~~~~ #     n_mp = 12
        # ~~~~ #     fine_mp = 'False'
        # ~~~~ #     if not resid:
        # ~~~~ #         modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ #     else:
        # ~~~~ #         modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ #     print('coarse modelname: ', modelname)
        # ~~~~ #     y_gnn_1 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap_gnn}")

        # ~~~~ #     # Load Model 2 (multiscale)
        # ~~~~ #     n_mp = 6
        # ~~~~ #     fine_mp = 'True'
        # ~~~~ #     if not resid:
        # ~~~~ #         modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ #     else:
        # ~~~~ #         modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ #     print('multiscale modelname: ', modelname)
        # ~~~~ #     y_gnn_2 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap_gnn}")

        # ~~~~ #     # element loop 
        # ~~~~ #     error_spectral = np.zeros((n_snaps,3))
        # ~~~~ #     error_gnn_1 = np.zeros((n_snaps,3))
        # ~~~~ #     error_gnn_2 = np.zeros((n_snaps,3))

        # ~~~~ #     print('processing....')
        # ~~~~ #     for i in range(n_snaps):
        # ~~~~ #         # print("processing element ", i)
        # ~~~~ #         # pos_7 = (y_7.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        # ~~~~ #         # pos_1 = (y_1.elem[i].pos).reshape((3, -1)).T
        # ~~~~ #         # pos_spectral = (y_spectral.elem[i].pos).reshape((3, -1)).T
        # ~~~~ #         # pos_gnn_1 = (y_gnn_1.elem[i].pos).reshape((3, -1)).T
        # ~~~~ #         # pos_gnn_2 = (y_gnn_2.elem[i].pos).reshape((3, -1)).T

        # ~~~~ #         vel_7 = (y_7.elem[i].vel).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        # ~~~~ #         vel_1 = (y_1.elem[i].vel).reshape((3, -1)).T
        # ~~~~ #         vel_spectral = (y_spectral.elem[i].vel).reshape((3, -1)).T
        # ~~~~ #         vel_gnn_1 = (y_gnn_1.elem[i].vel).reshape((3, -1)).T
        # ~~~~ #         vel_gnn_2 = (y_gnn_2.elem[i].vel).reshape((3, -1)).T

        # ~~~~ #         error_spectral[i] = np.mean((vel_spectral - vel_7)**2, axis=0)
        # ~~~~ #         error_gnn_1[i] = np.mean((vel_gnn_1 - vel_7)**2, axis=0)
        # ~~~~ #         error_gnn_2[i] = np.mean((vel_gnn_2 - vel_7)**2, axis=0)
        # ~~~~ #   
        # ~~~~ #     mse_spectral[s] = np.mean(error_spectral)
        # ~~~~ #     mse_gnn_1[s] = np.mean(error_gnn_1)
        # ~~~~ #     mse_gnn_2[s] = np.mean(error_gnn_2)

        # ~~~~ # np.savez(f"./outputs/mse_global_train_3200_eval_1600/std_vs_error_data_nei_{Re_str}_{nei}.npz", mse_spectral = mse_spectral, mse_gnn_1 = mse_gnn_1, mse_gnn_2 = mse_gnn_2)

        # ~~~~ # # READ -- NO RE EXTRAP: 
        # ~~~~ # Re_str = '1600'
        # ~~~~ # data_1600_0  = np.load(f"./outputs/mse_global/std_vs_error_data_nei_{Re_str}_0.npz")
        # ~~~~ # data_1600_6  = np.load(f"./outputs/mse_global/std_vs_error_data_nei_{Re_str}_6.npz")
        # ~~~~ # data_1600_26 = np.load(f"./outputs/mse_global/std_vs_error_data_nei_{Re_str}_26.npz")

        # ~~~~ # t = np.array([1,2,3,4,5,6])

        # ~~~~ # t = np.array([1,2,3])
        # ~~~~ # idx_plot = [1,3,5]
        # ~~~~ # #idx_plot = [0,2,4]
        # ~~~~ # width = 0.1
        # ~~~~ # 
        # ~~~~ # # Bar chart 
        # ~~~~ # plt.rcParams.update({'font.size': 16})
        # ~~~~ # fig, ax = plt.subplots(figsize=(6,4))

        # ~~~~ # # Model 1:
        # ~~~~ # ax.bar(t + 0*width, data_1600_0['mse_gnn_1'][idx_plot],  width, color='silver')
        # ~~~~ # ax.bar(t + 1*width, data_1600_6['mse_gnn_1'][idx_plot],  width, color='skyblue')
        # ~~~~ # ax.bar(t + 2*width, data_1600_26['mse_gnn_1'][idx_plot], width, color='tomato')

        # ~~~~ # # Model 2:
        # ~~~~ # ax.bar(t + 3*width + 0.05, data_1600_0['mse_gnn_2'][idx_plot],  width, color='silver', hatch="/")
        # ~~~~ # ax.bar(t + 4*width + 0.05, data_1600_6['mse_gnn_2'][idx_plot],  width, color='skyblue', hatch="/")
        # ~~~~ # ax.bar(t + 5*width + 0.05, data_1600_26['mse_gnn_2'][idx_plot], width, color='tomato', hatch="/")

        # ~~~~ # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ~~~~ # ax.set_ylabel('Mean-Squared Error')
        # ~~~~ # #ax.set_yscale('log')
        # ~~~~ # ax.set_ylim([0, 1e-2])
        # ~~~~ # ax.set_xticks([])
        # ~~~~ # ax.grid(False)

        # ~~~~ # plt.show(block=False)

        # READ -- RE EXTRAP: 
        Re_str_train = '1600'
        Re_str_eval = '3200'
        data_1600_0  = np.load(f"./outputs/mse_global_train_{Re_str_train}_eval_{Re_str_eval}/std_vs_error_data_nei_{Re_str_eval}_0.npz")
        data_1600_6  = np.load(f"./outputs/mse_global_train_{Re_str_train}_eval_{Re_str_eval}/std_vs_error_data_nei_{Re_str_eval}_6.npz")
        data_1600_26  = np.load(f"./outputs/mse_global_train_{Re_str_train}_eval_{Re_str_eval}/std_vs_error_data_nei_{Re_str_eval}_26.npz")


        t = np.array([1,2,3])
        idx_plot = [1,3,5]

        t = np.array([1,2,3,4,5,6])
        idx_plot = list(range(6))

        width = 0.1
        
        # Bar chart 
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(9,4))

        # Model 1:
        ax.bar(t + 0*width, data_1600_0['mse_gnn_1'][idx_plot],  width, color='silver')
        ax.bar(t + 1*width, data_1600_6['mse_gnn_1'][idx_plot],  width, color='skyblue')
        ax.bar(t + 2*width, data_1600_26['mse_gnn_1'][idx_plot], width, color='tomato')

        # Model 2:
        ax.bar(t + 3*width + 0.05, data_1600_0['mse_gnn_2'][idx_plot],  width, color='silver', hatch="/")
        ax.bar(t + 4*width + 0.05, data_1600_6['mse_gnn_2'][idx_plot],  width, color='skyblue', hatch="/")
        ax.bar(t + 5*width + 0.05, data_1600_26['mse_gnn_2'][idx_plot], width, color='tomato', hatch="/")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Mean-Squared Error')
        ax.set_ylim([0, 1e-2])
        ax.set_xticks([])
        ax.grid(False)

        #ax.set_yscale('log')
        #ax.set_ylim([1e-5, 1e-2])
        plt.show(block=False)


    # ~~~~ Scatter plots: RMS velocity vs GNN prediction error (FOR PAPER) 
    if 1 == 0:
        # PREDICTIONS -- FOR PAPER 
        snap = "newtgv0.f00021"
        snap_gnn = "newtgv_pred0.f00021"

        #Re_str = "1600"
        #Re_str_model = ""
         
        Re_str = "3200"
        Re_str_model = "_re3200"

        # # Target 
        # y_7 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_target/{snap}")
        # n_snaps = len(y_7.elem)

        # # Input
        # y_1 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_coarse_7to1/{snap}")

        # # nekrs spectral interp
        # y_spectral = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/snapshots_interp_1to7/{snap}") 


        # ~~~~ GNNs 
        nei = 6
        resid = True
        
        # ~~~~ # # Load Model 1 (coarse-scale)
        # ~~~~ # n_mp = 12
        # ~~~~ # fine_mp = 'False'
        # ~~~~ # if not resid:
        # ~~~~ #     modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ # else:
        # ~~~~ #     modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ # print('coarse modelname: ', modelname)
        # ~~~~ # y_gnn_1 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap_gnn}")

        # ~~~~ # # Load Model 2 (multiscale)
        # ~~~~ # n_mp = 6
        # ~~~~ # fine_mp = 'True'
        # ~~~~ # if not resid:
        # ~~~~ #     modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ # else:
        # ~~~~ #     modelname = f"gnn_lr_1em4_bs_4_nei_{nei}_c2f_multisnap_resid{Re_str_model}_3_7_132_128_3_2_{n_mp}_{fine_mp}"
        # ~~~~ # print('multiscale modelname: ', modelname)
        # ~~~~ # y_gnn_2 = readnek(f"./outputs/Re_{Re_str}_poly_7_testset/one_shot/predictions/{modelname}/{snap_gnn}")

        # ~~~~ # # element loop 
        # ~~~~ # std_coarse = np.zeros((n_snaps,3))
        # ~~~~ # error_spectral = np.zeros((n_snaps,3))
        # ~~~~ # error_gnn_1 = np.zeros((n_snaps,3))
        # ~~~~ # error_gnn_2 = np.zeros((n_snaps,3))

        # ~~~~ # print('processing....')
        # ~~~~ # for i in range(n_snaps):
        # ~~~~ #     # print("processing element ", i)
        # ~~~~ #     # pos_7 = (y_7.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        # ~~~~ #     # pos_1 = (y_1.elem[i].pos).reshape((3, -1)).T
        # ~~~~ #     # pos_spectral = (y_spectral.elem[i].pos).reshape((3, -1)).T
        # ~~~~ #     # pos_gnn_1 = (y_gnn_1.elem[i].pos).reshape((3, -1)).T
        # ~~~~ #     # pos_gnn_2 = (y_gnn_2.elem[i].pos).reshape((3, -1)).T

        # ~~~~ #     vel_7 = (y_7.elem[i].vel).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        # ~~~~ #     vel_1 = (y_1.elem[i].vel).reshape((3, -1)).T
        # ~~~~ #     vel_spectral = (y_spectral.elem[i].vel).reshape((3, -1)).T
        # ~~~~ #     vel_gnn_1 = (y_gnn_1.elem[i].vel).reshape((3, -1)).T
        # ~~~~ #     vel_gnn_2 = (y_gnn_2.elem[i].vel).reshape((3, -1)).T

        # ~~~~ #     std_coarse[i] = vel_1.std(axis=0)
        # ~~~~ #     error_spectral[i] = np.mean((vel_spectral - vel_7)**2, axis=0)
        # ~~~~ #     error_gnn_1[i] = np.mean((vel_gnn_1 - vel_7)**2, axis=0)
        # ~~~~ #     error_gnn_2[i] = np.mean((vel_gnn_2 - vel_7)**2, axis=0)
       
        # ~~~~ # # Save: 
        # ~~~~ # # np.savez(f"./outputs/std_vs_error_data_nei_{Re_str}_{nei}.npz", error_gnn_1=error_gnn_1, error_gnn_2=error_gnn_2, std_coarse=std_coarse)

        # Load:  
        data = np.load(f"./outputs/std_vs_error/std_vs_error_data_nei_{Re_str}_{nei}.npz")
        error_gnn_1 = data['error_gnn_1']
        error_gnn_2 = data['error_gnn_2']
        std_coarse = data['std_coarse']

        # Plot 
        #ms = 1 # for zoomed-out plot 
        ms = 5 # for zoomed-in plot
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        for c in range(3):
            #ax[c].scatter(error_spectral[:,c], std_coarse[:,c], color='gray', s=ms, label='SE Interp')
            ax[c].scatter(error_gnn_1[:,c], std_coarse[:,c]   , color='black', s=ms, label='Model 1: Coarse-Scale')
            ax[c].scatter(error_gnn_2[:,c], std_coarse[:,c]   , color='red', s=ms, label='Model 2: Multi-Scale')
            ax[c].grid(False)
            ax[c].set_xlabel('Mean-Squared Error')
            ax[c].set_ylabel('Input Element Standard Deviation')
            #if c == 0: ax[c].legend(fancybox=False, framealpha=1, edgecolor='black', prop={'size': 12})
            ax[c].set_xscale('log')
            ax[c].set_xlim([1e-8, 5e-1])
            ax[c].set_ylim([0,0.4])
        plt.show(block=False)
        
        pass 

    # ~~~~ 1d flow plots: Ux versus x (FOR PAPER)
    if 1 == 0:
        
        # P = 7: z = pi/2 is 215 
        file_target = "./outputs/snapshots_for_plotting/sec_4/snapshots_target/Re_1600/regtgv_reg0.f00002"
        file_coarse = "./outputs/snapshots_for_plotting/sec_4/snapshots_coarse_7to1/Re_1600/regtgv_reg0.f00002"
        file_interp = "./outputs/snapshots_for_plotting/sec_4/snapshots_interp_1to7/Re_1600/regtgv_reg0.f00002"
        file_model1 = "./outputs/snapshots_for_plotting/sec_4/snapshots_model_1_nei_26/Re_1600/regtgv_reg0.f00002"
        file_model2 = "./outputs/snapshots_for_plotting/sec_4/snapshots_model_2_nei_26/Re_1600/regtgv_reg0.f00002"

        file_list = [file_target, file_coarse, file_interp, file_model1, file_model2]

        # Load datasets
        ds_list = []
        for file in file_list:
            print(file)
            ds = open_dataset(file)
            ds_list.append(ds)

        # Get data
        # y - 0.1 [USE] 
        # y - 0.9 [USE]
        # y - 2 [USE]
        x_list = []
        u_list = []
        i = 0
        for file in file_list:
            print(file)
            ds = ds_list[i] 
            U = ds.ux.data
            V = ds.ux.data
            W = ds.ux.data
            U_mag = np.sqrt(U**2 + V**2 + W**2)
            x = ds.x.data
            y = ds.y.data
            z = ds.z.data
            
            z_idx = np.abs(z - np.pi/2).argmin()
            y_idx = np.abs(y - 0.9).argmin()
            print('z: ', z[z_idx])
            print('y: ', y[y_idx])

            x_list.append(x)
            u_list.append(U_mag[:,y_idx,z_idx])
            i+=1

        # Plot
        ms=0
        lw=1.5
        mew=0
        fig, ax = plt.subplots()
        ax.plot(x_list[0], u_list[0], ls='-', lw=lw, marker='o', ms=ms, mew=mew, fillstyle="none", label="target", color='black')
        #ax.plot(x_list[1], u_list[1], ls='-', lw=lw, marker='s', ms=ms, mew=mew, fillstyle="none", label="coarse")
        ax.plot(x_list[2], u_list[2], ls='-', lw=lw, marker='o', ms=ms, mew=mew, fillstyle="none", label="interp", color='gray')
        ax.plot(x_list[3], u_list[3], ls='--', lw=lw, marker='o', ms=ms, mew=mew, fillstyle="none", label="model1", color='red')
        ax.plot(x_list[4], u_list[4], ls='--', lw=lw, marker='o', ms=ms, mew=mew, fillstyle="none", label="model2", color='blue')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        #ax.set_ylim([u_list[0].min(), u_list[0].max()])
        #ax.set_yscale('log')
        ax.grid(False)
        plt.show(block=False)

        pass


    # ~~~~ postprocessing: training losses (FOR PAPER) 
    if 1 == 0:
        modelpath = "./saved_models/single_scale"

        #re="_re3200"
        re=""

        # Model 1: 
        n_mp = 12
        fine_mp = 'False'
        m1_0 = torch.load(f'{modelpath}/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid{re}_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        m1_6 = torch.load(f'{modelpath}/gnn_lr_1em4_bs_4_nei_6_c2f_multisnap_resid{re}_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        m1_26 = torch.load(f'{modelpath}/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid{re}_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')

        # Model 2:
        n_mp = 6
        fine_mp = 'True'
        m2_0 = torch.load(f'{modelpath}/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid{re}_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        m2_6 = torch.load(f'{modelpath}/gnn_lr_1em4_bs_4_nei_6_c2f_multisnap_resid{re}_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        m2_26 = torch.load(f'{modelpath}/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid{re}_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=(8,6))

        # model 1 
        ax.plot(m1_0['loss_hist_train'][0:], lw=2, color="black", label="M1-0", ls="-")
        ax.plot(m1_6['loss_hist_train'][0:], lw=2, color="black", label="M1-6", ls="--")
        ax.plot(m1_26['loss_hist_train'][0:], lw=2, color="black", label="M1-26", ls="-.")
        # model 2 
        ax.plot(m2_0['loss_hist_train'][0:], lw=2, color="red", label="M2-0", ls="-")
        ax.plot(m2_6['loss_hist_train'][0:], lw=2, color="red", label="M2-6", ls="--")
        ax.plot(m2_26['loss_hist_train'][0:], lw=2, color="red", label="M2-26", ls="-.")

        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)


        

        pass


    # ~~~~ postprocessing: training losses -- comparing a set of different models 
    if 1 == 0:
        n_mp = 6
        fine_mp = 'True'

        # one-shot -- single-scale 
        a = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = '1shot+0nei'
        a_color = 'black'
        a_ls = '-'

        # incremental - singlescale 
        b = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_6_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = '1shot+6nei'
        b_color = 'blue'
        b_ls = '-'

        # fine-scale neighbors -- single-scale 
        c = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        c_label = '1shot+26nei'
        c_color = 'red'
        c_ls = '-'

        # coarse-scale neighbors -- single-scale 
        d = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        d_label = '1shot+0nei, Resid'
        d_color = 'lime'
        d_ls = '-'

        # coarse-scale neighbors -- single-scale -- no resid 
        d2 = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        d2_label = '1shot+0nei'
        d2_color = 'green'
        d2_ls = '-'

        # coarse-scale neighbors -- single-scale 
        e = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_6_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        e_label = '1shot+6nei, Resid'
        e_color = 'gray'
        e_ls = '-'
    
        # coarse-scale neighbors -- single-scale 
        f = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        f_label = '1shot+26nei, Resid'
        f_color = 'magenta'
        f_ls = '-'

        # Incremental, 0 nei
        g = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_incr_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        g_label = 'Incr+0nei, Resid'
        g_color = 'orange'
        g_ls = '-.'


        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        #ax.plot(a['loss_hist_train'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        #ax.plot(b['loss_hist_train'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)
        #ax.plot(c['loss_hist_train'][0:], lw=2, color=c_color, label=c_label, ls=c_ls)
        ax.plot(d['loss_hist_train'][0:], lw=2, color=d_color, label=d_label, ls=d_ls)
        ax.plot(d2['loss_hist_train'][0:], lw=2, color=d2_color, label=d2_label, ls=d2_ls)
        ax.plot(e['loss_hist_train'][0:], lw=2, color=e_color, label=e_label, ls=e_ls)
        ax.plot(f['loss_hist_train'][0:], lw=2, color=f_color, label=f_label, ls=f_ls)
        ax.plot(g['loss_hist_train'][0:], lw=2, color=g_color, label=g_label, ls=g_ls)

        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)

    # ~~~~ postprocessing: training losses -- comparing effect of batch size  
    if 1 == 0:
        mp = 6 

        # one-shot -- multiscale 
        a = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_32_multisnap_3_7_128_3_2_6.tar')
        a_label = '8x32'
        b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_64_multisnap_3_7_128_3_2_6.tar')
        b_label = '8x64'
        c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_128_multisnap_3_7_128_3_2_6.tar')
        c_label = '8x128'

        # # incremental - singlescale 
        # a = torch.load('./saved_models/single_scale/gnn_lr_1em4_bs_32_multisnap_incr_v2_3_7_128_3_2_6.tar')
        # a_label = '8x32'
        # b = torch.load('./saved_models/single_scale/gnn_lr_1em4_bs_64_multisnap_incr_v2_3_7_128_3_2_6.tar')
        # b_label = '8x64'
        # c = torch.load('./saved_models/single_scale/gnn_lr_1em4_bs_128_multisnap_incr_v2_3_7_128_3_2_6.tar')
        # c_label = '8x128'

        epochs = list(range(1, 300))
        plt.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots()
        ax.plot(a['loss_hist_train'][1:], lw=2, color='red', label=a_label)
        #ax.plot(a['loss_hist_test'][:-1], lw=2, color='red', ls='--')
        ax.plot(b['loss_hist_train'][1:], lw=2, color='black', label=b_label)
        #ax.plot(b['loss_hist_test'][:-1], lw=2, color='black', ls='--')
        ax.plot(c['loss_hist_train'][1:], lw=2, color='blue', label=c_label)
        #ax.plot(c['loss_hist_test'][:-1], lw=2, color='blue', ls='--')

        # incremental - multiscale
        a = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_32_multisnap_incr_v2_3_7_128_3_2_6.tar')
        a_label = '8x32'
        b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_64_multisnap_incr_v2_3_7_128_3_2_6.tar')
        b_label = '8x64'
        c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_128_multisnap_incr_v2_3_7_128_3_2_6.tar')
        c_label = '8x128'

        ax.plot(a['loss_hist_train'][1:], lw=2, color='red', label=a_label, ls='--')
        #ax.plot(a['loss_hist_test'][:-1], lw=2, color='red', ls='--')
        ax.plot(b['loss_hist_train'][1:], lw=2, color='black', label=b_label, ls='--')
        #ax.plot(b['loss_hist_test'][:-1], lw=2, color='black', ls='--')
        ax.plot(c['loss_hist_train'][1:], lw=2, color='blue', label=c_label, ls='--')
        #ax.plot(c['loss_hist_test'][:-1], lw=2, color='blue', ls='--')

        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        
        # ax2 = ax.twinx()
        # ax2.plot(a['lr_hist'][1:],  label='mp %d' %(i), lw=2, color='blue')
        # ax2.set_ylabel('Learning Rate', color='blue')
        # ax2.tick_params(axis='y', labelcolor='blue')
        # ax2.grid(False)

        plt.show(block=False)

    # ~~~ [REVISION] Training loss -- training losses for BFS and cavity  
    if 1 == 0:
        n_mp = 6
        fine_mp = 'True'
        case = "bfs"

        # coarse-scale neighbors -- single-scale 
        a = torch.load(f'./saved_models/single_scale/bfs_sampled_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re1600_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = 'Re=1600, 0nei'
        a_color = 'blue'
        a_ls = '-'

        b = torch.load(f'./saved_models/single_scale/bfs_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re1600_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = 'Re=1600, 26nei'
        b_color = 'blue'
        b_ls = '--'

        c = torch.load(f'./saved_models/single_scale/bfs_sampled_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        c_label = 'Re=3200, 0nei'
        c_color = 'red'
        c_ls = '-'

        d = torch.load(f'./saved_models/single_scale/bfs_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        d_label = 'Re=3200, 26nei'
        d_color = 'red'
        d_ls = '--'

        e = torch.load(f'./saved_models/single_scale/cavity_sampled_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        e_label = 'Re=3200, 0nei (cavity)'
        e_color = 'black'
        e_ls = '-'

        f = torch.load(f'./saved_models/single_scale/cavity_sampled_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        f_label = 'Re=3200, 26nei (cavity)'
        f_color = 'black'
        f_ls = '--'

        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(a['loss_hist_train'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        ax.plot(b['loss_hist_train'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)
        ax.plot(c['loss_hist_train'][0:], lw=2, color=c_color, label=c_label, ls=c_ls)
        ax.plot(d['loss_hist_train'][0:], lw=2, color=d_color, label=d_label, ls=d_ls)
        ax.plot(e['loss_hist_train'][0:], lw=2, color=e_color, label=e_label, ls=e_ls)
        ax.plot(f['loss_hist_train'][0:], lw=2, color=f_color, label=f_label, ls=f_ls)

        #ax.set_yscale('log')
        ax.legend(fontsize=12, frameon=True, edgecolor='black', framealpha=1, fancybox=False, bbox_transform=None)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        #ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)

    # ~~~ [REVISION] Training Loss -- Effect of residual interpolation on TGV, at Re=1600 and 3200   
    if 1 == 0:
        n_mp = 6
        fine_mp = 'True'

        # coarse-scale neighbors -- single-scale 
        a = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = 'Re=1600, 0nei, resid'
        a_color = 'blue'
        a_ls = '-'

        b = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = 'Re=1600, 0nei, no resid'
        b_color = 'blue'
        b_ls = '--'

        c = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        c_label = 'Re=1600, 26nei, resid'
        c_color = 'black'
        c_ls = '-'

        d = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        d_label = 'Re=1600, 26nei, no resid'
        d_color = 'black'
        d_ls = '--'

        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        #ax.plot(a['loss_hist_train'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        #ax.plot(b['loss_hist_train'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)
        ax.plot(c['loss_hist_test'][0:], lw=2, color=c_color, label=c_label, ls=c_ls)
        ax.plot(d['loss_hist_test'][0:], lw=2, color=d_color, label=d_label, ls=d_ls)
        #ax.plot(e['loss_hist_train'][0:], lw=2, color=e_color, label=e_label, ls=e_ls)
        #ax.plot(f['loss_hist_train'][0:], lw=2, color=f_color, label=f_label, ls=f_ls)

        ax.set_yscale('log')
        ax.legend(fontsize=12, frameon=True, edgecolor='black', framealpha=1, fancybox=False, bbox_transform=None)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        #ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)


    # ~~~ [REVISION] Training Loss -- Effect of number of message passing layers on TGV and BFS models  
    if 1 == 1:
        fine_mp = 'True'

        # coarse-scale neighbors -- single-scale 
        n_mp = 6
        a = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = f'n_mp = {n_mp}'
        a_color = 'black'
        a_ls = '-'

        n_mp = 4
        b = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = f'n_mp = {n_mp}'
        b_color = 'blue'
        b_ls = '-'

        n_mp = 2
        c = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        c_label = f'n_mp = {n_mp}'
        c_color = 'red'
        c_ls = '-'

        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(a['loss_hist_test'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        ax.plot(b['loss_hist_test'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)
        ax.plot(c['loss_hist_test'][0:], lw=2, color=c_color, label=c_label, ls=c_ls)

        ax.set_yscale('log')
        ax.legend(fontsize=12, frameon=True, edgecolor='black', framealpha=1, fancybox=False, bbox_transform=None)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        #ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        #ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)


    # ~~~ [REVISION] -- Training loss -- effect of incremental training on TGV model   
    if 1 == 0:
        n_mp = 6
        fine_mp = 'True'

        # coarse-scale neighbors -- single-scale 
        a = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = 'Re=1600, 0nei, resid'
        a_color = 'blue'
        a_ls = '-'

        b = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_incr_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = 'Re=1600, 0nei, resid+incr'
        b_color = 'blue'
        b_ls = '--'

        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(a['loss_hist_train'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        ax.plot(b['loss_hist_train'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)

        ax.set_yscale('log')
        ax.legend(fontsize=12, frameon=True, edgecolor='black', framealpha=1, fancybox=False, bbox_transform=None)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        #ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)

    # ~~~ [REVISION] -- Training loss -- effect of finetuning from either TGV/BFS to Cavity 
    if 1 == 0:
        n_mp = 6
        fine_mp = 'True'

        # coarse-scale neighbors -- single-scale 
        a = torch.load(f'./saved_models/single_scale/finetune_bfs2cav_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = 'Re=3200, 0nei, bfs2cav'
        a_color = 'blue'
        a_ls = '-'

        b = torch.load(f'./saved_models/single_scale/finetune_bfs2cav_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = 'Re=3200, 26nei, bfs2cav'
        b_color = 'blue'
        b_ls = '--'

        c = torch.load(f'./saved_models/single_scale/finetune_tgv2cav_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        c_label = 'Re=3200, 0nei, tgv2cav'
        c_color = 'red'
        c_ls = '-'

        d = torch.load(f'./saved_models/single_scale/finetune_tgv2cav_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        d_label = 'Re=3200, 26nei, tgv2cav'
        d_color = 'red'
        d_ls = '--'

        e = torch.load(f'./saved_models/single_scale/cavity_gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        e_label = 'Re=3200, 0nei (cavity)'
        e_color = 'black'
        e_ls = '-'

        f = torch.load(f'./saved_models/single_scale/cavity_gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_re3200_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        f_label = 'Re=3200, 26nei (cavity)'
        f_color = 'black'
        f_ls = '--'

        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(a['loss_hist_train'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        ax.plot(b['loss_hist_train'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)
        ax.plot(c['loss_hist_train'][0:], lw=2, color=c_color, label=c_label, ls=c_ls)
        ax.plot(d['loss_hist_train'][0:], lw=2, color=d_color, label=d_label, ls=d_ls)
        ax.plot(e['loss_hist_train'][0:], lw=2, color=e_color, label=e_label, ls=e_ls)
        ax.plot(f['loss_hist_train'][0:], lw=2, color=f_color, label=f_label, ls=f_ls)

        ax.set_yscale('log')
        ax.legend(fontsize=12, frameon=True, edgecolor='black', framealpha=1, fancybox=False, bbox_transform=None)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        #ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)

    # ~~~~ Save predicted flowfield into .f file 
    # ONE-SHOT and INCREMENTAL, WITHOUT NEIGHBORS
    if 1 == 0:
        mode = "single_scale"
        #data_dir = "./datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)
        #test_dataset = torch.load(data_dir + "/valid_dataset.pt")
        #edge_index = test_dataset[0].edge_index

        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_bs_4_multisnap_3_7_128_3_2_%d.tar' %(mode,mp))
        #a = torch.load('./saved_models/%s/gnn_lr_1em4_bs_32_multisnap_incr_v2_3_7_128_3_2_%d.tar' %(mode,mp))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']

        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)

        model.load_state_dict(a['state_dict'])
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        model.to(device)
        model.eval()

        # Load eval and target snapshot 
        TORCH_FLOAT = torch.float32
        #nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
        #nrs_snap_dir = '/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr'
        nrs_snap_dir = '/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/one_shot'
        
        # Load in edge index 
        poly = 7
        case_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv"
        Re = '1600'
        edge_index_path = f"{case_path}/Re_{Re}_poly_7/gnn_outputs_poly_{poly}/edge_index_element_local_rank_0_size_4"
        if mode == "single_scale":
            edge_index = get_edge_index(edge_index_path)
        elif mode == "multi_scale":
            edge_index_vertex_path = f"{case_path}/Re_{Re}_poly_7/gnn_outputs_poly_{poly}/edge_index_element_local_vertex_rank_0_size_4"
            edge_index = get_edge_index(edge_index_path,
                                        edge_index_vertex_path)


        #t_str_list = ['00017','00019', '00020','00021'] # 1 takes ~5 min 
        #t_str_list = ['00017', '00020']
        t_str_list = ['00016', '00017', '00018', '00019', '00020', '00021']
        #t_str_list = ['000%02d' %(i) for i in range(12,41)]

        for t_str in t_str_list:
            # Incremental: 
            #x_field = readnek(nrs_snap_dir + f'/snapshots_interp_1to3/newtgv0.f{t_str}')
            #x_field = readnek(nrs_snap_dir + f'/snapshots_interp_{poly-2}to{poly}/newtgv0.f{t_str}')
            #x_field = readnek(nrs_snap_dir + f'/snapshots_gnn_correction_{mode}_interp_{poly-2}to{poly}/newtgv0.f{t_str}')
            # One-shot
            x_field = readnek(nrs_snap_dir + f'/snapshots_interp_1to{poly}/newtgv0.f{t_str}')

            # y_field = readnek(nrs_snap_dir + f'/snapshots_target/tgv0.f{t_str}')
            n_snaps = len(x_field.elem)

            with torch.no_grad():
                for i in range(n_snaps):
                    print(f"Evaluating snap {i}/{n_snaps}")
                    pos_i = torch.tensor(x_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    vel_x_i = torch.tensor(x_field.elem[i].vel).reshape((3, -1)).T
                    # vel_y_i = torch.tensor(y_field.elem[i].vel).reshape((3, -1)).T

                    # get x_mean and x_std 
                    x_mean_element = torch.mean(vel_x_i, dim=0).unsqueeze(0).repeat(vel_x_i.shape[0], 1)
                    x_std_element = torch.std(vel_x_i, dim=0).unsqueeze(0).repeat(vel_x_i.shape[0], 1)

                    # element lengthscale 
                    lengthscale_element = torch.norm(pos_i.max(dim=0)[0] - pos_i.min(dim=0)[0], p=2)

                    # create data 
                    data = Data( x = vel_x_i.to(dtype=TORCH_FLOAT),
                                      # y = vel_y_i.to(dtype=TORCH_FLOAT),
                                      x_mean = x_mean_element.to(dtype=TORCH_FLOAT),
                                      x_std = x_std_element.to(dtype=TORCH_FLOAT),
                                      L = lengthscale_element.to(dtype=TORCH_FLOAT),
                                      pos = pos_i.to(dtype=TORCH_FLOAT),
                                      pos_norm = (pos_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                                      edge_index = edge_index)

                    data = data.to(device)

                    # ~~~~ Model evaluation ~~~~ # 
                    # 1) Preprocessing: scale input  
                    eps = 1e-10
                    x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

                    # 2) evaluate model 
                    out_gnn = model(x_scaled, data.edge_index, data.pos_norm, data.batch)
                        
                    # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
                    y_pred = out_gnn * (data.x_std + eps) + data.x 

                    # ~~~~ Making the .f file ~~~~ # 
                    # Re-shape the prediction, convert back to fp64 numpy 
                    y_pred = y_pred.cpu()
                    orig_shape = x_field.elem[i].vel.shape
                    y_pred_rs = torch.reshape(y_pred.T, orig_shape).to(dtype=torch.float64).numpy()

                    # Place back in the snapshot data 
                    x_field.elem[i].vel[:,:,:,:] = y_pred_rs

                # Write 
                print('Writing...')
                # incremental: 
                #directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_full_{poly-2}to{poly}"
                #directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_{poly-2}to{poly}"
                # One shot: 
                directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_1to{poly}"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                    print(f"Directory '{directory_path}' created.")
                writenek(directory_path +  f"/newtgv0.f{t_str}", x_field)
                print(f'finished writing {t_str}') 

        # plt.rcParams.update({'font.size': 14})
        # fig, ax = plt.subplots(1,3, figsize=(10,4))
        # for comp in range(3):
        #     ax[comp].scatter(data.x[:,comp], data.y[:,comp], color='red')
        #     ax[comp].scatter(y_pred[:,comp], data.y[:,comp], color='lime')
        #     ax[comp].plot([data.y[:,comp].min(), data.y[:,comp].max()],
        #             [data.y[:,comp].min(), data.y[:,comp].max()],
        #             color='black', lw=2)
        #     ax[comp].set_title('n_mp=%d, comp=%d' %(mp,comp))
        #     ax[comp].set_xlabel('Prediction')
        #     ax[comp].set_ylabel('Target')
        # plt.show(block=False)

    # ~~~~ Save predicted flowfield into .f file 
    # COARSE-TO-FINE GNN 
    if 1 == 0:
        local = False
        use_residual = True
        n_element_neighbors = 0 
        # if use_residual:
        #     a = torch.load(f"./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar")
        # else:
        #     a = torch.load(f"./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar")

        Re_str = "_re3200"
        #Re_str = ""
        Re_data = '3200'

        if use_residual:
            model_list = [
                #f"gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap_resid_3_7_132_128_3_2_6_False.tar", # crs-6
                f"gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap_resid{Re_str}_3_7_132_128_3_2_12_False.tar",# crs-12
                f"gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap_resid{Re_str}_3_7_132_128_3_2_6_True.tar"] # fne-6
        else:
            model_list = [
                #f"gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap_3_7_132_128_3_2_6_False.tar",
                f"gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap{Re_str}_3_7_132_128_3_2_12_False.tar",
                f"gnn_lr_1em4_bs_4_nei_{n_element_neighbors}_c2f_multisnap{Re_str}_3_7_132_128_3_2_6_True.tar"]

        for model_path in model_list:
            a = torch.load(f"./saved_models/single_scale/{model_path}")
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

            # Load eval and target snapshot 
            TORCH_FLOAT = torch.float32
            if local: 
                #nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
                nrs_snap_dir = './temp'
            else:
                nrs_snap_dir = f'/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_{Re_data}_poly_7_testset/one_shot'
            
            # Load in edge index 
            poly_lo = 1
            poly_hi = 7
            if local:
                case_path = "./temp"
            else: 
                case_path = f"/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_{Re_data}_poly_7"
            edge_index_path_lo = f"{case_path}/gnn_outputs_poly_{poly_lo}/edge_index_element_local_rank_0_size_4"
            edge_index_path_hi = f"{case_path}/gnn_outputs_poly_{poly_hi}/edge_index_element_local_rank_0_size_4"
            edge_index_lo = get_edge_index(edge_index_path_lo)
            edge_index_hi = get_edge_index(edge_index_path_hi)
            
            #t_str_list = ['00019', '00020','00021'] # 1 takes ~5 min 
            #t_str_list = ['00017', '00019', '00020', '00021'] # 1 takes ~5 min 
            t_str_list = ['00016', '00017', '00018', '00019', '00020', '00021'] # 1 takes ~5 min 
            #t_str_list = ['000%02d' %(i) for i in range(12,41)]

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
                if os.path.exists(directory_path + f"/newtgv_pred0.f{t_str}"):
                    print(directory_path + f"/newtgv_pred0.f{t_str} \n already exists!!! Skipping...") 
                    continue

                # One-shot
                xlo_field = readnek(nrs_snap_dir + f'/snapshots_coarse_{poly_hi}to{poly_lo}/newtgv0.f{t_str}')
                xhi_field = readnek(nrs_snap_dir + f'/snapshots_target/newtgv0.f{t_str}')
                #xhi_field = readnek(nrs_snap_dir + f'/snapshots_interp_{poly_lo}to{poly_hi}/newtgv0.f{t_str}')
                xhi_field_pred = readnek(nrs_snap_dir + f'/snapshots_target/newtgv0.f{t_str}')
                xhi_field_error = readnek(nrs_snap_dir + f'/snapshots_target/newtgv0.f{t_str}')

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
                        print(f"Evaluating snap {i}/{n_snaps}")
                        
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
                    print('Writing...')
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                        print(f"Directory '{directory_path}' created.")
                    writenek(directory_path +  f"/newtgv_pred0.f{t_str}", xhi_field_pred)
                    writenek(directory_path +  f"/newtgv_error0.f{t_str}", xhi_field_error)
                    print(f'finished writing {t_str}') 

    # ~~~~ Save predicted flowfield into .f file 
    # Just the KNN interpolation!!! 
    if 1 == 0:
        # Load eval and target snapshot 
        TORCH_FLOAT = torch.float32
        
        xlo_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr/snapshots_knninterp_3to5"
        xhi_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr/snapshots_target"
        xknn_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr/snapshots_knninterp_5to7"

        for snap in ["16", "17", "18", "19", "20", "21"]:
            xlo_field = readnek(f"{xlo_path}/newtgv0.f000{snap}")
            xhi_field = readnek(f"{xhi_path}/tgv0.f000{snap}")
            n_snaps = len(xlo_field.elem)

            with torch.no_grad():
                for i in range(n_snaps):
                    print(f"knn interp -- Evaluating snap {i}/{n_snaps}")
                    
                    pos_xlo_i = torch.tensor(xlo_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    vel_xlo_i = torch.tensor(xlo_field.elem[i].vel).reshape((3, -1)).T
                    pos_xhi_i = torch.tensor(xhi_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    
                    # ~~~~ kNN evaluation ~~~~ # 
                    y_pred = tgnn.unpool.knn_interpolate(
                            x = vel_xlo_i,
                            pos_x = pos_xlo_i,
                            pos_y = pos_xhi_i,
                            k = 8)

                    # ~~~~ Making the .f file ~~~~ # 
                    # Re-shape the prediction, convert back to fp64 numpy 
                    y_pred = y_pred.cpu()
                    orig_shape = xhi_field.elem[i].vel.shape
                    y_pred_rs = torch.reshape(y_pred.T, orig_shape).to(dtype=torch.float64).numpy()

                    # Place back in the snapshot data 
                    xhi_field.elem[i].vel[:,:,:,:] = y_pred_rs

                # Write 
                print('Writing...')
                writenek(f"{xknn_path}/newtgv0.f000{snap}", xhi_field)

    # ~~~~ Analyze predictions -- all elements 
    if 1 == 0:
        mode = "multi_scale"
        data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)
        # Load data 
        train_dataset = torch.load(data_dir + "/train_dataset.pt")
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")
        
        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mode,mp))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']

        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model.load_state_dict(a['state_dict'])
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        model.to(device)
        model.eval()

        with torch.no_grad():
            x_input_list = [] 
            y_pred_list = []
            y_target_list = []

            dataset = test_dataset 
            N = len(dataset) 
            mse_before_gnn = torch.zeros(N)
            mse_after_gnn = torch.zeros(N) 

            sample = dataset[0]


            for i in range(N):
                print('evaluating %d/%d' %(i+1, N))
                data = dataset[i]

                # 1) Preprocessing: scale input  
                eps = 1e-10
                x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

                # 2) evaluate model 
                out_gnn = model(x_scaled, data.edge_index, data.pos_norm, data.batch)
                    
                # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
                y_pred = out_gnn * (data.x_std + eps) + data.x 
  
                # loss before gnn 
                mse_before_gnn[i] = F.mse_loss(data.x, data.y)

                # loss after gnn 
                mse_after_gnn[i] = F.mse_loss(y_pred, data.y) 

        # Visualize
        element_ids = torch.arange(N) 
    
        # Get the difference in MSE 
        # if negative, bad
        # if positive, good 
        difference = mse_before_gnn - mse_after_gnn

        # sort the difference  
        idx_sort = torch.sort(difference, descending=True)[1]

        element_ids_sorted = element_ids[idx_sort]
        difference_sorted = difference[idx_sort]

        plt.rcParams.update({'font.size': 14})

        # Plotting error difference versus element 
        fig, ax = plt.subplots()
        ax.plot(element_ids, difference_sorted, lw=2)
        #ax.plot(element_ids, difference_sorted.abs(), lw=2)
        ax.set_xlabel('Element IDs (Sorted)')
        ax.set_ylabel('Model Gain')
        ax.set_yscale('log')
        plt.show(block=False)
    
    # ~~~~ Visualize model predictions -- a single element  
    if 1 == 0:
        mode = "multi_scale"
        data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)

        # Load data 
        train_dataset = torch.load(data_dir + "/train_dataset.pt")
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")

        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mode,mp))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']

        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model.load_state_dict(a['state_dict'])
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'


        model.to(device)
        model.eval()

        with torch.no_grad():
            #data = test_dataset[4000]
            data = train_dataset[4000]
             
            # 1) Preprocessing: scale input  
            eps = 1e-10
            x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

            # 2) evaluate model 
            out_gnn = model(x_scaled, data.edge_index, data.pos_norm, data.batch)
                
            # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
            y_pred = out_gnn * (data.x_std + eps) + data.x 

        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(1,3, figsize=(10,4))
        for comp in range(3):
            ax[comp].scatter(data.x[:,comp], data.y[:,comp], color='red')
            ax[comp].scatter(y_pred[:,comp], data.y[:,comp], color='lime')
            ax[comp].plot([data.y[:,comp].min(), data.y[:,comp].max()],
                    [data.y[:,comp].min(), data.y[:,comp].max()],
                    color='black', lw=2)
            ax[comp].set_title('n_mp=%d, comp=%d' %(mp,comp))
            ax[comp].set_xlabel('Prediction')
            ax[comp].set_ylabel('Target')
        plt.show(block=False)

    # Comparing two models, on the full dataset testing 
    if 1 == 0:
        mode = 'multi_scale'
        data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)

        # Load data 
        train_dataset = torch.load(data_dir + "/train_dataset.pt")
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")

        # Load model 1 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_3_7_128_3_2_6.tar' %(mode))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']
        model_1 = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model_1.load_state_dict(a['state_dict'])
        device = 'cpu'
        model_1.to(device)
        model_1.eval()

        # Load model 2 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_unscaledResidual_3_7_128_3_2_6.tar' %(mode))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']
        model_2 = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model_2.load_state_dict(a['state_dict'])
        device = 'cpu'
        model_2.to(device)
        model_2.eval()

        N_test = len(test_dataset)

        error_1 = torch.zeros((N_test, 3))
        error_2 = torch.zeros((N_test, 3))
        y_std = torch.zeros((N_test))

        with torch.no_grad():
            for i in range(N_test):
                print(f"iter {i} / {N_test}")
                data = test_dataset[i]
                 
                # 1) Preprocessing: scale input  
                eps = 1e-10
                x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

                # 2) evaluate model 
                out_gnn_1 = model_1(x_scaled, data.edge_index, data.pos_norm, data.batch)
                out_gnn_2 = model_2(x_scaled, data.edge_index, data.pos_norm, data.batch)
                    
                # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
                y_pred_1 = out_gnn_1 * (data.x_std + eps) + data.x 
                y_pred_2 = out_gnn_2 + data.x 

                error_1[i] = torch.mean( (y_pred_1 - data.y)**2, dim=0 ) 
                error_2[i] = torch.mean( (y_pred_2 - data.y)**2, dim=0 ) 
                y_std[i] = torch.std(data.y)


        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(1,3, figsize=(10,4))
        for comp in range(3):
            ax[comp].scatter(error_1[:,comp], y_std, color='black')
            ax[comp].scatter(error_2[:,comp], y_std, color='blue')
            ax[comp].set_title('y_std vs error, comp=%d' %(comp))
            ax[comp].set_xlabel('Error')
            ax[comp].set_ylabel('y_std')
            ax[comp].set_xscale('log')
        plt.show(block=False)
       
    # ~~~~ Postprocess run logs: KE, dissipation, enst
    if 1 == 0:
        import re

        def read_nrs_log(file_path):
            lines = []
            values = []
            with open(file_path, 'r') as file:
                 for line in file:
                     if line.startswith('  time'):
                         line=line.strip()
                         lines.append(line)
                         numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                         numbers = [float(num) for num in numbers]
                         values.append(np.array(numbers, ndmin=2))
            del lines[0]
            del values[0]
            values = np.concatenate(values, axis=0) 
            return lines, values 

        # Re = 1600 
        _, values_1600 = read_nrs_log('./outputs/run_logs/36_cubed/Re_1600_poly_7.log')
        _, values_2000 = read_nrs_log('./outputs/run_logs/36_cubed/Re_2000_poly_7.log')
        _, values_2400 = read_nrs_log('./outputs/run_logs/36_cubed/Re_2400_poly_7.log')
        _, values_3200 = read_nrs_log('./outputs/run_logs/36_cubed/Re_3200_poly_7.log')

        _, values_1600_2 = read_nrs_log('./outputs/run_logs/72_cubed/Re_1600_poly_7.log')
        
        #   0      1     2          4        5       6
        # [time, enst, energy, -2*nu*enst, dE/dt, nuEff/nu]

        # Kinetic energy  
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], values_1600[:,2], label='Re=1600, 36^3, P=7', lw=lw)
        #ax.plot(values_1600_2[:,0], values_1600_2[:,2], label='Re=1600, 72^3, P=7', lw=lw)
        # ax.plot(values_2000[:,0], values_2000[:,2], label='Re=2000', lw=lw)
        # ax.plot(values_2400[:,0], values_2400[:,2], label='Re=2400', lw=lw)
        ax.set_title('Kinetic Energy')
        #ax.set_xlim([7.5, 10.5])
        plt.show(block=False)

        # dissipation rate ( dEk/dt )
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], -values_1600[:,4], label='Re=1600, 36^3, P=7', lw=lw)
        ax.vlines(x=8,  ymin =0, ymax = 0.013, color='black', lw=2)
        ax.vlines(x=9,  ymin =0, ymax = 0.013, color='black', lw=2)
        ax.vlines(x=10, ymin =0, ymax = 0.013, color='black', lw=2)
        ax.vlines(x=8.5, ymin =0, ymax = 0.013, color='red', lw=2, ls='--')
        ax.vlines(x=9.5, ymin =0, ymax = 0.013, color='red', lw=2, ls='--')
        ax.vlines(x=10.5, ymin =0, ymax = 0.013, color='red', lw=2, ls='--')
        #ax.plot(values_1600_2[:,0], -values_1600_2[:,4], label='Re=1600, 72^3, P=7', lw=lw, ls='--')
        #ax.plot(values_2000[:,0], -values_2000[:,5], label='Re=2000', lw=lw)
        #ax.plot(values_2400[:,0], -values_2400[:,5], label='Re=2400', lw=lw)
        ax.set_title('Dissipation Rate')
        ax.set_xlim([7.5, 11])
        #ax.legend()
        ax.set_xlabel('t_c')
        #ax.set_ylabel('-dEk/dt')
        ax.set_ylabel('-dEk/dt')
        plt.show(block=False)

        lw=1.5
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(values_1600[:,0], -values_1600[:,4], label='Re=1600', color='black', lw=lw)
        ax.plot(values_3200[:,0], -values_2400[:,4], label='Re=3200', color='blue', ls='--', lw=lw)
        ax.set_title('Dissipation Rate')
        ax.set_xlim([0,20])
        ax.legend()
        ax.set_xlabel('Time')
        #ax.set_ylabel('-dEk/dt')
        ax.set_ylabel('-dEk/dt')
        plt.show(block=False)


    # ~~~~ Plotting  
    if 1 == 0:
        print('Plotting...')

        # Plot the edges 
        if 1 == 0:
            #for rl in range(SIZE):
            data = train_dataset[0]
            G = utils.to_networkx(data=data)

            
            pos = dict(enumerate(np.array(data.pos)))
            node_xyz = np.array([pos[v] for v in sorted(G)])
            edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
            pos = data.pos

            ms = 50
            lw_edge = 2
            lw_marker = 1
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot the edges
            count = 0
            for vizedge in edge_xyz:
                #ax.plot(*vizedge.T, color="black", lw=lw_edge, alpha=0.1)
                ax.plot(*vizedge.T, color="black", alpha=0.3)
                count += 1

            # plot the nodes 
            ax.scatter(*pos.T, s=ms, ec='black', lw=lw_marker, c='black', alpha=1)

            ax.grid(False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=34, azim=-24, roll=0)
            ax.set_aspect('equal')
            fig.tight_layout()
            plt.show(block=False)
    
        # Plot scatter plots of node features 
        if 1 == 0:
            e_id = 0

            sample = train_dataset[e_id]

            color='red' 
            fig, ax = plt.subplots(1,3, figsize=(10,4))
            
            for comp in range(3):
                ax[comp].scatter(sample.x[:,comp], sample.y[:,comp], c=color)
                ax[comp].plot( [sample.y[:,comp].min(), sample.y[:,comp].max()], 
                               [sample.y[:,comp].min(), sample.y[:,comp].max()],
                           color='black', lw=2)

            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[0].set_title('e_id = %d' %(e_id))
            ax[0].set_aspect('equal')

            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[1].set_title('e_id = %d' %(e_id))
            ax[1].set_aspect('equal')

            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
            ax[2].set_title('e_id = %d' %(e_id))
            ax[2].set_aspect('equal')
            plt.show(block=False)

        # RMS versus reconstruction MSE 
        if 1 == 0:
            fig, ax = plt.subplots(figsize=(8,7))
            ax.scatter(rms_lo, element_error, facecolor='none', edgecolor='black', lw=0.75, s=75, label='Target')
            ax.scatter(rms_hi, element_error, facecolor='none', edgecolor='red', lw=0.75, s=75, label='Predicted')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('RMS Velocity', fontsize=20)
            ax.set_ylabel('Reconstruction MSE', fontsize=20)
            ax.legend(fontsize=20)
            plt.show(block=False)

    # ~~~~ Test the model 
    if 1 == 0:   
        sample = train_dataset[0]
        input_node_channels = sample.x.shape[1]
        input_edge_channels = sample.pos.shape[1] + sample.x.shape[1] + 1 
        hidden_channels = 128  
        output_node_channels = sample.y.shape[1] 
        n_mlp_hidden_layers = 2 
        n_messagePassing_layers = 8

        name = 'gnn' 
        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)

        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
        )

        for bidx, data in enumerate(train_loader):
            # 1) Preprocessing: scale input  
            eps = 1e-10
            x_scaled = (data.x - data.x_mean)/(data.x_std + eps) 

            # 2) Evaluate the model 
            gnn_out = model(x_scaled, data.edge_index, data.pos_norm, data.batch)

            # 3) Set the target 
            target = (data.y - data.x)/(data.x_std + eps)

            # 4) Inference -- Make a prediction 
            # y_pred = gnn_out * (data.x_std.unsqueeze(0) + eps) + data.x 

            break

    # ~~~~ Compute POD basis 
    if 1 == 0:
        print('pod basis')
        # mode = "multi_scale"
        # data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)
        # train_dataset = torch.load(data_dir + "/train_dataset.pt")
        # test_dataset = torch.load(data_dir + "/valid_dataset.pt")

        raw_data_path = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7/snapshots_target/gnn_outputs_poly_7'
        time_str = '10.0'
        rank = 0
        SIZE = 4 
        x = torch.tensor(np.loadtxt(raw_data_path + '/fld_u_time_%s_rank_%d_size_%d' %(time_str,rank,SIZE)))
        node_element_ids = torch.tensor(np.loadtxt(raw_data_path + '/node_element_ids_rank_%d_size_%d' %(rank, SIZE), dtype=np.int64))
        x = torch.stack(utils.unbatch(x, node_element_ids, dim=0)).numpy()


        Ne = x.shape[0]
        Np = x.shape[1]
        Nf = x.shape[2]

        x = x.reshape((Ne, Np*Nf))
        
        print('Covariance...')
        cov = (1.0 / (Ne - 1.0)) * x.T @ x
        lam, evec = np.linalg.eig(cov)

        # a = np.diag(evec.T @ evec)

        fig, ax = plt.subplots()
        ax.plot(lam)
        ax.set_yscale('log')
        plt.show(block=False)
        




        # Project a single element in the POD basis 
        # ELE --- [Nf, 1]
        # MODE --- [Nf, 4]
        # A -- ELE.T @ MODE --- [1, 4]  
        # RECON -- A @ MODE.T 

        N_modes = Nf*Np 
        sample = x[0, :].reshape((1,-1)) 
        modes = evec[:, 0:N_modes]
        coeffs = (sample @ modes).squeeze()
        
        fig, ax = plt.subplots()
        ax.plot(coeffs)
        ax.set_xscale('log')
        plt.show(block=False)
 

    # ~~~~ PyMech tests -- SB: this is how we go between pymech / pygeom representations. 
    if 1 == 0:
        #nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
        nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/bfs_2/Re_5100_p_7/one_shot'
        field1 = readnek(nrs_snap_dir + '/snapshots_target/newbfs0.f00010')
        first_element = field1.elem[0]
        print("Type =", type(first_element))
        print(first_element)

        #field2 = readnek(nrs_snap_dir + '/snapshots_interp_1to7/newtgv0.f00010')
        field2 = readnek(nrs_snap_dir + '/snapshots_coarse_7to1/newbfs0.f00010')

        
        # i_err = []
        # for i in range(len(field1.elem)):
        #     pos_1 = field1.elem[i].pos
        #     pos_2 = field2.elem[i].pos

        #     x_gll = pos_1[0,0,0,:]
        #     dx_min = x_gll[1] - x_gll[0] 

        #     # x = pos_1[0,0,0,:]
        #     # y = np.ones_like(x)
        #     # fig, ax = plt.subplots()
        #     # ax.plot(x, y, marker='o', ms=20)
        #     # plt.show(block=False)

        #     error_max = (pos_1 - pos_2).max()
        #     rel_error = (error_max / dx_min)*100

        #     if rel_error> 1e-2:
        #         print(f"i={i} \t error_max = {error_max} \t rel_error = {rel_error}")
        #         print("WARNING --- relative error in positions exceeds 0.01%")
        #         i_err.append(i)

        
        field_fine = field1
        field_crse = field2

        eid = 145 
        pos_fine = (field_fine.elem[eid].pos).reshape((3,-1)).T
        pos_crse = (field_crse.elem[eid].pos).reshape((3,-1)).T

        fig, ax = plt.subplots()
        ax.scatter(pos_fine[:,0], pos_fine[:,1])
        ax.scatter(pos_crse[:,0], pos_crse[:,1])
        plt.show(block=False)


        # Test reshaping : 
        if 1 == 0:
            pos_1 = torch.tensor(first_element.pos).reshape((3, -1)).T # pygeom pos format -- [N, 3]
            x_1 = torch.tensor(first_element.vel).reshape((3, -1)).T # pygeom x format -- [N, 3]

            pos_orig = first_element.pos
            x_orig = first_element.vel

            pos_2 = torch.reshape(pos_1.T, (3,8,8,8))
            x_2 = torch.reshape(x_1.T, (3,8,8,8))


        # Test read/write: 
        if 1 == 0:
            # Write new file 
            x_1 = torch.tensor(first_element.vel).reshape((3, -1)).T # pygeom x format -- [N, 3]
            y = x_1 + 84.0
            y = torch.reshape(y.T, (3,8,8,8))


            # modify one lemenet data and re-write into new .f file 
            print(f"data before: {field.elem[0].vel}")
            field.elem[0].vel = y.numpy()
            print(f"data after: {field.elem[0].vel}")

            
            print("Writing...")
            writenek("./temp/tgv_override0.f00001", field)



            print("Reading again...")
            field2 = readnek("./temp/tgv_override0.f00001")
            y_test = field.elem[0].vel


        # Test plotting: 
        if 1 == 0:
            # Read in element-local edge index 
            edge_index_poly_7 = torch.tensor(np.loadtxt("./temp/gnn_outputs_poly_7/edge_index_element_local_rank_0_size_4").astype(np.int64).T)
            edge_index_poly_1 = torch.tensor(np.loadtxt("./temp/gnn_outputs_poly_1/edge_index_element_local_rank_0_size_4").astype(np.int64).T)

            element_poly_7 = field1.elem[1234]
            element_poly_1 = field2.elem[1234]

            # Plot 
            element = element_poly_1
            edge_index = edge_index_poly_1
            pos = torch.tensor(element.pos).reshape((3, -1)).T
            edge_xyz = pos[edge_index].permute(1,0,2)
            
            ms = 50
            lw_edge = 2
            lw_marker = 1
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot the edges
            count = 0
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color="black", lw=lw_edge, alpha=0.3)
                #ax.plot(*vizedge.T, color="black", lw=lw_edge * edge_weights[RANK][count], alpha=0.3)
                count += 1
            ax.scatter(*pos.T, s=ms, ec='black', lw=lw_marker, c='black', alpha=1)
            
            ax.grid(False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=34, azim=-24, roll=0)
            ax.set_aspect('equal')
            fig.tight_layout()
            plt.show(block=False)

        # Test data pruning 
        if 1 == 1:

            xhi = field1    
            xlo = field2

            chi = np.zeros((xhi.nel, 3), dtype=np.float64)
            clo = np.zeros((xlo.nel, 3), dtype=np.float64)
            for i in range(xhi.nel):
                chi[i,:] = xhi.elem[i].centroid[:]
            for i in range(xlo.nel):
                clo[i,:] = xlo.elem[i].centroid[:]

            # Keep 100% of elements below y=0, only a 20% above y>0 
            eid = list(range(xhi.nel))
            eid = np.array(eid, dtype=np.longlong)
            is_below_step = chi[:,1] < 0
            eid_below_step = eid[is_below_step]
            eid_above_step = eid[~is_below_step]

            sample_size = int(0.1 * len(eid_above_step))
            eid_above_step_sampled = np.random.choice(eid_above_step, size=sample_size, replace=False)

            eid_keep = np.concatenate((eid_below_step, eid_above_step_sampled))
            eid_keep.sort()

            fig, ax = plt.subplots(figsize=(12,6))
            ax.scatter(chi[eid_below_step,0], chi[eid_below_step,1], s=15, color='red')
            ax.scatter(chi[eid_above_step,0], chi[eid_above_step,1], s=15, color='black')
            ax.scatter(chi[eid_above_step_sampled,0], chi[eid_above_step_sampled,1], s=15, color='red')
            ax.set_aspect('equal')
            ax.grid(False)
            plt.show(block=False)

            fig, ax = plt.subplots(figsize=(12,6))
            ax.scatter(chi[eid_keep,0], chi[eid_keep,1], s=15, color='red')
            ax.set_aspect('equal')
            ax.grid(False)
            plt.show(block=False)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(chi[eid_below_step,0], chi[eid_below_step,1], chi[eid_below_step,2], s=5, color='red')
            # ax.scatter(chi[eid_above_step,0], chi[eid_above_step,1], chi[eid_above_step,2], s=5, color='red')
            # ax.set_aspect('equal')
            # plt.show(block=False)


            pass
        

