import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
import matplotlib.pyplot as plt
import dataprep.nekrs_graph_setup as ngs
import models.gnn as gnn
from pymech.neksuite import readnek,writenek
from pymech.dataset import open_dataset


seed = 122
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_grad_enabled(False)

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

if __name__ == "__main__":
    # ~~~~ postprocessing: training losses 
    if 1 == 0:
        mp = 6 
        # a = torch.load('./saved_models/single_scale/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mp))
        # b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mp))
        # #c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mp))
        # c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_32_multisnap_3_7_128_3_2_%d.tar' %(mp))

        a = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_32_multisnap_3_7_128_3_2_6.tar')
        a_label = '8x4'
        b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_multisnap_3_7_128_3_2_6.tar')
        b_label = '8x8'
        c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_128_multisnap_3_7_128_3_2_6.tar')
        c_label = '8x16'

        epochs = list(range(1, 300))
        plt.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots()
        ax.plot(a['loss_hist_train'][1:], lw=2, color='red', label=a_label)
        ax.plot(a['loss_hist_test'][:-1], lw=2, color='red', ls='--')
        ax.plot(b['loss_hist_train'][1:], lw=2, color='black', label=b_label)
        ax.plot(b['loss_hist_test'][:-1], lw=2, color='black', ls='--')
        ax.plot(c['loss_hist_train'][1:], lw=2, color='blue', label=c_label)
        ax.plot(c['loss_hist_test'][:-1], lw=2, color='blue', ls='--')

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

    # ~~~~ Save predicted flowfield into .f file 
    if 1 == 1:
        mode = "multi_scale"
        data_dir = "./datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")
        edge_index = test_dataset[0].edge_index

        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_bs_32_multisnap_3_7_128_3_2_%d.tar' %(mode,mp))
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
        nrs_snap_dir = '/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset'


        t_str_list = ['00017','00019','00021']

        for t_str in t_str_list:
            x_field = readnek(nrs_snap_dir + f'/snapshots_interp_1to7/newtgv0.f{t_str}')
            y_field = readnek(nrs_snap_dir + f'/snapshots_target/tgv0.f{t_str}')

            n_snaps = len(x_field.elem)

            with torch.no_grad():
                for i in range(n_snaps):
                    print(f"Evaluating snap {i}/{n_snaps}")
                    pos_i = torch.tensor(x_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    vel_x_i = torch.tensor(x_field.elem[i].vel).reshape((3, -1)).T
                    vel_y_i = torch.tensor(y_field.elem[i].vel).reshape((3, -1)).T

                    # get x_mean and x_std 
                    x_mean_element = torch.mean(vel_x_i, dim=0).unsqueeze(0).repeat(vel_x_i.shape[0], 1)
                    x_std_element = torch.std(vel_x_i, dim=0).unsqueeze(0).repeat(vel_x_i.shape[0], 1)

                    # element lengthscale 
                    lengthscale_element = torch.norm(pos_i.max(dim=0)[0] - pos_i.min(dim=0)[0], p=2)

                    # create data 
                    data = Data( x = vel_x_i.to(dtype=TORCH_FLOAT),
                                      y = vel_y_i.to(dtype=TORCH_FLOAT),
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
                writenek(nrs_snap_dir + f"/snapshots_interp_1to7_gnn_{mode}/newtgv0.f{t_str}", x_field)
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

        _, values_1600_2 = read_nrs_log('./outputs/run_logs/72_cubed/Re_1600_poly_7.log')
        
        #   0      1     2          4        5       6
        # [time, enst, energy, -2*nu*enst, dE/dt, nuEff/nu]

        # Kinetic energy  
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], values_1600[:,2], label='Re=1600, 36^3, P=7', lw=lw)
        ax.plot(values_1600_2[:,0], values_1600_2[:,2], label='Re=1600, 72^3, P=7', lw=lw)
        # ax.plot(values_2000[:,0], values_2000[:,2], label='Re=2000', lw=lw)
        # ax.plot(values_2400[:,0], values_2400[:,2], label='Re=2400', lw=lw)
        ax.set_title('Kinetic Energy')
        #ax.set_xlim([7.5, 10.5])
        plt.show(block=False)

        # dissipation rate ( dEk/dt )
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], -values_1600[:,4], label='Re=1600, 36^3, P=7', lw=lw)
        ax.plot(values_1600_2[:,0], -values_1600_2[:,4], label='Re=1600, 72^3, P=7', lw=lw, ls='--')
        #ax.plot(values_2000[:,0], -values_2000[:,5], label='Re=2000', lw=lw)
        #ax.plot(values_2400[:,0], -values_2400[:,5], label='Re=2400', lw=lw)
        ax.set_title('Dissipation Rate')
        #ax.set_xlim([7.5, 10.5])
        ax.legend()
        ax.set_xlabel('t_c')
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
        nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
        field1 = readnek(nrs_snap_dir + '/snapshots_target/newtgv0.f00010')
        first_element = field1.elem[0]
        print("Type =", type(first_element))
        print(first_element)

        #field2 = readnek(nrs_snap_dir + '/snapshots_interp_1to7/newtgv0.f00010')
        field2 = readnek(nrs_snap_dir + '/snapshots_coarse_7to1/newtgv0.f00010')

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
        #     # asdf

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


        # construct interpolation matrices 
        #pos_fine = (field_fine.elem[eid].pos).reshape((3,-1)).T
        #pos_crse = (field_crse.elem[eid].pos).reshape((3,-1)).T
        eid = 0
        pos_fine = field_fine.elem[eid].pos 
        pos_crse = field_crse.elem[eid].pos


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
            edge_index = torch.tensor(np.loadtxt("./temp/edge_index_element_local_rank_0_size_4").astype(np.int64).T)


            # Plot 
            pos = torch.tensor(first_element.pos).reshape((3, -1)).T
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
        

