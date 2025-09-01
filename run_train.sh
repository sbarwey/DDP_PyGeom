#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module load frameworks
source /lus/flare/projects/SCIML-CFD/sbarwey/codes/nek/nekRS-ML-devel/examples/tgv_gnn_traj_offline_checkpoint/_pyg/bin/activate
module list

export NEKRS_HOME=/home/sbarwey/.local/nekrs
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0

export CCL_ALLTOALLV_MONOLITHIC_KERNEL=0

# CPU BIND: from https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/#code-changes-to-train-on-multiple-gpus-using-ddp
export CPU_BIND="list:4:9:14:19:20:25:56:61:66:71:74:79" # 12 ppn to 12 cores
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"


# TEST - bfs/acv, Re=1600 - poly 3, 20 snaps, 12 ranks per case
MASTER_ADDR=$(hostname -f)
echo $MASTER_ADDR > master_addr.txt
case_path=/lus/flare/projects/SCIML-CFD/sbarwey/codes/nek/examples_v24_gnn
mpiexec -n 24 -ppn 12 --cpu-bind=${CPU_BIND} python main.py backend=ccl \
gnn_outputs_path=[${case_path}/bfs_2_rampup/gnn_outputs/gnn_outputs_poly_3,${case_path}/cavity_rampup/gnn_outputs/gnn_outputs_poly_3] \
traj_data_path=[${case_path}/bfs_2_rampup/Re_1600_p3/traj_poly_3/20_snaps/tinit_0.000000_dtfactor_1,${case_path}/cavity_rampup/Re_1600_p3/traj_poly_3/20_snaps/tinit_0.000000_dtfactor_1] \
size_list=[12,12] \
reynolds_number_list=[1600,1600] \
hidden_channels=128 \
n_mlp_hidden_layers=2 \
n_messagePassing_layers=8 \
layer_norm=True \
use_residual=True \
halo_swap_mode=all_to_all_opt \
phase1_steps=5 \
phase2_steps=5 \
phase3_steps=0 \
lr_phase12=0.0001 \
lr_phase23=0.000003 \
max_rollout_steps=1 \
verbose=False \
restart=False \
ckptfreq=5 \
master_addr=$MASTER_ADDR
