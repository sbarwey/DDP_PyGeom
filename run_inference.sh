#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module restore
module load frameworks
source {YOUR_VENV_PATH} 
module list

export NEKRS_HOME=/home/sbarwey/.local/nekrs
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
export UR_L0_USE_COPY_ENGINE=0

export CCL_ALLTOALLV_MONOLITHIC_KERNEL=0

# CPU BIND: from https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/#code-changes-to-train-on-multiple-gpus-using-ddp
export CPU_BIND="list:4:9:14:19:20:25:56:61:66:71:74:79" # 12 ppn to 12 cores
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"

# Inference
MASTER_ADDR=$(hostname -f)
echo $MASTER_ADDR > master_addr.txt
case_path={YOUR_CASE_PATH}
mpiexec -n 12 -ppn 12 --cpu-bind=${CPU_BIND} python inference.py backend=ccl \
model_task=inference \
gnn_outputs_path=[${case_path}/{YOUR_GNNOUTPUTS_PATH}] \
traj_data_path=[${case_path}/{YOUR_TRAJ_DATA_PATH}] \
load_stats=False \
size_list=[12] \
reynolds_number_list=[1600] \
reynolds_number_inference=1600 \
hidden_channels=128 \
n_mlp_hidden_layers=2 \
n_messagePassing_layers=8 \
layer_norm=True \
use_residual=True \
halo_swap_mode=all_to_all_opt \
lr_phase12=0.0001 \
lr_phase23=0.000003 \
max_rollout_steps=1 \
consistency=True \
halo_swap_mode=all_to_all_opt \
verbose=False \
restart=False \
master_addr=$MASTER_ADDR
