#!/bin/sh
TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

# Other flags 
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=NUMA
export FI_OFI_RXM_RX_SIZE=65536
export FI_CXI_RX_MATCH_MODE=hybrid

# Get number of ranks 
NUM_NODES=$(wc -l < "${PBS_NODEFILE}")

# Get number of GPUs per node
NGPUS_PER_NODE=$(nvidia-smi -L | wc -l)

# Get total number of GPUs 
NGPUS="$((${NUM_NODES}*${NGPUS_PER_NODE}))"

# Print 
echo $NUM_NODES $NGPUS_PER_NODE $NGPUS

# run 
mpiexec -n $NGPUS --ppn $NGPUS_PER_NODE -d 8 --cpu-bind depth \
./set_affinity_gpu_polaris.sh \
python3 main.py \
hidden_channels=128 \
n_mlp_hidden_layers=2 \
n_messagePassing_layers=2 \
lr_init=0.0001 \
n_element_neighbors=6 \
model_name=GNN_TGV \
data_dir="${PWD}/datasets/tgv_nei_6_re_1600/" \
model_dir="${PWD}/saved_models/tgv_re_1600/" \
epochs=1 \
batch_size=4 \
test_batch_size=4 \
use_residual=True \
use_fine_messagePassing=True \
restart=False
