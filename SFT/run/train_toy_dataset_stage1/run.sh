#!/bin/bash

# =====================================================================================
# Script for launching the training on Node 1 of 4.
# This simplified version only includes non-default arguments.
# Assumes the script is executed from the project root directory.
# =====================================================================================
set -e

# --- Node-specific Configuration ---
node_rank=$RANK

# --- Shared Configuration ---
# FIXME: Replace with your master node's IP/hostname
master_addr=$(getent hosts "$MASTER_ADDR" | awk '{ print $1 }' || echo "127.0.0.1")
num_nodes=1
master_port=29500
nproc_per_node=8

# --- Path & Core Configuration ---
model_path='BAGEL-7B-MoT' # replace it with the model path
dataset_config_path='data/configs/train_toy_dataset_stage1.yaml'
project_name='train_toy_dataset_stage1'

# --- Environment Setup ---
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=3600000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=1

export NCCL_TIMEOUT=3600

export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7

export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=2

# --- Training Command ---
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --results_dir "results/${project_name}" \
  --checkpoint_dir "results/${project_name}/checkpoints" \
  --wandb_project "IRG-training" \
  --wandb_name "${project_name}" \
  --auto_resume \
  --resume_from $model_path \
  --finetune_from_ema \
  --finetune_from_hf \
  --log_every 1 \
  --save_every 1000 \
  --warmup_steps 500 \
  --lr 2e-05 \
  --ema 0.995 \
  --timestep_shift 4.0 \
  --expected_num_tokens 40000 \
  --max_num_tokens 40000 \
  --max_num_tokens_per_sample 32768 \
  --num_replicate $num_nodes \
  --dataset_config_file $dataset_config_path \
  --num_workers 8 \
  --model_path $model_path \
  --max_latent_size 64 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --resume_model_only \
  &> "run/${project_name}/log_${node_rank}.txt"
  
  