#!/usr/bin/env bash
# Teacher-forcing (XE) stage on Kaggle T4 x2. Run from the repo root.
#   bash kaggle/run_train.sh              # fresh start
#   bash kaggle/run_train.sh --resume     # continue from /kaggle/working/cosnet_output
set -e

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
# Kaggle gives 4 vCPUs; 2 DDP procs x 2 workers already saturates them.
export OMP_NUM_THREADS=2

python -u train_net.py \
    --num-gpus 2 \
    --config-file kaggle/cosnet_kaggle.yaml \
    "$@"
