#!/usr/bin/env bash
# Self-critical stage. Run from the repo root.
set -e

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

python -u train_net.py \
    --num-gpus 2 \
    --config-file kaggle/cosnet_rl_kaggle.yaml \
    "$@"
