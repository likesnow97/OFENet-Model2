#!/usr/bin/env bash
set -euo pipefail

cd /nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2

python train_model2_combined3.py \
  --subjects all \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --optimizer adam \
  --lr 1e-4 \
  --device cuda \
  --run-name model2_combined3_loso_e100_bs32_adam_lr1e-4 \
  --pin-memory \
  --no-pretrained-warning
