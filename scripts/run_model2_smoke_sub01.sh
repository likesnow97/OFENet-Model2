#!/usr/bin/env bash
set -euo pipefail

cd /nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2

python train_model2_combined3.py \
  --subjects sub01 \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 0 \
  --optimizer adam \
  --lr 1e-4 \
  --device cuda \
  --run-name smoke_sub01_epoch1 \
  --pin-memory \
  --no-pretrained-warning
