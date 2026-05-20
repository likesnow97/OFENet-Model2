#!/usr/bin/env bash
set -euo pipefail

cd /nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2

export CUDA_VISIBLE_DEVICES=0

VARIANTS=("full" "no_pretrain" "no_rfr" "no_adaf" "no_rfr_no_adaf")

for VAR in "${VARIANTS[@]}"; do
  echo "========== Ablation variant: ${VAR} on GPU0 =========="

  python train_model2_combined3.py \
    --subjects all \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 4 \
    --optimizer adam \
    --lr 1e-4 \
    --device cuda \
    --variant "${VAR}" \
    --run-name "ablation_combined3_${VAR}_loso_e100_bs32_adam_lr1e-4_gpu0" \
    --pin-memory \
    --no-pretrained-warning

  python summarize_model2_results.py \
    --run-dir "results/ablation_combined3_${VAR}_loso_e100_bs32_adam_lr1e-4_gpu0"
done
