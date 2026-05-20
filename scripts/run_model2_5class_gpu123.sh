#!/usr/bin/env bash
set -euo pipefail

cd /nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2
mkdir -p logs

# GPU1: CASME2 single-dataset 5-class
CUDA_VISIBLE_DEVICES=1 nohup python train_model2_single5.py \
  --db casme2 \
  --raf-path /nfs/users/lixinglin/data/microemotion \
  --subjects all \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --optimizer adam \
  --lr 1e-4 \
  --device cuda \
  --variant full \
  --run-name model2_casme2_5class_loso_e100_bs32_adam_lr1e-4_gpu1 \
  --pin-memory \
  --no-pretrained-warning \
  > logs/model2_casme2_5class_loso_e100_bs32_adam_lr1e-4_gpu1.log 2>&1 &

# GPU2: SAMM single-dataset 5-class
CUDA_VISIBLE_DEVICES=2 nohup python train_model2_single5.py \
  --db samm \
  --raf-path /nfs/users/lixinglin/data/microemotion \
  --subjects all \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --optimizer adam \
  --lr 1e-4 \
  --device cuda \
  --variant full \
  --run-name model2_samm_5class_loso_e100_bs32_adam_lr1e-4_gpu2 \
  --pin-memory \
  --no-pretrained-warning \
  > logs/model2_samm_5class_loso_e100_bs32_adam_lr1e-4_gpu2.log 2>&1 &

# GPU3: CASME2+SAMM mixed 5-class
CUDA_VISIBLE_DEVICES=3 nohup python train_model2_combined3.py \
  --raf-path runtime_data/mixed5_casme2_samm \
  --subjects all \
  --num-classes 5 \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --optimizer adam \
  --lr 1e-4 \
  --device cuda \
  --variant full \
  --run-name model2_mixed5_casme2_samm_loso_e100_bs32_adam_lr1e-4_gpu3 \
  --pin-memory \
  --no-pretrained-warning \
  > logs/model2_mixed5_casme2_samm_loso_e100_bs32_adam_lr1e-4_gpu3.log 2>&1 &

echo "Started:"
echo "  GPU1 -> CASME2 single 5-class"
echo "  GPU2 -> SAMM single 5-class"
echo "  GPU3 -> CASME2+SAMM mixed 5-class"
echo
echo "Logs:"
echo "  logs/model2_casme2_5class_loso_e100_bs32_adam_lr1e-4_gpu1.log"
echo "  logs/model2_samm_5class_loso_e100_bs32_adam_lr1e-4_gpu2.log"
echo "  logs/model2_mixed5_casme2_samm_loso_e100_bs32_adam_lr1e-4_gpu3.log"
