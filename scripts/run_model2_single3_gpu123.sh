#!/usr/bin/env bash
set -euo pipefail

cd /nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2

mkdir -p logs

run_one () {
  local DB=$1
  local GPU=$2
  local RUN_NAME="model2_${DB}_3class_loso_e100_bs32_adam_lr1e-4_gpu${GPU}"
  local LOG_FILE="logs/${RUN_NAME}.log"

  echo "Starting ${DB} on GPU${GPU}: ${RUN_NAME}"

  (
    export CUDA_VISIBLE_DEVICES=${GPU}

    python train_model2_combined3.py \
      --raf-path "runtime_data/${DB}" \
      --subjects all \
      --epochs 100 \
      --batch-size 32 \
      --num-workers 4 \
      --optimizer adam \
      --lr 1e-4 \
      --device cuda \
      --variant full \
      --run-name "${RUN_NAME}" \
      --pin-memory \
      --no-pretrained-warning

    python summarize_model2_results.py \
      --run-dir "results/${RUN_NAME}" \
      --source-csv "runtime_data/${DB}/3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv"
  ) > "${LOG_FILE}" 2>&1 &
}

run_one casme2 1
run_one samm   2
run_one smic   3

echo "Started:"
echo "  GPU1 -> CASME2 3-class"
echo "  GPU2 -> SAMM 3-class"
echo "  GPU3 -> SMIC 3-class"
echo
echo "Use:"
echo "  tail -f logs/model2_casme2_3class_loso_e100_bs32_adam_lr1e-4_gpu1.log"
echo "  tail -f logs/model2_samm_3class_loso_e100_bs32_adam_lr1e-4_gpu2.log"
echo "  tail -f logs/model2_smic_3class_loso_e100_bs32_adam_lr1e-4_gpu3.log"
