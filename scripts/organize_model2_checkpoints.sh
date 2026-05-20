#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2"
RESULTS_DIR="${PROJECT_ROOT}/results"
CKPT_ROOT="/nfs/users/lixinglin/projects/checkpoint/microemotion/model2"
MASTER_INDEX="${RESULTS_DIR}/checkpoint_location_model2.txt"

mkdir -p "${CKPT_ROOT}"

{
  echo "Model 2 checkpoint location:"
  echo
  echo "Root:"
  echo "${CKPT_ROOT}"
  echo
  echo "Run mapping:"
} > "${MASTER_INDEX}"

find "${RESULTS_DIR}" -mindepth 2 -maxdepth 3 -type d -name checkpoints | sort | while read -r SRC_CKPT_DIR; do
  RUN_DIR="$(dirname "${SRC_CKPT_DIR}")"
  RUN_NAME="$(basename "${RUN_DIR}")"
  DEST_RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
  DEST_CKPT_DIR="${DEST_RUN_DIR}/checkpoints"

  mkdir -p "${DEST_CKPT_DIR}"

  NUM_PTH=$(find "${SRC_CKPT_DIR}" -maxdepth 1 -type f -name "*.pth" | wc -l || true)

  if [ "${NUM_PTH}" -eq 0 ]; then
    continue
  fi

  echo "Moving ${NUM_PTH} pth files:"
  echo "  from: ${SRC_CKPT_DIR}"
  echo "  to:   ${DEST_CKPT_DIR}"

  find "${SRC_CKPT_DIR}" -maxdepth 1 -type f -name "*.pth" -print0 \
    | xargs -0 -I{} mv -n "{}" "${DEST_CKPT_DIR}/"

  {
    echo "Model 2 checkpoint location:"
    echo
    echo "Run:"
    echo "${RUN_NAME}"
    echo
    echo "Checkpoint directory:"
    echo "${DEST_CKPT_DIR}"
    echo
    echo "Original results directory:"
    echo "${RUN_DIR}"
    echo
    echo "Note:"
    echo "Checkpoints are stored outside the project repository to keep the code repository lightweight."
  } > "${RUN_DIR}/checkpoint_location.txt"

  {
    echo
    echo "${RUN_NAME}:"
    echo "  results:     ${RUN_DIR}"
    echo "  checkpoints: ${DEST_CKPT_DIR}"
  } >> "${MASTER_INDEX}"

  # Replace empty local checkpoints directory with symlink to external checkpoint directory.
  if [ -d "${SRC_CKPT_DIR}" ] && [ -z "$(find "${SRC_CKPT_DIR}" -mindepth 1 -maxdepth 1 | head -1)" ]; then
    rmdir "${SRC_CKPT_DIR}"
    ln -s "${DEST_CKPT_DIR}" "${SRC_CKPT_DIR}"
  fi
done

{
  echo
  echo "Note:"
  echo "Checkpoints are stored outside the project repository to keep the code repository lightweight."
} >> "${MASTER_INDEX}"

echo "Done."
echo "Master checkpoint index:"
echo "${MASTER_INDEX}"
