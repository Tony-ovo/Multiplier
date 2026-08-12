#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BASE_JSON=${BASE_JSON:-"${ROOT}/best_approx66_inits.json"}
SEEDS=${SEEDS:-"500 800 1100 1400"}
GPU_LIST=${GPU_LIST:-"0 1"}
OUT_PREFIX=${OUT_PREFIX:-"runs88_cascade"}
CROSS_INIT_MODE=${CROSS_INIT_MODE:-approx62}
TRAIN_MAX_WCE=${TRAIN_MAX_WCE:--1}
MAX_WCE=${MAX_WCE:-4500}
ESCAPE_ITERS=${ESCAPE_ITERS:-40}
PYTHON=${PYTHON:-python3}
MAX_PARALLEL=${MAX_PARALLEL:-0}

read -r -a seed_arr <<< "${SEEDS}"
read -r -a gpu_arr <<< "${GPU_LIST}"

if [ "${#gpu_arr[@]}" -eq 0 ]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 2
fi

export BASE_JSON CROSS_INIT_MODE TRAIN_MAX_WCE MAX_WCE ESCAPE_ITERS PYTHON

echo "[multi] ROOT=${ROOT}"
echo "[multi] BASE_JSON=${BASE_JSON}"
echo "[multi] SEEDS=${SEEDS}"
echo "[multi] GPU_LIST=${GPU_LIST}"
echo "[multi] OUT_PREFIX=${OUT_PREFIX}"
echo "[multi] CROSS_INIT_MODE=${CROSS_INIT_MODE}"
echo "[multi] MAX_PARALLEL=${MAX_PARALLEL} (0 means launch all seeds at once)"

cd "${ROOT}"

launched=0
for S in "${seed_arr[@]}"; do
  GPU=${gpu_arr[$((launched % ${#gpu_arr[@]}))]}
  OUT_ROOT="${ROOT}/${OUT_PREFIX}_s${S}"
  mkdir -p "${OUT_ROOT}"
  LOG="${OUT_ROOT}/pipeline.log"

  echo "[launch] seed=${S} gpu=${GPU} out=${OUT_ROOT} log=${LOG}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    ./run_approx88_cascade_pipeline.sh "${BASE_JSON}" "${OUT_ROOT}" "${S}"
  ) > "${LOG}" 2>&1 &

  launched=$((launched + 1))
  if [ "${MAX_PARALLEL}" -gt 0 ] && [ $((launched % MAX_PARALLEL)) -eq 0 ]; then
    wait
  fi
done

wait
echo "[multi done]"
