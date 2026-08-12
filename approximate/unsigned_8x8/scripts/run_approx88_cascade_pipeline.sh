#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BASE_JSON=${1:-"${ROOT}/best_approx66_inits.json"}
OUT_ROOT=${2:-"${ROOT}/runs88_cascade_$(date +%Y%m%d_%H%M%S)"}
SEED=${3:-500}

PYTHON=${PYTHON:-python3}
CROSS_INIT_MODE=${CROSS_INIT_MODE:-approx62}
TRAIN_MAX_WCE=${TRAIN_MAX_WCE:--1}
FINAL_MAX_WCE=${MAX_WCE:-4500}
ESCAPE_ITERS=${ESCAPE_ITERS:-40}

export PYTHON CROSS_INIT_MODE

echo "[pipeline] ROOT=${ROOT}"
echo "[pipeline] BASE_JSON=${BASE_JSON}"
echo "[pipeline] OUT_ROOT=${OUT_ROOT}"
echo "[pipeline] SEED=${SEED}"
echo "[pipeline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[pipeline] CROSS_INIT_MODE=${CROSS_INIT_MODE}"
echo "[pipeline] TRAIN_MAX_WCE=${TRAIN_MAX_WCE} FINAL_MAX_WCE=${FINAL_MAX_WCE} ESCAPE_ITERS=${ESCAPE_ITERS}"

cd "${ROOT}"

./run_approx88_cascade_01_init_from66.sh \
  "${BASE_JSON}" \
  "${OUT_ROOT}/01_init" \
  "${SEED}"

MAX_WCE="${TRAIN_MAX_WCE}" ./run_approx88_cascade_02_cross_top_train.sh \
  "${OUT_ROOT}/01_init/final_best_approx88_cascade_inits.json" \
  "${OUT_ROOT}/02_cross_top" \
  "$((SEED + 100))"

MAX_WCE="${TRAIN_MAX_WCE}" ./run_approx88_cascade_03_joint_all_train.sh \
  "${OUT_ROOT}/02_cross_top/final_best_approx88_cascade_inits.json" \
  "${OUT_ROOT}/03_joint_all" \
  "$((SEED + 200))"

MAX_WCE="${FINAL_MAX_WCE}" ESCAPE_ITERS="${ESCAPE_ITERS}" ./run_approx88_cascade_04_light_polish.sh \
  "${OUT_ROOT}/03_joint_all/final_best_approx88_cascade_inits.json" \
  "${OUT_ROOT}/04_light_polish" \
  "$((SEED + 300))"

cp "${OUT_ROOT}/04_light_polish/final_best_approx88_cascade_inits.json" "${OUT_ROOT}/final_best_approx88_cascade_inits.json"
cp "${OUT_ROOT}/04_light_polish/final_best_approx88_cascade.v" "${OUT_ROOT}/final_best_approx88_cascade.v"

echo "[pipeline done] ${OUT_ROOT}/final_best_approx88_cascade_inits.json"
echo "[pipeline done] ${OUT_ROOT}/final_best_approx88_cascade.v"
