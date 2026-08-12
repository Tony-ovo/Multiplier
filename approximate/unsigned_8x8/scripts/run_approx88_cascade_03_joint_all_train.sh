#!/usr/bin/env bash
set -euo pipefail

# Stage 03: jointly train inherited low66, HL/LH approx62, and top comp88 LUTs.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

BASE_JSON=${1:-final_best_approx88_cascade_inits.json}
OUT_ROOT=${2:-runs88_cascade_03_joint_all}
SEED=${3:-200}
PY=${PYTHON:-python3}
TRAIN_PY=${TRAIN_PY:-train_approx88_cascade.py}
MAX_WCE=${MAX_WCE:--1}
CROSS_INIT_MODE=${CROSS_INIT_MODE:-approx62}

mkdir -p "${OUT_ROOT}"
cp "${BASE_JSON}" "${OUT_ROOT}/input_best.json"

echo "[run] BASE_JSON=${BASE_JSON}"
echo "[run] OUT_ROOT=${OUT_ROOT}"
echo "[run] SEED=${SEED} MAX_WCE=${MAX_WCE} CROSS_INIT_MODE=${CROSS_INIT_MODE}"

${PY} "${TRAIN_PY}" \
  --stage-name 03_joint_all_cascade_train \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/input_best.json" \
  --cross-init-mode "${CROSS_INIT_MODE}" \
  --train-scope all \
  --epochs 1000 \
  --lr 0.0005 \
  --seed "${SEED}" \
  --init-p 0.96 \
  --noise-std 0.014 \
  --c-init 2.0 \
  --c-out 2.0 \
  --c-anneal \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.00001 \
  --grad-clip 0.8 \
  --restart-from-best-every 150 \
  --restart-init-p 0.975 \
  --restart-noise-std 0.008 \
  --restart-lr-decay 0.82 \
  --min-lr 0.00003 \
  --max-wce "${MAX_WCE}" \
  --single-after \
  --single-rounds 22 \
  --single-mode first \
  --single-random-order \
  --single-lut-names all \
  --pair-after \
  --pair-rounds 3 \
  --pair-mode first \
  --pair-random-order \
  --pair-lut-names cross_top,u_comp23,u_comp89,u_comp4,u_comp5,u_comp6,u_comp7 \
  --pair-max-pairs 70000 \
  --out-dir "${OUT_ROOT}/03_joint_all_train" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/03_joint_all_train/best_approx88_cascade_inits.json" "${OUT_ROOT}/final_best_approx88_cascade_inits.json"
cp "${OUT_ROOT}/03_joint_all_train/best_approx88_cascade.v" "${OUT_ROOT}/final_best_approx88_cascade.v"

echo "[done] ${OUT_ROOT}/final_best_approx88_cascade_inits.json"
echo "[done] ${OUT_ROOT}/final_best_approx88_cascade.v"
