#!/usr/bin/env bash
set -euo pipefail

# Stage 02: freeze inherited low66, train HL/LH approx62 plus top comp88 LUTs.
# Random-order first-improvement post search is intentional so different seeds can diverge.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

BASE_JSON=${1:-final_best_approx88_cascade_inits.json}
OUT_ROOT=${2:-runs88_cascade_02_cross_top}
SEED=${3:-100}
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
  --stage-name 02_cross_top_train \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/input_best.json" \
  --cross-init-mode "${CROSS_INIT_MODE}" \
  --train-scope cross_top \
  --epochs 1200 \
  --lr 0.0008 \
  --seed "${SEED}" \
  --init-p 0.94 \
  --noise-std 0.025 \
  --c-init 1.6 \
  --c-out 1.6 \
  --c-anneal \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.00002 \
  --grad-clip 1.0 \
  --restart-from-best-every 160 \
  --restart-init-p 0.96 \
  --restart-noise-std 0.012 \
  --restart-lr-decay 0.8 \
  --min-lr 0.00004 \
  --max-wce "${MAX_WCE}" \
  --single-after \
  --single-rounds 18 \
  --single-mode first \
  --single-random-order \
  --single-lut-names cross_top \
  --pair-after \
  --pair-rounds 2 \
  --pair-mode first \
  --pair-random-order \
  --pair-lut-names cross_top \
  --pair-max-pairs 40000 \
  --out-dir "${OUT_ROOT}/02_cross_top_train" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/02_cross_top_train/best_approx88_cascade_inits.json" "${OUT_ROOT}/final_best_approx88_cascade_inits.json"
cp "${OUT_ROOT}/02_cross_top_train/best_approx88_cascade.v" "${OUT_ROOT}/final_best_approx88_cascade.v"

echo "[done] ${OUT_ROOT}/final_best_approx88_cascade_inits.json"
echo "[done] ${OUT_ROOT}/final_best_approx88_cascade.v"
