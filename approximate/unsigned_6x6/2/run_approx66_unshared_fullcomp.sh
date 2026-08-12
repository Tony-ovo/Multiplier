#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_approx66_unshared_fullcomp.sh [BASE_JSON] [OUT_ROOT]
#
# BASE_JSON can be:
#   - old shared best_approx66_inits.json from train_approx66_ste_search.py
#   - new unshared/fullcomp best_approx66_inits.json from this script
# If omitted, it uses ./best_approx66_inits.json.

BASE_JSON=${1:-best_approx66_inits.json}
OUT_ROOT=${2:-runs_unshared_fullcomp_$(date +%Y%m%d_%H%M%S)}
PY=${PYTHON:-python3}

if [[ ! -f "${BASE_JSON}" ]]; then
  echo "ERROR: BASE_JSON not found: ${BASE_JSON}"
  echo "Usage: CUDA_VISIBLE_DEVICES=0 $0 [BASE_JSON] [OUT_ROOT]"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_PY="${SCRIPT_DIR}/train_approx66_unshared_fullcomp.py"

mkdir -p "${OUT_ROOT}"

echo "BASE_JSON = ${BASE_JSON}"
echo "OUT_ROOT  = ${OUT_ROOT}"
echo "TRAIN_PY  = ${TRAIN_PY}"

# ------------------------------------------------------------
# Stage 1: expand shared approx62 into low/mid/high + train comp OR LUTs.
# Small lr/noise: preserve the good old solution, then allow divergence.
# ------------------------------------------------------------
${PY} "${TRAIN_PY}" \
  --stage-name 01_expand_unshared_fullcomp \
  --epochs 900 \
  --lr 0.001 \
  --seed 10 \
  --init-mode json \
  --base-inits-json "${BASE_JSON}" \
  --init-p 0.92 \
  --noise-std 0.03 \
  --c-init 1.8 \
  --c-out 1.8 \
  --c-anneal \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.00002 \
  --grad-clip 1.0 \
  --restart-from-best-every 150 \
  --restart-init-p 0.94 \
  --restart-noise-std 0.02 \
  --restart-lr-decay 0.85 \
  --min-lr 0.00005 \
  --bitflip-after \
  --bitflip-rounds 20 \
  --bitflip-random-order \
  --bitflip-mode first \
  --out-dir "${OUT_ROOT}/01_expand" \
  --log-file terminal_log.txt

# ------------------------------------------------------------
# Stage 2: refine from Stage 1 best. Lower lr, weaker noise.
# ------------------------------------------------------------
${PY} "${TRAIN_PY}" \
  --stage-name 02_refine_unshared_fullcomp \
  --epochs 900 \
  --lr 0.0006 \
  --seed 11 \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/01_expand/best_approx66_inits.json" \
  --init-p 0.95 \
  --noise-std 0.015 \
  --c-init 2.0 \
  --c-out 2.0 \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.00001 \
  --grad-clip 0.8 \
  --restart-from-best-every 150 \
  --restart-init-p 0.96 \
  --restart-noise-std 0.01 \
  --restart-lr-decay 0.85 \
  --min-lr 0.00003 \
  --bitflip-after \
  --bitflip-rounds 20 \
  --bitflip-random-order \
  --bitflip-mode first \
  --out-dir "${OUT_ROOT}/02_refine" \
  --log-file terminal_log.txt

# ------------------------------------------------------------
# Stage 3: bitflip-only polish with best-improvement.
# This is slower, but on the already-good result it is useful.
# ------------------------------------------------------------
${PY} "${TRAIN_PY}" \
  --stage-name 03_bitflip_best_improvement \
  --seed 12 \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/02_refine/best_approx66_inits.json" \
  --bitflip-only \
  --bitflip-rounds 60 \
  --bitflip-mode best \
  --out-dir "${OUT_ROOT}/03_bitflip_best" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/03_bitflip_best/best_approx66_inits.json" "${OUT_ROOT}/final_best_approx66_inits.json"
cp "${OUT_ROOT}/03_bitflip_best/best_approx66_unshared_fullcomp.v" "${OUT_ROOT}/final_best_approx66_unshared_fullcomp.v"

echo ""
echo "Done. Final files:"
echo "  ${OUT_ROOT}/final_best_approx66_inits.json"
echo "  ${OUT_ROOT}/final_best_approx66_unshared_fullcomp.v"
echo "  ${OUT_ROOT}/03_bitflip_best/terminal_log.txt"
