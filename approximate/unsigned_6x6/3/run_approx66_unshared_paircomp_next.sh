#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_approx66_unshared_paircomp_next.sh input_best.json runs_paircomp_next
# If no args are given, it reads ./best_approx66_inits.json and writes ./runs_paircomp_next_<timestamp>.

BASE_JSON="${1:-best_approx66_inits.json}"
OUT_ROOT="${2:-runs_paircomp_next_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PY="${TRAIN_PY:-train_approx66_unshared_paircomp.py}"

mkdir -p "${OUT_ROOT}"
cp "${BASE_JSON}" "${OUT_ROOT}/input_best.json"

echo "[run] BASE_JSON=${BASE_JSON}"
echo "[run] OUT_ROOT=${OUT_ROOT}"
echo "[run] TRAIN_PY=${TRAIN_PY}"

# ------------------------------------------------------------
# Stage 01: Pair-aware comp training from current best.
# This changes comp66 prod[4:7] from independent 3-input OR/LUTs
# to pair-aware 6-input LUTs, while keeping LOW/MID/HIGH approx62 unshared.
# ------------------------------------------------------------
"${PYTHON_BIN}" "${TRAIN_PY}" \
  --stage-name 01_paircomp_train \
  --epochs 800 \
  --lr 0.0008 \
  --seed 10 \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/input_best.json" \
  --init-p 0.92 \
  --noise-std 0.02 \
  --c-init 2.0 \
  --c-out 2.0 \
  --c-anneal \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.00002 \
  --grad-clip 1.0 \
  --restart-from-best-every 150 \
  --restart-init-p 0.94 \
  --restart-noise-std 0.015 \
  --restart-lr-decay 0.8 \
  --min-lr 0.00005 \
  --single-after \
  --single-rounds 20 \
  --single-mode best \
  --out-dir "${OUT_ROOT}/01_paircomp_train" \
  --log-file terminal_log.txt

# ------------------------------------------------------------
# Stage 02: Targeted pair-bit flip.
# Pair flip is expensive, so by default it targets the new pair-aware comp LUTs
# and the most influential approx62 LUTs. You can widen --pair-lut-names later.
# ------------------------------------------------------------
"${PYTHON_BIN}" "${TRAIN_PY}" \
  --stage-name 02_target_pair_flip \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/01_paircomp_train/best_approx66_inits.json" \
  --pair-only \
  --pair-rounds 8 \
  --pair-mode first \
  --pair-random-order \
  --pair-max-pairs 120000 \
  --pair-lut-names u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut3,mid_lut3,high_lut3,low_lut2,mid_lut2,high_lut2 \
  --seed 20 \
  --out-dir "${OUT_ROOT}/02_target_pair_flip" \
  --log-file terminal_log.txt

# ------------------------------------------------------------
# Stage 03: Basin hopping.
# Randomly flips 2~5 bits, then performs single-bit local search.
# This helps escape a single-bit local optimum.
# ------------------------------------------------------------
"${PYTHON_BIN}" "${TRAIN_PY}" \
  --stage-name 03_basin_hop \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/02_target_pair_flip/best_approx66_inits.json" \
  --basin-only \
  --basin-iters 30 \
  --basin-flip-min 2 \
  --basin-flip-max 5 \
  --basin-single-rounds 8 \
  --single-mode best \
  --basin-lut-names u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut3,mid_lut3,high_lut3,low_lut2,mid_lut2,high_lut2 \
  --seed 30 \
  --out-dir "${OUT_ROOT}/03_basin_hop" \
  --log-file terminal_log.txt

# ------------------------------------------------------------
# Stage 04: Final single-bit best-improvement polishing.
# ------------------------------------------------------------
"${PYTHON_BIN}" "${TRAIN_PY}" \
  --stage-name 04_final_single_polish \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/03_basin_hop/best_approx66_inits.json" \
  --single-only \
  --single-rounds 30 \
  --single-mode best \
  --single-lut-names all \
  --seed 40 \
  --out-dir "${OUT_ROOT}/04_final_single_polish" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/04_final_single_polish/best_approx66_inits.json" "${OUT_ROOT}/final_best_approx66_inits.json"
cp "${OUT_ROOT}/04_final_single_polish/best_approx66_unshared_paircomp.v" "${OUT_ROOT}/final_best_approx66_unshared_paircomp.v"

echo "[done] Final JSON:    ${OUT_ROOT}/final_best_approx66_inits.json"
echo "[done] Final Verilog: ${OUT_ROOT}/final_best_approx66_unshared_paircomp.v"
