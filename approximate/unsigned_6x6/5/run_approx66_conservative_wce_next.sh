#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_approx66_conservative_wce_next.sh best_mred_0108535.json runs_conservative_0108535
#
# Requirements:
#   train_approx66_unshared_paircomp.py
#   train_approx66_conservative_wce.py
#   must be in the same directory.

BASE_JSON=${1:-best_approx66_inits.json}
OUT_ROOT=${2:-runs_conservative_wce_$(date +%Y%m%d_%H%M%S)}
PY=${PYTHON:-python3}

mkdir -p "${OUT_ROOT}"
echo "BASE_JSON=${BASE_JSON}"
echo "OUT_ROOT=${OUT_ROOT}"

# Stage 1: top-error guided constrained single + pair search.
# Keep WCE <= 930. This protects your current key advantage.
${PY} train_approx66_conservative_wce.py \
  --stage-name 01_top_guided_wce930 \
  --base-inits-json "${BASE_JSON}" \
  --out-dir "${OUT_ROOT}/01_top_guided_wce930" \
  --seed 101 \
  --max-wce 930 \
  --topk 120 \
  --print-topk 30 \
  --candidate-mode top_neutral \
  --neutral-top 180 \
  --neutral-margin 0.003 \
  --do-single \
  --single-rounds 30 \
  --single-mode best \
  --do-pair \
  --pair-rounds 6 \
  --pair-mode first \
  --pair-random-order \
  --pair-max-pairs 150000 \
  --do-final-single \
  --log-file terminal_log.txt

# Stage 2: neutral multi-bit escape under WCE<=930.
${PY} train_approx66_conservative_wce.py \
  --stage-name 02_neutral_escape_wce930 \
  --base-inits-json "${OUT_ROOT}/01_top_guided_wce930/best_approx66_inits.json" \
  --out-dir "${OUT_ROOT}/02_neutral_escape_wce930" \
  --seed 202 \
  --max-wce 930 \
  --topk 160 \
  --print-topk 30 \
  --candidate-mode top_neutral \
  --neutral-top 220 \
  --neutral-margin 0.005 \
  --do-escape \
  --escape-iters 160 \
  --kmin 1 \
  --kmax 5 \
  --escape-single-rounds 10 \
  --escape-pair-after \
  --escape-pair-rounds 2 \
  --escape-pair-max-pairs 80000 \
  --do-final-single \
  --single-rounds 40 \
  --single-mode best \
  --log-file terminal_log.txt

# Stage 3: optional slightly relaxed WCE search, useful if WCE=930 is too tight.
# It may find lower MRED with WCE <= 1000. Keep this as a separate candidate, not overwrite the WCE930 best.
${PY} train_approx66_conservative_wce.py \
  --stage-name 03_relaxed_wce1000_candidate \
  --base-inits-json "${OUT_ROOT}/02_neutral_escape_wce930/best_approx66_inits.json" \
  --out-dir "${OUT_ROOT}/03_relaxed_wce1000_candidate" \
  --seed 303 \
  --max-wce 1000 \
  --topk 180 \
  --print-topk 30 \
  --candidate-mode top_neutral \
  --neutral-top 260 \
  --neutral-margin 0.006 \
  --do-single \
  --single-rounds 30 \
  --single-mode best \
  --do-pair \
  --pair-rounds 8 \
  --pair-mode first \
  --pair-random-order \
  --pair-max-pairs 200000 \
  --do-escape \
  --escape-iters 120 \
  --kmin 1 \
  --kmax 6 \
  --escape-single-rounds 8 \
  --escape-pair-after \
  --escape-pair-rounds 2 \
  --escape-pair-max-pairs 80000 \
  --do-final-single \
  --log-file terminal_log.txt

# Copy the strict WCE<=930 best as final strict result.
cp "${OUT_ROOT}/02_neutral_escape_wce930/best_approx66_inits.json" "${OUT_ROOT}/final_wce930_best_approx66_inits.json"
cp "${OUT_ROOT}/02_neutral_escape_wce930/best_approx66_unshared_paircomp.v" "${OUT_ROOT}/final_wce930_best_approx66_unshared_paircomp.v"

# Copy relaxed result separately.
cp "${OUT_ROOT}/03_relaxed_wce1000_candidate/best_approx66_inits.json" "${OUT_ROOT}/final_wce1000_candidate_approx66_inits.json"
cp "${OUT_ROOT}/03_relaxed_wce1000_candidate/best_approx66_unshared_paircomp.v" "${OUT_ROOT}/final_wce1000_candidate_approx66_unshared_paircomp.v"

echo ""
echo "Done. Key outputs:"
echo "  Strict WCE<=930 JSON: ${OUT_ROOT}/final_wce930_best_approx66_inits.json"
echo "  Strict WCE<=930 Verilog: ${OUT_ROOT}/final_wce930_best_approx66_unshared_paircomp.v"
echo "  Relaxed WCE<=1000 JSON: ${OUT_ROOT}/final_wce1000_candidate_approx66_inits.json"
echo "  Relaxed WCE<=1000 Verilog: ${OUT_ROOT}/final_wce1000_candidate_approx66_unshared_paircomp.v"
