#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   chmod +x run_approx66_explore_refine.sh
#   ./run_approx66_explore_refine.sh
# or:
#   CUDA_VISIBLE_DEVICES=0 ./run_approx66_explore_refine.sh my_run_dir

OUT_ROOT=${1:-runs_approx66_search_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-python3}

echo "Output root: ${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

# ============================================================
# Stage 1: exploration
# Random hard INIT base + relatively large lr/noise.
# Current loss may fluctuate. Judge this stage by best_MRED only.
# ============================================================

# init-mode manual: use DEFAULT_APPROX62 / DEFAULT_COMP66
# init-mode random: use random hard INIT base

${PYTHON} train_approx66_ste_search.py \
  --stage-name explore \
  --epochs 4000 \
  --lr 0.01 \
  --seed 0 \
  --init-mode manual \
  --init-p 0.70 \
  --noise-std 0.25 \
  --c-init 2.0 \
  --c-out 2.0 \
  --c-anneal \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.0001 \
  --grad-clip 5.0 \
  --restart-from-best-every 0 \
  --out-dir "${OUT_ROOT}/01_explore" \
  --log-file terminal_log.txt

# ============================================================
# Stage 2: fine search
# Restart from exploration best + smaller lr/noise + periodic best restart.
# The terminal output is mirrored exactly to terminal_log.txt.
# ============================================================
${PYTHON} train_approx66_ste_search.py \
  --stage-name refine \
  --epochs 1000 \
  --lr 0.002 \
  --seed 1 \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/01_explore/best_approx66_inits.json" \
  --init-p 0.85 \
  --noise-std 0.05 \
  --c-init 2.0 \
  --c-out 2.0 \
  --zero-weight 0.01 \
  --med-weight 0.0 \
  --bin-weight 0.00005 \
  --grad-clip 1.0 \
  --restart-from-best-every 120 \
  --restart-init-p 0.88 \
  --restart-noise-std 0.03 \
  --restart-lr-decay 0.75 \
  --min-lr 0.0001 \
  --bitflip-after \
  --bitflip-rounds 3 \
  --out-dir "${OUT_ROOT}/02_refine" \
  --log-file terminal_log.txt

# Convenience copy of final best files.
cp "${OUT_ROOT}/02_refine/best_approx66_inits.json" "${OUT_ROOT}/final_best_approx66_inits.json"
cp "${OUT_ROOT}/02_refine/best_approx66_verilog_snippet.v" "${OUT_ROOT}/final_best_approx66_verilog_snippet.v"

echo "Done."
echo "Explore log: ${OUT_ROOT}/01_explore/terminal_log.txt"
echo "Refine  log: ${OUT_ROOT}/02_refine/terminal_log.txt"
echo "Final best JSON: ${OUT_ROOT}/final_best_approx66_inits.json"
echo "Final Verilog snippet: ${OUT_ROOT}/final_best_approx66_verilog_snippet.v"
