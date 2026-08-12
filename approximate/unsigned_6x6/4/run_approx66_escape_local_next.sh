#!/usr/bin/env bash
set -euo pipefail

BASE_JSON=${1:-best_approx66_inits.json}
OUT_ROOT=${2:-runs_escape_local_next}
PY=${PY:-python3}

mkdir -p "${OUT_ROOT}"

# This script assumes train_approx66_unshared_paircomp.py and
# train_approx66_escape_local.py are in the current directory.

# Stage 1: broad neutral-kflip search over all effective bits.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ${PY} train_approx66_escape_local.py \
  --stage-name 01_broad_neutral_escape \
  --base-inits-json "${BASE_JSON}" \
  --seed 100 \
  --top-cases 30 \
  --iters 80 \
  --pool-lut-names all \
  --neutral-top 220 \
  --refresh-every 10 \
  --kmin 2 \
  --kmax 8 \
  --single-lut-names all \
  --single-rounds 8 \
  --single-mode first \
  --pair-lut-names u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut2,low_lut3,mid_lut2,mid_lut3,high_lut2,high_lut3 \
  --pair-rounds 1 \
  --pair-max-pairs 60000 \
  --beam-size 10 \
  --out-dir "${OUT_ROOT}/01_broad_escape" \
  --log-file terminal_log.txt

# Stage 2: focus on comp + lut2/lut3, where most improvements usually occur.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ${PY} train_approx66_escape_local.py \
  --stage-name 02_comp_lut23_escape \
  --base-inits-json "${OUT_ROOT}/01_broad_escape/final_best_approx66_inits.json" \
  --seed 200 \
  --top-cases 30 \
  --iters 100 \
  --pool-lut-names u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut2,low_lut3,mid_lut2,mid_lut3,high_lut2,high_lut3 \
  --neutral-top 180 \
  --refresh-every 8 \
  --kmin 2 \
  --kmax 10 \
  --single-lut-names all \
  --single-rounds 10 \
  --single-mode first \
  --pair-lut-names u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut2,low_lut3,mid_lut2,mid_lut3,high_lut2,high_lut3 \
  --pair-rounds 2 \
  --pair-max-pairs 90000 \
  --beam-size 10 \
  --out-dir "${OUT_ROOT}/02_comp_lut23_escape" \
  --log-file terminal_log.txt

# Stage 3: low/MID/HIGH segment-specific escape. MRED is often sensitive to low exact products.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ${PY} train_approx66_escape_local.py \
  --stage-name 03_segment_escape \
  --base-inits-json "${OUT_ROOT}/02_comp_lut23_escape/final_best_approx66_inits.json" \
  --seed 300 \
  --top-cases 30 \
  --iters 80 \
  --pool-lut-names low_lut1,low_lut2,low_lut3,low_lut4,mid_lut1,mid_lut2,mid_lut3,mid_lut4,high_lut1,high_lut2,high_lut3,high_lut4 \
  --neutral-top 180 \
  --refresh-every 8 \
  --kmin 2 \
  --kmax 8 \
  --single-lut-names all \
  --single-rounds 10 \
  --single-mode first \
  --pair-lut-names low_lut2,low_lut3,mid_lut2,mid_lut3,high_lut2,high_lut3,u_comp23,u_comp89 \
  --pair-rounds 2 \
  --pair-max-pairs 90000 \
  --beam-size 10 \
  --out-dir "${OUT_ROOT}/03_segment_escape" \
  --log-file terminal_log.txt

# Stage 4: exact best single-bit polish to finish.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ${PY} train_approx66_unshared_paircomp.py \
  --stage-name 04_final_best_single \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/03_segment_escape/final_best_approx66_inits.json" \
  --single-only \
  --single-rounds 80 \
  --single-mode best \
  --single-lut-names all \
  --seed 400 \
  --out-dir "${OUT_ROOT}/04_final_single" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/04_final_single/best_approx66_inits.json" "${OUT_ROOT}/final_best_approx66_inits.json"
cp "${OUT_ROOT}/04_final_single/best_approx66_unshared_paircomp.v" "${OUT_ROOT}/final_best_approx66_unshared_paircomp.v"

echo ""
echo "Final results:"
echo "  ${OUT_ROOT}/final_best_approx66_inits.json"
echo "  ${OUT_ROOT}/final_best_approx66_unshared_paircomp.v"
