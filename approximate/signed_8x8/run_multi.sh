#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
DESIGN=${DESIGN:-fast}
NUM_RUNS=${NUM_RUNS:-8}
JOBS=${JOBS:-1}
GPU_LIST=${GPU_LIST:-0}
SEED_BASE=${SEED_BASE:-0}
SEED_STRIDE=${SEED_STRIDE:-1000}
OUT_ROOT=${OUT_ROOT:-"${ROOT}/runs_${DESIGN}_$(date +%Y%m%d_%H%M%S)"}
CALIBRATION_CSV=${CALIBRATION_CSV:-"${ROOT}/data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"}
EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p "$OUT_ROOT"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
pids=(); fail=0
run_one(){ local i=$1 seed=$2 gpu=$3; local out="$OUT_ROOT/run_$(printf '%02d' "$i")_seed_${seed}"; mkdir -p "$out"; echo "[launch] $i seed=$seed gpu=$gpu"; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/train.py" --design "$DESIGN" --seed "$seed" --out-dir "$out" --calibration-csv "$CALIBRATION_CSV" $EXTRA_ARGS >"$out/pipeline.log" 2>&1; }
for ((i=0;i<NUM_RUNS;i++)); do seed=$((SEED_BASE+i*SEED_STRIDE)); gpu=${GPUS[$((i%${#GPUS[@]}))]}; run_one "$i" "$seed" "$gpu" & pids+=("$!"); if ((${#pids[@]}>=JOBS)); then for pid in "${pids[@]}"; do wait "$pid" || fail=1; done; pids=(); fi; done
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
((fail==0)) || exit 1
"$PYTHON" "$ROOT/summarize.py" "$OUT_ROOT"
echo "[done] $OUT_ROOT"
