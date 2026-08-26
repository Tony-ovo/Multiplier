#!/usr/bin/env bash
# Control arm for the default-topology experiment: binary-logit unified
# trainer (same config as wce_batch1) followed by the same refine.py polish
# used for the quad arm, so both pipelines are directly comparable.
# Usage: ./run_control_default.sh <gpu> <name> [extra train.py args...]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIFIED="$ROOT/../signed88_unified_trainer"
PYTHON=${PYTHON:-python3}
GPU=$1; NAME=$2; shift 2
OUT="$ROOT/runs_default_control/$NAME"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$UNIFIED/train.py" \
  --design default --out-dir "$OUT" \
  --wce-weight 0.05 --wce-beta 0.25 \
  --score-wce-weight 1.0 --bias-effective-k 4096 \
  "$@" >"$OUT/pipeline.log" 2>&1

"$PYTHON" "$UNIFIED/refine.py" \
  --base-inits-json "$OUT/best_signed88_inits.json" \
  --out-dir "$OUT/refined" \
  --bit-rounds 40 --pair-rounds 4 \
  --pair-candidate-bits 56 --pair-max-pairs 1540 \
  --basin-iters 20 --seed 7 >>"$OUT/pipeline.log" 2>&1

echo "[done] $NAME"
tail -2 "$OUT/pipeline.log"
