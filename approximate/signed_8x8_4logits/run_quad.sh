#!/usr/bin/env bash
# Launch one quad4 training run and post-process it:
#   train_quad.py -> refine.py (exact discrete polish, same objective)
# Usage: ./run_quad.sh <gpu> <name> [extra train_quad.py args...]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIFIED="$ROOT/../signed88_unified_trainer"
PYTHON=${PYTHON:-python3}
GPU=$1; NAME=$2; shift 2
OUT_BASE=${OUT_BASE:-$ROOT/runs_quad}
OUT="$OUT_BASE/$NAME"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/train_quad.py" \
  --out-dir "$OUT" "$@" >"$OUT/pipeline.log" 2>&1

PAIR_BITS=${PAIR_BITS:-112}
PAIR_MAX=${PAIR_MAX:-6216}
"$PYTHON" "$UNIFIED/refine.py" \
  --base-inits-json "$OUT/best_signed88_inits.json" \
  --out-dir "$OUT/refined" \
  --bit-rounds 40 --pair-rounds 4 \
  --pair-candidate-bits "$PAIR_BITS" --pair-max-pairs "$PAIR_MAX" \
  --basin-iters 20 --seed 7 >>"$OUT/pipeline.log" 2>&1

echo "[done] $NAME"
tail -2 "$OUT/pipeline.log"
