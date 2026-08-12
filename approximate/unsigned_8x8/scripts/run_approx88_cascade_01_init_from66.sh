#!/usr/bin/env bash
set -euo pipefail

# Stage 01: build an approx88 cascade JSON from an existing 6x6 best JSON.
# LL inherits the trained 6x6 INITs. HL/LH use CROSS_INIT_MODE. Top comp88 starts exact.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

BASE_JSON=${1:-best_approx66_inits.json}
OUT_ROOT=${2:-runs88_cascade_01_init}
SEED=${3:-0}
PY=${PYTHON:-python3}
TRAIN_PY=${TRAIN_PY:-train_approx88_cascade.py}
CROSS_INIT_MODE=${CROSS_INIT_MODE:-approx62}

mkdir -p "${OUT_ROOT}"
cp "${BASE_JSON}" "${OUT_ROOT}/input_low66_best.json"

echo "[run] BASE_JSON=${BASE_JSON}"
echo "[run] OUT_ROOT=${OUT_ROOT}"
echo "[run] SEED=${SEED}"
echo "[run] CROSS_INIT_MODE=${CROSS_INIT_MODE}"

${PY} "${TRAIN_PY}" \
  --stage-name "01_init_from_6x6_best_${CROSS_INIT_MODE}_cascade" \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/input_low66_best.json" \
  --cross-init-mode "${CROSS_INIT_MODE}" \
  --seed "${SEED}" \
  --mred-denom total \
  --eval-only \
  --out-dir "${OUT_ROOT}/01_init" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/01_init/best_approx88_cascade_inits.json" "${OUT_ROOT}/final_best_approx88_cascade_inits.json"
cp "${OUT_ROOT}/01_init/best_approx88_cascade.v" "${OUT_ROOT}/final_best_approx88_cascade.v"

echo "[done] ${OUT_ROOT}/final_best_approx88_cascade_inits.json"
echo "[done] ${OUT_ROOT}/final_best_approx88_cascade.v"
