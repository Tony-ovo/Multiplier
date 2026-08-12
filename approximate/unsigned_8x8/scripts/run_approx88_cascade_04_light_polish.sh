#!/usr/bin/env bash
set -euo pipefail

# Stage 04: light WCE-constrained discrete polish for the cascade model.
# Defaults are intentionally smaller than the old cross62 04 stage.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

BASE_JSON=${1:-final_best_approx88_cascade_inits.json}
OUT_ROOT=${2:-runs88_cascade_04_light_polish}
SEED=${3:-300}
PY=${PYTHON:-python3}
TRAIN_PY=${TRAIN_PY:-train_approx88_cascade.py}
MAX_WCE=${MAX_WCE:-4500}
ESCAPE_ITERS=${ESCAPE_ITERS:-40}
PAIR_MAX_PAIRS=${PAIR_MAX_PAIRS:-50000}
ESCAPE_PAIR_MAX_PAIRS=${ESCAPE_PAIR_MAX_PAIRS:-25000}
CROSS_INIT_MODE=${CROSS_INIT_MODE:-approx62}

mkdir -p "${OUT_ROOT}"
cp "${BASE_JSON}" "${OUT_ROOT}/input_best.json"

echo "[run] BASE_JSON=${BASE_JSON}"
echo "[run] OUT_ROOT=${OUT_ROOT}"
echo "[run] SEED=${SEED} MAX_WCE=${MAX_WCE} ESCAPE_ITERS=${ESCAPE_ITERS}"

${PY} "${TRAIN_PY}" \
  --stage-name 04_light_cascade_polish \
  --init-mode json \
  --base-inits-json "${OUT_ROOT}/input_best.json" \
  --cross-init-mode "${CROSS_INIT_MODE}" \
  --seed "${SEED}" \
  --mred-denom total \
  --max-wce "${MAX_WCE}" \
  --single-lut-names all \
  --candidate-mode top_neutral \
  --topk 160 \
  --neutral-top 240 \
  --neutral-margin 0.006 \
  --do-single \
  --single-rounds 20 \
  --single-mode first \
  --single-random-order \
  --do-pair \
  --pair-rounds 2 \
  --pair-mode first \
  --pair-random-order \
  --pair-lut-names all \
  --pair-max-pairs "${PAIR_MAX_PAIRS}" \
  --do-escape \
  --escape-iters "${ESCAPE_ITERS}" \
  --kmin 1 \
  --kmax 5 \
  --escape-single-rounds 5 \
  --escape-pair-after \
  --escape-pair-rounds 1 \
  --escape-pair-max-pairs "${ESCAPE_PAIR_MAX_PAIRS}" \
  --out-dir "${OUT_ROOT}/04_light_polish" \
  --log-file terminal_log.txt

cp "${OUT_ROOT}/04_light_polish/best_approx88_cascade_inits.json" "${OUT_ROOT}/final_best_approx88_cascade_inits.json"
cp "${OUT_ROOT}/04_light_polish/best_approx88_cascade.v" "${OUT_ROOT}/final_best_approx88_cascade.v"

echo "[done] ${OUT_ROOT}/final_best_approx88_cascade_inits.json"
echo "[done] ${OUT_ROOT}/final_best_approx88_cascade.v"
