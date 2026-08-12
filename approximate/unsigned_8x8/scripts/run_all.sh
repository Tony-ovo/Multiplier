#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage:
  ./run_all.sh SAMPLE:GPU [SAMPLE:GPU ...]
  ./run_all.sh GPU=SAMPLE,SAMPLE [GPU=SAMPLE,SAMPLE ...]

Examples:
  ./run_all.sh 0:0 1:0 2:1 3:1
  ./run_all.sh 0=0,1 1=2,3
  JOBS="0=0,1 1=2,3,4,5" ./run_all.sh

Environment:
  SEED=500
  CROSS_INIT_MODE=approx62
  TRAIN_MAX_WCE=-1
  MAX_WCE=4500
  ESCAPE_ITERS=40
  PYTHON=python3
  NORMALIZE_MODE=move      move or copy final_wce1000_candidate_approx66_inits.json to approx66_inits.json
  SKIP_DONE=1              skip sample dirs that already have final_best_approx88_cascade_inits.json
  ALLOW_DUPLICATE=0        keep 0; duplicate sample dirs would overwrite the same log/output
  STAGGER_SEC=0            sleep N seconds between launches
  DRY_RUN=0                set 1 to print jobs without launching training
EOF
}

SEED=${SEED:-500}
PYTHON=${PYTHON:-python3}
CROSS_INIT_MODE=${CROSS_INIT_MODE:-approx62}
TRAIN_MAX_WCE=${TRAIN_MAX_WCE:--1}
MAX_WCE=${MAX_WCE:-4500}
ESCAPE_ITERS=${ESCAPE_ITERS:-40}
NORMALIZE_MODE=${NORMALIZE_MODE:-move}
SKIP_DONE=${SKIP_DONE:-1}
ALLOW_DUPLICATE=${ALLOW_DUPLICATE:-0}
STAGGER_SEC=${STAGGER_SEC:-0}
DRY_RUN=${DRY_RUN:-0}

SOURCE_NAME=${SOURCE_NAME:-final_wce1000_candidate_approx66_inits.json}
BASE_NAME=${BASE_NAME:-approx66_inits.json}

export PYTHON CROSS_INIT_MODE TRAIN_MAX_WCE MAX_WCE ESCAPE_ITERS

args=("$@")
if [ "${#args[@]}" -eq 0 ]; then
  if [ -n "${JOBS:-}" ]; then
    read -r -a args <<< "${JOBS}"
  else
    usage
    exit 2
  fi
fi

samples=()
gpus=()
declare -A seen_samples=()

add_job() {
  local sample="$1"
  local gpu="$2"

  sample="${sample#/}"
  gpu="${gpu#gpu}"
  gpu="${gpu#GPU}"

  if [ -z "${sample}" ] || [ -z "${gpu}" ]; then
    echo "[error] invalid job sample='${sample}' gpu='${gpu}'" >&2
    exit 2
  fi

  if [ "${ALLOW_DUPLICATE}" != "1" ] && [ -n "${seen_samples[${sample}]+x}" ]; then
    echo "[error] duplicate sample '${sample}' would write to the same directory twice" >&2
    echo "        Fix the command, or set ALLOW_DUPLICATE=1 if you really mean it." >&2
    exit 2
  fi

  seen_samples["${sample}"]=1
  samples+=("${sample}")
  gpus+=("${gpu}")
}

for token in "${args[@]}"; do
  if [[ "${token}" == *=* ]]; then
    gpu="${token%%=*}"
    list="${token#*=}"
    gpu="${gpu#gpu}"
    gpu="${gpu#GPU}"
    list="${list//,/ }"
    for sample in ${list}; do
      add_job "${sample}" "${gpu}"
    done
  elif [[ "${token}" == *:* ]]; then
    sample="${token%%:*}"
    gpu="${token#*:}"
    add_job "${sample}" "${gpu}"
  else
    echo "[error] cannot parse job token: ${token}" >&2
    usage
    exit 2
  fi
done

if [ "${#samples[@]}" -eq 0 ]; then
  echo "[error] no jobs parsed" >&2
  exit 2
fi

if [ "${NORMALIZE_MODE}" != "copy" ] && [ "${NORMALIZE_MODE}" != "move" ]; then
  echo "[error] NORMALIZE_MODE must be copy or move, got: ${NORMALIZE_MODE}" >&2
  exit 2
fi

echo "[run_all] ROOT=${ROOT}"
echo "[run_all] jobs=${#samples[@]} SEED=${SEED}"
echo "[run_all] CROSS_INIT_MODE=${CROSS_INIT_MODE}"
echo "[run_all] TRAIN_MAX_WCE=${TRAIN_MAX_WCE} MAX_WCE=${MAX_WCE} ESCAPE_ITERS=${ESCAPE_ITERS}"
echo "[run_all] NORMALIZE_MODE=${NORMALIZE_MODE} SKIP_DONE=${SKIP_DONE} STAGGER_SEC=${STAGGER_SEC} DRY_RUN=${DRY_RUN}"

cd "${ROOT}"

pids=()
labels=()
launched=0

for i in "${!samples[@]}"; do
  sample="${samples[$i]}"
  gpu="${gpus[$i]}"
  exp_dir="${ROOT}/${sample}"
  src_json="${exp_dir}/${SOURCE_NAME}"
  base_json="${exp_dir}/${BASE_NAME}"
  final_json="${exp_dir}/final_best_approx88_cascade_inits.json"
  log="${exp_dir}/pipeline.log"

  if [ ! -d "${exp_dir}" ]; then
    echo "[error] sample dir not found: ${exp_dir}" >&2
    exit 2
  fi

  if [ ! -f "${base_json}" ]; then
    if [ ! -f "${src_json}" ]; then
      echo "[error] missing base json: ${src_json}" >&2
      exit 2
    fi

    if [ "${DRY_RUN}" = "1" ]; then
      echo "[dry-run] would ${NORMALIZE_MODE} ${SOURCE_NAME} -> ${BASE_NAME}"
    elif [ "${NORMALIZE_MODE}" = "move" ]; then
      mv "${src_json}" "${base_json}"
      echo "[sample ${sample}] renamed ${SOURCE_NAME} -> ${BASE_NAME}"
    else
      cp "${src_json}" "${base_json}"
      echo "[sample ${sample}] copied ${SOURCE_NAME} -> ${BASE_NAME}"
    fi
  else
    echo "[sample ${sample}] using existing ${BASE_NAME}"
  fi

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${final_json}" ]; then
    echo "[sample ${sample}] skip done: ${final_json}"
    continue
  fi

  echo "[launch] sample=${sample} gpu=${gpu} log=${log}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] CUDA_VISIBLE_DEVICES=${gpu} ${ROOT}/run_approx88_cascade_pipeline.sh ${base_json} ${exp_dir} ${SEED}"
    continue
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${ROOT}/run_approx88_cascade_pipeline.sh" "${base_json}" "${exp_dir}" "${SEED}"
  ) > "${log}" 2>&1 &

  pids+=("$!")
  labels+=("${sample}:gpu${gpu}")
  launched=$((launched + 1))

  if [ "${STAGGER_SEC}" != "0" ]; then
    sleep "${STAGGER_SEC}"
  fi
done

if [ "${launched}" -eq 0 ]; then
  echo "[run_all] no jobs launched"
  exit 0
fi

fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  label="${labels[$i]}"
  if wait "${pid}"; then
    echo "[done] ${label}"
  else
    code=$?
    echo "[failed] ${label} exit=${code}" >&2
    fail=1
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "[run_all] some jobs failed" >&2
  exit 1
fi

echo "[run_all done]"
