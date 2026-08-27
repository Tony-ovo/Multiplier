#!/usr/bin/env bash
# Multi-seed experiment for fully independent random initialization of every
# trainable Default-LUT logit.  Additional train.py options are accepted as an
# argv array (for example: ./run_default_random_logits.sh --stage1-epochs 8000).
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
NUM_RUNS=${NUM_RUNS:-16}
JOBS=${JOBS:-1}
GPU_LIST=${GPU_LIST:-0}
SEED_BASE=${SEED_BASE:-0}
SEED_STRIDE=${SEED_STRIDE:-1}
K_EFF=${K_EFF:-4096}
RANDOM_LOGIT_MEAN=${RANDOM_LOGIT_MEAN:-0.0}
RANDOM_LOGIT_STD=${RANDOM_LOGIT_STD:-1.0}
POPULATION_SIZE=${POPULATION_SIZE:-0}
CALIBRATION_CSV=${CALIBRATION_CSV:-"${ROOT}/data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"}
EXISTING_POLICY=${EXISTING_POLICY:-error} # error | resume | replace

timestamp=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-"${ROOT}/runs_random_logits/default_k${K_EFF}_${timestamp}_pid$$"}

die() {
    echo "[error] $*" >&2
    exit 2
}

is_nonnegative_integer() {
    [[ $1 =~ ^[0-9]+$ ]]
}

is_positive_integer() {
    [[ $1 =~ ^[1-9][0-9]*$ ]]
}

is_positive_integer "$NUM_RUNS" || die "NUM_RUNS must be a positive integer"
is_positive_integer "$JOBS" || die "JOBS must be a positive integer"
is_nonnegative_integer "$SEED_BASE" || die "SEED_BASE must be a non-negative integer"
is_positive_integer "$SEED_STRIDE" || die "SEED_STRIDE must be a positive integer"
is_nonnegative_integer "$POPULATION_SIZE" || die "POPULATION_SIZE must be a non-negative integer"
[[ $EXISTING_POLICY == error || $EXISTING_POLICY == resume || $EXISTING_POLICY == replace ]] || \
    die "EXISTING_POLICY must be error, resume, or replace"
[[ -r $CALIBRATION_CSV ]] || die "calibration CSV is not readable: $CALIBRATION_CSV"
command -v "$PYTHON" >/dev/null 2>&1 || die "Python executable not found: $PYTHON"

OUT_ROOT=$("$PYTHON" - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)
[[ $OUT_ROOT != / && $OUT_ROOT != "$ROOT" ]] || \
    die "OUT_ROOT must be a dedicated experiment directory, not '$OUT_ROOT'"

"$PYTHON" - "$K_EFF" "$RANDOM_LOGIT_MEAN" "$RANDOM_LOGIT_STD" <<'PY' || exit 2
import math
import sys

k, mean, std = map(float, sys.argv[1:])
if not math.isfinite(k) or k < 1:
    raise SystemExit("[error] K_EFF must be finite and >= 1")
if not math.isfinite(mean):
    raise SystemExit("[error] RANDOM_LOGIT_MEAN must be finite")
if not math.isfinite(std) or std <= 0:
    raise SystemExit("[error] RANDOM_LOGIT_STD must be finite and > 0")
PY

if (( JOBS > NUM_RUNS )); then
    JOBS=$NUM_RUNS
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
((${#GPUS[@]} > 0)) || die "GPU_LIST must contain at least one GPU index or cpu"
for gpu in "${GPUS[@]}"; do
    [[ $gpu == cpu || $gpu =~ ^[0-9]+$ ]] || \
        die "invalid GPU_LIST entry '$gpu' (use e.g. 0,1 or cpu)"
done

# The script owns these arguments so every run in one experiment is comparable.
# All remaining train.py options retain exact argv boundaries; no eval or unsafe
# EXTRA_ARGS word splitting is used.
if [[ ${1:-} == -- ]]; then
    shift
fi
TRAIN_EXTRA_ARGS=("$@")
for arg in "${TRAIN_EXTRA_ARGS[@]}"; do
    case "$arg" in
        --design|--design=*|--init-mode|--init-mode=*|--random-logit-mean|--random-logit-mean=*|\
        --random-logit-std|--random-logit-std=*|--seed|--seed=*|--out-dir|--out-dir=*|\
        --device|--device=*|--calibration-csv|--calibration-csv=*|--bias-effective-k|--bias-effective-k=*|\
        --population-size|--population-size=*)
            die "argument '$arg' is managed by run_default_random_logits.sh; use its environment variable instead"
            ;;
    esac
done

archive_path() {
    local path=$1
    local archived="${path}.archived_${timestamp}_pid$$"
    local suffix=0
    while [[ -e $archived ]]; do
        suffix=$((suffix + 1))
        archived="${path}.archived_${timestamp}_pid$$_${suffix}"
    done
    mv -- "$path" "$archived"
    echo "[archive] $path -> $archived"
}

# A manifest prevents an explicit OUT_ROOT from accidentally mixing runs that
# used different logit distributions, seeds, calibration data, or train flags.
if [[ -d $OUT_ROOT && -n $(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit) && \
      ! -f $OUT_ROOT/experiment_config.json ]]; then
    die "non-empty OUT_ROOT was not created by this script (missing experiment_config.json): $OUT_ROOT"
fi
mkdir -p "$OUT_ROOT"

candidate_manifest="$OUT_ROOT/.experiment_config.candidate.$$"
"$PYTHON" - \
    "$candidate_manifest" "$ROOT" "$CALIBRATION_CSV" "$SEED_BASE" "$SEED_STRIDE" \
    "$K_EFF" "$RANDOM_LOGIT_MEAN" "$RANDOM_LOGIT_STD" "$POPULATION_SIZE" "${TRAIN_EXTRA_ARGS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    output, root, calibration, seed_base, seed_stride, k_eff,
    random_mean, random_std, population_size, *extra_args
) = sys.argv[1:]
trainer = (Path(root) / "train.py").resolve()
config = {
    "format": "signed88-default-random-logits-multiseed-v2",
    "trainer": str(trainer),
    "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
    "design": "default",
    "init_mode": "random_logits",
    "calibration_csv": str(Path(calibration).resolve()),
    "seed_base": int(seed_base),
    "seed_stride": int(seed_stride),
    "bias_effective_k": float(k_eff),
    "random_logit_mean": float(random_mean),
    "random_logit_std": float(random_std),
    "population_size": int(population_size),
    "extra_train_args": extra_args,
}
Path(output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

manifest="$OUT_ROOT/experiment_config.json"
if [[ -f $manifest ]] && ! cmp -s "$manifest" "$candidate_manifest"; then
    if [[ $EXISTING_POLICY == replace ]]; then
        archive_path "$OUT_ROOT"
        mkdir -p "$OUT_ROOT"
        manifest="$OUT_ROOT/experiment_config.json"
        candidate_manifest="$OUT_ROOT/.experiment_config.candidate.$$"
        "$PYTHON" - \
            "$candidate_manifest" "$ROOT" "$CALIBRATION_CSV" "$SEED_BASE" "$SEED_STRIDE" \
            "$K_EFF" "$RANDOM_LOGIT_MEAN" "$RANDOM_LOGIT_STD" "$POPULATION_SIZE" "${TRAIN_EXTRA_ARGS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    output, root, calibration, seed_base, seed_stride, k_eff,
    random_mean, random_std, population_size, *extra_args
) = sys.argv[1:]
trainer = (Path(root) / "train.py").resolve()
config = {
    "format": "signed88-default-random-logits-multiseed-v2",
    "trainer": str(trainer),
    "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
    "design": "default",
    "init_mode": "random_logits",
    "calibration_csv": str(Path(calibration).resolve()),
    "seed_base": int(seed_base),
    "seed_stride": int(seed_stride),
    "bias_effective_k": float(k_eff),
    "random_logit_mean": float(random_mean),
    "random_logit_std": float(random_std),
    "population_size": int(population_size),
    "extra_train_args": extra_args,
}
Path(output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    else
        rm -f -- "$candidate_manifest"
        die "OUT_ROOT configuration differs from this invocation; use a new OUT_ROOT or EXISTING_POLICY=replace"
    fi
fi
mv -f -- "$candidate_manifest" "$manifest"

"$PYTHON" - "$OUT_ROOT/planned_runs.json" "$NUM_RUNS" "$SEED_BASE" "$SEED_STRIDE" <<'PY'
import json
import sys
from pathlib import Path

path, count, base, stride = sys.argv[1:]
count, base, stride = int(count), int(base), int(stride)
runs = [{"index": i, "seed": base + i * stride} for i in range(count)]
Path(path).write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")
PY

run_is_complete() {
    local out=$1
    local seed=$2
    [[ -f $out/summary.json && -f $out/best_signed88_inits.json && \
       -f $out/best_rtl/trained_artifact.json ]] || return 1
    "$PYTHON" - "$out/summary.json" "$seed" "$K_EFF" "$RANDOM_LOGIT_MEAN" "$RANDOM_LOGIT_STD" "$POPULATION_SIZE" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seed = int(sys.argv[2])
k_eff, mean, std = map(float, sys.argv[3:6])
population_size = int(sys.argv[6])
args = summary.get("train_args", {})
ok = (
    summary.get("design") == "default"
    and summary.get("seed") == seed
    and args.get("init_mode") == "random_logits"
    and math.isclose(float(args.get("bias_effective_k", float("nan"))), k_eff)
    and math.isclose(float(args.get("random_logit_mean", float("nan"))), mean)
    and math.isclose(float(args.get("random_logit_std", float("nan"))), std)
    and args.get("population_size") == population_size
)
raise SystemExit(0 if ok else 1)
PY
}

prepare_run_dir() {
    local out=$1
    local seed=$2
    if [[ ! -e $out ]]; then
        mkdir -p "$out"
        return 0
    fi
    if [[ $EXISTING_POLICY == resume ]] && run_is_complete "$out" "$seed"; then
        echo "[resume] skip complete seed=$seed out=$out"
        return 1
    fi
    if [[ $EXISTING_POLICY == error ]]; then
        die "run directory already exists: $out (use EXISTING_POLICY=resume or replace)"
    fi
    # train.py has no optimizer-state resume.  At batch level, resume means
    # retaining completed runs and safely restarting only incomplete runs.
    archive_path "$out"
    mkdir -p "$out"
    return 0
}

run_one() {
    local index=$1
    local seed=$2
    local gpu=$3
    local out="$OUT_ROOT/run_$(printf '%03d' "$index")_seed_${seed}"
    if ! prepare_run_dir "$out" "$seed"; then
        return 0
    fi

    local -a cmd=(
        "$PYTHON" "$ROOT/train.py"
        --design default
        --init-mode random_logits
        --random-logit-mean "$RANDOM_LOGIT_MEAN"
        --random-logit-std "$RANDOM_LOGIT_STD"
        --bias-effective-k "$K_EFF"
        --seed "$seed"
        --out-dir "$out"
        --calibration-csv "$CALIBRATION_CSV"
        --population-size "$POPULATION_SIZE"
        "${TRAIN_EXTRA_ARGS[@]}"
    )

    echo "[launch] index=$index seed=$seed gpu=$gpu out=$out"
    local status
    if [[ $gpu == cpu ]]; then
        if "${cmd[@]}" --device cpu >"$out/pipeline.log" 2>&1; then
            echo "[pass] seed=$seed"
            return 0
        else
            status=$?
        fi
    else
        if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" --device cuda >"$out/pipeline.log" 2>&1; then
            echo "[pass] seed=$seed gpu=$gpu"
            return 0
        else
            status=$?
        fi
    fi
    printf 'seed=%s status=%s\n' "$seed" "$status" >"$out/FAILED"
    echo "[fail] seed=$seed gpu=$gpu status=$status log=$out/pipeline.log" >&2
    return "$status"
}

worker() {
    local worker_id=$1
    local gpu=${GPUS[$((worker_id % ${#GPUS[@]}))]}
    local failed=0
    local index seed
    for ((index=worker_id; index<NUM_RUNS; index+=JOBS)); do
        seed=$((SEED_BASE + index * SEED_STRIDE))
        run_one "$index" "$seed" "$gpu" || failed=1
    done
    return "$failed"
}

echo "[experiment] default random_logits runs=$NUM_RUNS jobs=$JOBS GPUs=$GPU_LIST"
echo "[initialization] Normal(mean=$RANDOM_LOGIT_MEAN, std=$RANDOM_LOGIT_STD), K_eff=$K_EFF"
echo "[search] population_size=$POPULATION_SIZE"
echo "[output] $OUT_ROOT"

pids=()
for ((worker_id=0; worker_id<JOBS; worker_id++)); do
    worker "$worker_id" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if ((failed != 0)); then
    echo "[error] one or more runs failed; summary was not generated" >&2
    echo "[hint] inspect run_*/pipeline.log, then rerun with the same OUT_ROOT and EXISTING_POLICY=resume" >&2
    exit 1
fi

"$PYTHON" "$ROOT/summarize.py" "$OUT_ROOT"
echo "[done] $OUT_ROOT"
