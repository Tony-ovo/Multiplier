#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
DESIGN=${DESIGN:-fast}
NUM_RUNS=${NUM_RUNS:-8}
TOP_K=${TOP_K:-2}
PIPE_ROOT=${PIPE_ROOT:-"${ROOT}/pipeline_${DESIGN}_$(date +%Y%m%d_%H%M%S)"}
MLP="$PIPE_ROOT/01_train"; REF="$PIPE_ROOT/02_refine"; mkdir -p "$MLP" "$REF"
DESIGN="$DESIGN" NUM_RUNS="$NUM_RUNS" OUT_ROOT="$MLP" "$ROOT/run_multi.sh"
"$PYTHON" - "$MLP/summary.csv" "$TOP_K" > "$PIPE_ROOT/selected.txt" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],encoding='utf-8')))
for r in rows[:int(sys.argv[2])]: print(r['run']+'\t'+r['best_json'])
PY
rank=0
while IFS=$'\t' read -r run json; do rank=$((rank+1)); "$PYTHON" "$ROOT/refine.py" --design "$DESIGN" --base-inits-json "$json" --out-dir "$REF/rank_${rank}_${run}"; done < "$PIPE_ROOT/selected.txt"
"$PYTHON" - "$REF" "$PIPE_ROOT" <<'PY'
import json,pathlib,shutil,sys
ref=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2]);rows=[]
for p in ref.glob('rank_*/best_signed88_inits.json'):
 o=json.loads(p.read_text());rows.append((o['metrics']['objective_score'],p,o))
rows.sort(key=lambda x:x[0]);
if rows:
 shutil.copy2(rows[0][1],root/'final_best_signed88_inits.json');print('[final]',rows[0][0],rows[0][1])
PY
