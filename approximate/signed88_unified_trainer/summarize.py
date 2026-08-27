#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def _artifact_path(summary_path: Path, declared: str, fallback_name: str) -> Path:
    path = Path(declared) if declared else summary_path.parent / fallback_name
    if path.exists():
        return path.resolve()
    fallback = summary_path.parent / fallback_name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f'artifact not found: declared={path}, fallback={fallback}')


def _init_hash(path: Path) -> str:
    obj = json.loads(path.read_text(encoding='utf-8'))
    inits = obj.get('inits', obj)
    raw = json.dumps(inits, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize completed signed88 training runs')
    parser.add_argument('root')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []

    for path in sorted(root.glob('run_*/summary.json')):
        obj = json.loads(path.read_text(encoding='utf-8'))
        metrics = obj['best_metrics']
        initial = obj.get('initial_metrics', {})
        train_args = obj.get('train_args', {})
        initialization = obj.get('initialization', {})
        initial_score = initial.get('objective_score')
        best_score = metrics['objective_score']
        improvement = None if initial_score is None else float(initial_score) - float(best_score)
        improvement_pct = None
        if initial_score not in (None, 0):
            improvement_pct = 100.0 * improvement / abs(float(initial_score))
        best_json = _artifact_path(path, obj.get('best_json', ''), 'best_signed88_inits.json')
        best_rtl = _artifact_path(path, obj.get('best_rtl', ''), 'best_rtl')
        rows.append({
            'run': path.parent.name,
            'design': obj['design'],
            'seed': obj.get('seed'),
            'initMode': train_args.get('init_mode', initialization.get('mode')),
            'logitMean': train_args.get('random_logit_mean'),
            'logitStd': train_args.get('random_logit_std'),
            'mutableLogits': initialization.get('mutable_logit_count'),
            'initialHardOnes': initialization.get('initial_hard_one_count'),
            'initialScore': initial_score,
            'score': best_score,
            'improvement': improvement,
            'improvementPct': improvement_pct,
            'bestStage': obj.get('best_stage'),
            'initSha256': _init_hash(best_json),
            'wNMSE': metrics.get('workload_NMSE'),
            'wMSE': metrics.get('workload_MSE'),
            'wRMSE': metrics.get('workload_RMSE'),
            'wBias': metrics.get('workload_bias'),
            'aCondBiasRMS': metrics.get('workload_conditional_bias_a_rms'),
            'bCondBiasRMS': metrics.get('workload_conditional_bias_b_rms'),
            'gemmNMSE': metrics.get('workload_gemm_NMSE'),
            'dotRMSE': metrics.get('workload_predicted_dot_RMSE'),
            'driftSigma': metrics.get('workload_bias_drift_sigma'),
            'wMRED': metrics['workload_MRED'],
            'wER': metrics['workload_ER'],
            'wMAE': metrics['workload_MED'],
            'uMRED': metrics['MRED'],
            'uER': metrics['ER'],
            'uMAE': metrics['MED'],
            'WCE': metrics['WCE'],
            'bias': metrics['bias'],
            'best_json': str(best_json),
            'best_rtl': str(best_rtl),
        })

    rows.sort(key=lambda row: row['score'])
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / 'summary.json').write_text(
        json.dumps({'rows': rows, 'best': rows[0] if rows else None}, indent=2),
        encoding='utf-8',
    )
    if rows:
        best = rows[0]
        shutil.copy2(best['best_json'], root / 'overall_best_signed88_inits.json')
        print(
            f"[best] {best['run']} score={best['score']:.12g} "
            f"initial={best['initialScore']:.12g} improvement={best['improvementPct']:.3f}% "
            f"wNMSE={best['wNMSE']:.12g} wBias={best['wBias']:+.6g}"
        )
    print(root / 'summary.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
