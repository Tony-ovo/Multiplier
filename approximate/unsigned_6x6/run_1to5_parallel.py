#!/usr/bin/env python3
"""Run approx66 steps 1..5 for multiple random restarts.

Each restart is independent:
  step1 random shared search -> step2 fullcomp -> step3 paircomp
  -> step4 escape-local -> step5 conservative WCE search.

Restarts run concurrently up to --jobs. Within one restart, steps are
executed sequentially because each step consumes the previous best INIT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
STEP1 = ROOT / "1"
STEP2 = ROOT / "2"
STEP3 = ROOT / "3"
STEP4 = ROOT / "4"
STEP5 = ROOT / "5"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("metrics", {})


def metric_key(path: Path) -> tuple:
    m = read_metrics(path)
    return (
        float(m.get("MRED", 1e99)),
        int(m.get("WCE", 10**9)),
        float(m.get("MED", 1e99)),
        float(m.get("ER", 1e99)),
    )


def append_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"[cwd] {cwd}\n")
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.write("=" * 100 + "\n")
        log.flush()
        if dry_run:
            return
        subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def final_json_for_step(step_dir: Path, strict: bool = False) -> Path:
    if strict:
        return step_dir / "final_wce930_best_approx66_inits.json"
    return step_dir / "final_best_approx66_inits.json"


def run_step1(args: argparse.Namespace, run_dir: Path, seed: int, env: Dict[str, str], log_path: Path) -> Path:
    out = run_dir / "01_step1_shared_random"
    final_json = final_json_for_step(out)
    if args.resume and final_json.exists():
        return final_json

    explore_dir = out / "01_explore"
    refine_dir = out / "02_refine"

    init_mode = "random" if args.random_step1 else "manual"
    explore_cmd = [
        args.python,
        "train_approx66_ste_search.py",
        "--stage-name",
        "explore",
        "--epochs",
        str(args.step1_explore_epochs),
        "--lr",
        "0.01",
        "--seed",
        str(seed),
        "--init-mode",
        init_mode,
        "--random-init-prob",
        str(args.random_init_prob),
        "--init-p",
        "0.70",
        "--noise-std",
        "0.25",
        "--c-init",
        "2.0",
        "--c-out",
        "2.0",
        "--c-anneal",
        "--zero-weight",
        "0.01",
        "--med-weight",
        "0.0",
        "--bin-weight",
        "0.0001",
        "--grad-clip",
        "5.0",
        "--restart-from-best-every",
        "0",
        "--out-dir",
        str(explore_dir),
        "--log-file",
        "terminal_log.txt",
    ]
    run_cmd(explore_cmd, cwd=STEP1, env=env, log_path=log_path, dry_run=args.dry_run)

    refine_cmd = [
        args.python,
        "train_approx66_ste_search.py",
        "--stage-name",
        "refine",
        "--epochs",
        str(args.step1_refine_epochs),
        "--lr",
        "0.002",
        "--seed",
        str(seed + 1),
        "--init-mode",
        "json",
        "--base-inits-json",
        str(explore_dir / "best_approx66_inits.json"),
        "--init-p",
        "0.85",
        "--noise-std",
        "0.05",
        "--c-init",
        "2.0",
        "--c-out",
        "2.0",
        "--zero-weight",
        "0.01",
        "--med-weight",
        "0.0",
        "--bin-weight",
        "0.00005",
        "--grad-clip",
        "1.0",
        "--restart-from-best-every",
        "120",
        "--restart-init-p",
        "0.88",
        "--restart-noise-std",
        "0.03",
        "--restart-lr-decay",
        "0.75",
        "--min-lr",
        "0.0001",
        "--bitflip-after",
        "--bitflip-rounds",
        "3",
        "--out-dir",
        str(refine_dir),
        "--log-file",
        "terminal_log.txt",
    ]
    run_cmd(refine_cmd, cwd=STEP1, env=env, log_path=log_path, dry_run=args.dry_run)

    if not args.dry_run:
        copy_if_exists(refine_dir / "best_approx66_inits.json", final_json)
        copy_if_exists(refine_dir / "best_approx66_verilog_snippet.v", out / "final_best_approx66_verilog_snippet.v")
        if not final_json.exists():
            raise FileNotFoundError(f"step1 final JSON not found: {final_json}")
    return final_json


def run_shell_step(
    args: argparse.Namespace,
    *,
    step_name: str,
    script_dir: Path,
    script_name: str,
    base_json: Path,
    out_dir: Path,
    env: Dict[str, str],
    log_path: Path,
    strict_final: bool = False,
) -> Path:
    final_json = final_json_for_step(out_dir, strict=strict_final)
    if args.resume and final_json.exists():
        return final_json
    cmd = ["bash", str(script_dir / script_name), str(base_json), str(out_dir)]
    run_cmd(cmd, cwd=script_dir, env=env, log_path=log_path, dry_run=args.dry_run)
    if not args.dry_run and not final_json.exists():
        raise FileNotFoundError(f"{step_name} final JSON not found: {final_json}")
    return final_json


def run_one_restart(args: argparse.Namespace, restart_idx: int) -> Dict[str, Any]:
    seed = args.seed_base + restart_idx * args.seed_stride
    run_name = f"restart_{restart_idx:02d}_seed_{seed}"
    run_dir = args.out_root / run_name
    log_path = run_dir / "pipeline.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHON"] = args.python
    env["PYTHON_BIN"] = args.python
    env["PY"] = args.python
    if args.cuda_devices:
        devices = [x.strip() for x in args.cuda_devices.split(",") if x.strip()]
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[restart_idx % len(devices)]

    status: Dict[str, Any] = {
        "restart": restart_idx,
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "running",
        "steps": {},
    }
    append_json(run_dir / "status.json", status)

    try:
        print(f"[start] {run_name}")
        j1 = run_step1(args, run_dir, seed, env, log_path)
        status["steps"]["1"] = str(j1)
        append_json(run_dir / "status.json", status)

        j2 = run_shell_step(
            args,
            step_name="step2_fullcomp",
            script_dir=STEP2,
            script_name="run_approx66_unshared_fullcomp.sh",
            base_json=j1,
            out_dir=run_dir / "02_step2_unshared_fullcomp",
            env=env,
            log_path=log_path,
        )
        status["steps"]["2"] = str(j2)
        append_json(run_dir / "status.json", status)

        j3 = run_shell_step(
            args,
            step_name="step3_paircomp",
            script_dir=STEP3,
            script_name="run_approx66_unshared_paircomp_next.sh",
            base_json=j2,
            out_dir=run_dir / "03_step3_paircomp",
            env=env,
            log_path=log_path,
        )
        status["steps"]["3"] = str(j3)
        append_json(run_dir / "status.json", status)

        j4 = run_shell_step(
            args,
            step_name="step4_escape",
            script_dir=STEP4,
            script_name="run_approx66_escape_local_next.sh",
            base_json=j3,
            out_dir=run_dir / "04_step4_escape_local",
            env=env,
            log_path=log_path,
        )
        status["steps"]["4"] = str(j4)
        append_json(run_dir / "status.json", status)

        j5 = run_shell_step(
            args,
            step_name="step5_wce930",
            script_dir=STEP5,
            script_name="run_approx66_conservative_wce_next.sh",
            base_json=j4,
            out_dir=run_dir / "05_step5_conservative_wce",
            env=env,
            log_path=log_path,
            strict_final=True,
        )
        status["steps"]["5_strict_wce930"] = str(j5)

        step5_dir = run_dir / "05_step5_conservative_wce"
        strict_v = step5_dir / "final_wce930_best_approx66_unshared_paircomp.v"
        relaxed_json = step5_dir / "final_wce1000_candidate_approx66_inits.json"
        relaxed_v = step5_dir / "final_wce1000_candidate_approx66_unshared_paircomp.v"

        copy_if_exists(j5, run_dir / "final_wce930_best_approx66_inits.json")
        copy_if_exists(strict_v, run_dir / "final_wce930_best_approx66_unshared_paircomp.v")
        copy_if_exists(relaxed_json, run_dir / "final_wce1000_candidate_approx66_inits.json")
        copy_if_exists(relaxed_v, run_dir / "final_wce1000_candidate_approx66_unshared_paircomp.v")

        if not args.dry_run:
            status["metrics_wce930"] = read_metrics(j5)
            if relaxed_json.exists():
                status["metrics_wce1000"] = read_metrics(relaxed_json)
        status["status"] = "ok"
        print(f"[done]  {run_name}")
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
        print(f"[fail]  {run_name}: {exc}")
        raise
    finally:
        append_json(run_dir / "status.json", status)
    return status


def collect_overall(args: argparse.Namespace, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_results = [r for r in results if r.get("status") == "ok"]
    summary: Dict[str, Any] = {
        "out_root": str(args.out_root),
        "num_runs": args.num_runs,
        "jobs": args.jobs,
        "seed_base": args.seed_base,
        "seed_stride": args.seed_stride,
        "random_init_prob": args.random_init_prob,
        "results": results,
    }
    strict_candidates = []
    relaxed_candidates = []
    for r in ok_results:
        rd = Path(r["run_dir"])
        strict = rd / "final_wce930_best_approx66_inits.json"
        relaxed = rd / "final_wce1000_candidate_approx66_inits.json"
        if strict.exists():
            strict_candidates.append(strict)
        if relaxed.exists():
            relaxed_candidates.append(relaxed)

    if strict_candidates:
        best = min(strict_candidates, key=metric_key)
        summary["overall_best_wce930_json"] = str(best)
        summary["overall_best_wce930_metrics"] = read_metrics(best)
        copy_if_exists(best, args.out_root / "overall_best_wce930_approx66_inits.json")
        copy_if_exists(
            best.with_name("final_wce930_best_approx66_unshared_paircomp.v"),
            args.out_root / "overall_best_wce930_approx66_unshared_paircomp.v",
        )

    if relaxed_candidates:
        best = min(relaxed_candidates, key=metric_key)
        summary["overall_best_wce1000_json"] = str(best)
        summary["overall_best_wce1000_metrics"] = read_metrics(best)
        copy_if_exists(best, args.out_root / "overall_best_wce1000_approx66_inits.json")
        copy_if_exists(
            best.with_name("final_wce1000_candidate_approx66_unshared_paircomp.v"),
            args.out_root / "overall_best_wce1000_approx66_unshared_paircomp.v",
        )

    append_json(args.out_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run approx66_mlp62 steps 1..5 with parallel random restarts.")
    ap.add_argument("--out-root", type=Path, default=ROOT / f"runs_1to5_parallel_{now_tag()}")
    ap.add_argument("--num-runs", type=int, default=1, help="Number of independent random INIT restarts.")
    ap.add_argument("--jobs", type=int, default=1, help="Maximum restarts to run concurrently.")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--seed-stride", type=int, default=1000)
    ap.add_argument("--random-init-prob", type=float, default=0.5)
    ap.add_argument("--manual-step1", dest="random_step1", action="store_false", help="Use manual INIT in step1 instead of random INIT.")
    ap.set_defaults(random_step1=True)
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable or "python3"))
    ap.add_argument("--cuda-devices", default="", help="Optional comma-separated CUDA_VISIBLE_DEVICES assignment per restart, e.g. 0,1.")
    ap.add_argument("--resume", action="store_true", help="Skip a step when its final output already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Write commands to logs without executing them.")
    ap.add_argument("--step1-explore-epochs", type=int, default=4000)
    ap.add_argument("--step1-refine-epochs", type=int, default=1000)
    args = ap.parse_args()
    if args.num_runs < 1:
        ap.error("--num-runs must be >= 1")
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    args.out_root = args.out_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"[root] {args.out_root}")
    print(f"[runs] num_runs={args.num_runs} jobs={args.jobs} seed_base={args.seed_base} seed_stride={args.seed_stride}")

    results: List[Dict[str, Any]] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one_restart, args, i) for i in range(args.num_runs)]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:
                failures += 1
                results.append({"status": "failed", "error": repr(exc)})

    summary = collect_overall(args, results)
    print(f"[summary] {args.out_root / 'summary.json'}")
    if "overall_best_wce930_json" in summary:
        m = summary["overall_best_wce930_metrics"]
        print(
            "[best-wce930] "
            f"MRED={m.get('MRED')} MED={m.get('MED')} ER={m.get('ER')} WCE={m.get('WCE')}"
        )
        print(f"[best-wce930] {args.out_root / 'overall_best_wce930_approx66_inits.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
