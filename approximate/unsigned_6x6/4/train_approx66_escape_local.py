#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escape-local optimizer for approx66_unshared_paircomp.

Use this AFTER train_approx66_unshared_paircomp.py has reached a single-bit local optimum.
It imports train_approx66_unshared_paircomp.py and adds:
  1) top relative-error case reporting
  2) neutral-bit scan: find bit flips that are least harmful alone
  3) random k-bit perturbation on neutral bits + local single-bit polish
  4) optional quick pair polish

Terminal output is mirrored exactly to terminal_log.txt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Must be in the same directory as this script, or on PYTHONPATH.
import train_approx66_unshared_paircomp as core


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def install_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = sys.stdout
    return f, old_stdout, old_stderr


def load_inits(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return core.normalize_inits(obj)


def metric_str(m: core.Metrics) -> str:
    return f"MRED={m.MRED:.10f} MED={m.MED:.4f} ER={m.ER:.4f} WCE={m.WCE} err={m.error_cases}/{m.total_cases}"


def write_best(out_dir: Path, stage: str, inits: Dict[str, str], m: core.Metrics):
    best = {
        "stage": stage,
        "epoch": "escape_local",
        "loss": None,
        "metrics": core.metrics_to_dict(m),
        "inits": core.normalize_inits(inits),
    }
    core.write_best(out_dir, best, "best")
    # stable final aliases
    (out_dir / "final_best_approx66_inits.json").write_text((out_dir / "best_approx66_inits.json").read_text(encoding="utf-8"), encoding="utf-8")
    if (out_dir / "best_approx66_unshared_paircomp.v").exists():
        (out_dir / "final_best_approx66_unshared_paircomp.v").write_text((out_dir / "best_approx66_unshared_paircomp.v").read_text(encoding="utf-8"), encoding="utf-8")


def approx_values(ints: Dict[str, int]) -> np.ndarray:
    """Return approx product for all 4096 cases using core.EVAL internals."""
    E = core.EVAL
    tabs = E.tables(ints)
    plow = E.approx62("low", E.lowb, tabs)
    pmid = E.approx62("mid", E.midb, tabs)
    phigh = E.approx62("high", E.highb, tabs)
    prod = [None] * 12
    prod[0], prod[1], prod[10], prod[11] = plow[0], plow[1], phigh[6], phigh[7]
    prod[2], prod[3] = E.lut62(tabs["u_comp23"], plow[2], pmid[0], plow[3], pmid[1], E.z, E.o)
    prod[4] = E.lut6(tabs["u_comp4"], plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
    prod[5] = E.lut6(tabs["u_comp5"], plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
    prod[6] = E.lut6(tabs["u_comp6"], plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
    prod[7] = E.lut6(tabs["u_comp7"], plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
    prod[8], prod[9] = E.lut62(tabs["u_comp89"], pmid[6], phigh[4], pmid[7], phigh[5], E.z, E.o)
    val = np.zeros_like(E.exact)
    for i, p in enumerate(prod):
        val += p.astype(np.int64) << i
    return val


def print_top_cases(inits: Dict[str, str], topn: int):
    ints = core.inits_to_ints(inits)
    val = approx_values(ints)
    E = core.EVAL
    err = np.abs(val - E.exact)
    rel = np.zeros_like(E.exact, dtype=np.float64)
    rel[E.mask] = err[E.mask] / E.exact[E.mask]
    idx = np.argsort(-rel)[:topn]
    print(f"\n[topcases] top {topn} relative-error cases")
    print("rank,a,b,exact,approx,abs_err,rel_err")
    for r, i in enumerate(idx, 1):
        print(f"{r},{int(E.a[i])},{int(E.b[i])},{int(E.exact[i])},{int(val[i])},{int(err[i])},{float(rel[i]):.10f}")


def parse_names(s: str) -> List[str]:
    return core.parse_lut_names(s)


def bits_for_names(names: List[str]) -> List[Tuple[str, int]]:
    return core.candidate_bits(names, random_order=False)


def flip_bits(ints: Dict[str, int], flips: List[Tuple[str, int]]) -> Dict[str, int]:
    out = dict(ints)
    for n, b in flips:
        out[n] ^= (1 << b)
    return out


def neutral_scan(inits: Dict[str, str], names: List[str], topn: int, allow_positive: bool = True) -> List[Tuple[str, int, float]]:
    """Return bits sorted by single-flip delta MRED. Negative delta improves, positive worsens."""
    cur = core.inits_to_ints(inits)
    base_m = core.eval_ints(cur)
    cand = bits_for_names(names)
    scored: List[Tuple[str, int, float]] = []
    print(f"\n[neutral] scan start {metric_str(base_m)} candidates={len(cand)} names={','.join(names)}")
    best_improve = None
    for n, b in cand:
        trial = dict(cur)
        trial[n] ^= (1 << b)
        m = core.eval_ints(trial)
        delta = m.MRED - base_m.MRED
        if allow_positive or delta <= 0:
            scored.append((n, b, delta))
        if best_improve is None or delta < best_improve[2]:
            best_improve = (n, b, delta, m)
    scored.sort(key=lambda x: x[2])
    print(f"[neutral] best single candidate: lut={best_improve[0]} bit={best_improve[1]:02d} delta={best_improve[2]:+.10f} -> {metric_str(best_improve[3])}")
    print(f"[neutral] top {min(topn, len(scored))} least-harmful/improving bits:")
    for n, b, d in scored[:min(topn, len(scored), 30)]:
        print(f"  {n:10s} bit={b:02d} delta={d:+.10f}")
    return scored[:topn]


def weighted_sample(pool: List[Tuple[str, int, float]], k: int) -> List[Tuple[str, int]]:
    """Sample k flips, biased toward lower single-flip delta."""
    if k >= len(pool):
        return [(n, b) for n, b, _ in pool]
    deltas = np.array([d for _, _, d in pool], dtype=np.float64)
    # Convert deltas to ranks; this is numerically safer than exp(-delta/temp).
    ranks = np.arange(len(pool), dtype=np.float64)
    weights = 1.0 / np.power(ranks + 1.0, 0.75)
    weights = weights / weights.sum()
    idx = np.random.choice(len(pool), size=k, replace=False, p=weights)
    return [(pool[i][0], pool[i][1]) for i in idx]


def local_polish(inits: Dict[str, str], names: List[str], single_rounds: int, single_mode: str, pair_rounds: int, pair_names: List[str], pair_max_pairs: int) -> Tuple[Dict[str, str], core.Metrics]:
    loc, lm = core.greedy_single(inits, single_rounds, single_mode, names, random_order=True)
    if pair_rounds > 0:
        loc2, lm2 = core.greedy_pair(loc, pair_rounds, "first", pair_names, True, pair_max_pairs)
        if lm2.MRED < lm.MRED:
            return loc2, lm2
    return loc, lm


def escape_search(
    start: Dict[str, str],
    out_dir: Path,
    stage: str,
    iters: int,
    pool_names: List[str],
    neutral_top: int,
    refresh_every: int,
    kmin: int,
    kmax: int,
    max_start_factor: float,
    single_names: List[str],
    single_rounds: int,
    single_mode: str,
    pair_names: List[str],
    pair_rounds: int,
    pair_max_pairs: int,
    beam_size: int,
):
    best = core.normalize_inits(start)
    best_m = core.compute_hard_metrics(best)
    beam: List[Tuple[float, Dict[str, str]]] = [(best_m.MRED, best)]
    write_best(out_dir, stage, best, best_m)
    print(f"\n[escape] start {metric_str(best_m)}")

    pool = neutral_scan(best, pool_names, neutral_top)
    if not pool:
        print("[escape] neutral pool is empty; stop")
        return best, best_m

    for it in range(1, iters + 1):
        if refresh_every > 0 and (it == 1 or (it - 1) % refresh_every == 0):
            pool = neutral_scan(best, pool_names, neutral_top)

        # choose source from beam: mostly best, sometimes near-best
        if len(beam) > 1 and random.random() < 0.35:
            _, src = random.choice(beam[:min(len(beam), beam_size)])
        else:
            src = best
        src_m = core.compute_hard_metrics(src)
        k = random.randint(kmin, kmax)
        flips = weighted_sample(pool, k)
        trial_ints = flip_bits(core.inits_to_ints(src), flips)
        trial = core.ints_to_inits(trial_ints)
        trial_m = core.compute_hard_metrics(trial)
        flip_s = ",".join([f"{n}:{b}" for n, b in flips])
        print(f"\n[escape] iter={it}/{iters} src_MRED={src_m.MRED:.10f} k={k} trial_MRED={trial_m.MRED:.10f} flips={flip_s}")

        # Avoid wasting local polish on completely destroyed candidates.
        if trial_m.MRED > best_m.MRED * max_start_factor:
            print(f"[escape] skip local polish: trial too bad, threshold={best_m.MRED * max_start_factor:.10f}")
            continue

        loc, loc_m = local_polish(trial, single_names, single_rounds, single_mode, pair_rounds, pair_names, pair_max_pairs)
        print(f"[escape] local result {metric_str(loc_m)} best={best_m.MRED:.10f}")

        # maintain beam even if not global best
        beam.append((loc_m.MRED, loc))
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_size]

        if loc_m.MRED < best_m.MRED:
            old = best_m.MRED
            best, best_m = loc, loc_m
            print(f"[escape] ACCEPT NEW BEST {old:.10f} -> {best_m.MRED:.10f}")
            write_best(out_dir, stage, best, best_m)
            print_top_cases(best, min(20, 10))
        else:
            print("[escape] reject global best update")

    print(f"\n[escape] final {metric_str(best_m)}")
    write_best(out_dir, stage, best, best_m)
    return best, best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-inits-json", required=True)
    ap.add_argument("--out-dir", default="runs_escape_local")
    ap.add_argument("--log-file", default="terminal_log.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-name", default="escape_local")

    ap.add_argument("--top-cases", type=int, default=30)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--pool-lut-names", default="all")
    ap.add_argument("--neutral-top", type=int, default=180)
    ap.add_argument("--refresh-every", type=int, default=10)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-start-factor", type=float, default=8.0)

    ap.add_argument("--single-lut-names", default="all")
    ap.add_argument("--single-rounds", type=int, default=10)
    ap.add_argument("--single-mode", choices=["first", "best"], default="first")

    ap.add_argument("--pair-lut-names", default="u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut2,low_lut3,mid_lut2,mid_lut3,high_lut2,high_lut3")
    ap.add_argument("--pair-rounds", type=int, default=1)
    ap.add_argument("--pair-max-pairs", type=int, default=80000)
    ap.add_argument("--beam-size", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    log_handle, old_stdout, old_stderr = install_tee(out_dir / args.log_file)
    try:
        print(f"Log file: {out_dir / args.log_file}")
        print(f"Args: {vars(args)}")
        base = load_inits(Path(args.base_inits_json))
        base_m = core.compute_hard_metrics(base)
        print(f"Base {metric_str(base_m)}")
        for n in core.TRAIN_NAMES:
            print(f"  {n:10s} = {base[n]}")
        write_best(out_dir, args.stage_name + "+base", base, base_m)
        print_top_cases(base, args.top_cases)

        pool_names = parse_names(args.pool_lut_names)
        single_names = parse_names(args.single_lut_names)
        pair_names = parse_names(args.pair_lut_names)
        escape_search(
            start=base,
            out_dir=out_dir,
            stage=args.stage_name,
            iters=args.iters,
            pool_names=pool_names,
            neutral_top=args.neutral_top,
            refresh_every=args.refresh_every,
            kmin=args.kmin,
            kmax=args.kmax,
            max_start_factor=args.max_start_factor,
            single_names=single_names,
            single_rounds=args.single_rounds,
            single_mode=args.single_mode,
            pair_names=pair_names,
            pair_rounds=args.pair_rounds,
            pair_max_pairs=args.pair_max_pairs,
            beam_size=args.beam_size,
        )
    finally:
        sys.stdout.flush()
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        log_handle.close()


if __name__ == "__main__":
    main()
