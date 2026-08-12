#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative post-search for approx66_unshared_paircomp.

Purpose:
  Start from a strong JSON INIT (e.g. MRED ~= 0.1085) and continue searching with:
    1) WCE constraint, usually WCE <= 930
    2) top-relative-error guided candidate bits
    3) constrained single-bit and pair-bit local search
    4) neutral-bit k-flip escape + constrained polish

This script intentionally imports the previous paircomp implementation:
  train_approx66_unshared_paircomp.py
Put both files in the same directory.

Terminal output is mirrored exactly to terminal_log.txt.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np

# Import architecture/evaluator/writer from previous script.
try:
    from train_approx66_unshared_paircomp import (
        TRAIN_NAMES, USED_BITS, SEGMENTS, APPROX_IDS,
        normalize_inits, load_json_inits, inits_to_ints, ints_to_inits,
        compute_hard_metrics, metrics_to_dict, write_best, EVAL,
        parse_lut_names, candidate_bits, set_seed, install_tee,
    )
except Exception as e:
    print("ERROR: failed to import train_approx66_unshared_paircomp.py")
    print("Put train_approx66_conservative_wce.py and train_approx66_unshared_paircomp.py in the same directory.")
    raise

Bit = Tuple[str, int]

# ============================================================
# Hard forward with values and trace helpers
# ============================================================

def hard_values_and_bits(ints: Dict[str, int]):
    """Return approx values and intermediate bit vectors for all 4096 cases."""
    tabs = EVAL.tables(ints)
    plow = EVAL.approx62("low", EVAL.lowb, tabs)
    pmid = EVAL.approx62("mid", EVAL.midb, tabs)
    phigh = EVAL.approx62("high", EVAL.highb, tabs)
    prod = [None] * 12
    prod[0], prod[1], prod[10], prod[11] = plow[0], plow[1], phigh[6], phigh[7]
    prod[2], prod[3] = EVAL.lut62(tabs["u_comp23"], plow[2], pmid[0], plow[3], pmid[1], EVAL.z, EVAL.o)
    prod[4] = EVAL.lut6(tabs["u_comp4"], plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
    prod[5] = EVAL.lut6(tabs["u_comp5"], plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
    prod[6] = EVAL.lut6(tabs["u_comp6"], plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
    prod[7] = EVAL.lut6(tabs["u_comp7"], plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
    prod[8], prod[9] = EVAL.lut62(tabs["u_comp89"], pmid[6], phigh[4], pmid[7], phigh[5], EVAL.z, EVAL.o)
    val = np.zeros_like(EVAL.exact)
    for i, p in enumerate(prod):
        val += p.astype(np.int64) << i
    return val, {"plow": plow, "pmid": pmid, "phigh": phigh, "prod": prod}


def addr_lut62_scalar(seg: str, lut_id: str, a: int, b2: int) -> Tuple[int, int]:
    """Return O5 addr and O6 addr for one approx62 LUT instance."""
    b0 = b2 & 1
    b1 = (b2 >> 1) & 1
    abit = [(a >> i) & 1 for i in range(6)]
    if lut_id == "lut1":
        addr5 = b0 + (b1 << 1) + (abit[0] << 2) + (abit[1] << 3) + (1 << 4)
    elif lut_id == "lut2":
        addr5 = b0 + (b1 << 1) + (abit[1] << 2) + (abit[2] << 3) + (abit[3] << 4)
    elif lut_id == "lut3":
        addr5 = b0 + (b1 << 1) + (abit[3] << 2) + (abit[4] << 3) + (abit[5] << 4)
    elif lut_id == "lut4":
        addr5 = b0 + (b1 << 1) + (abit[4] << 2) + (abit[5] << 3) + (1 << 4)
    else:
        raise ValueError(lut_id)
    return int(addr5), int(addr5 + 32)


def top_error_indices(ints: Dict[str, int], topk: int):
    val, _bits = hard_values_and_bits(ints)
    exact = EVAL.exact
    mask = exact > 0
    rel = np.zeros_like(exact, dtype=np.float64)
    rel[mask] = np.abs(val[mask] - exact[mask]) / exact[mask]
    order = np.argsort(-rel)
    order = [int(i) for i in order if exact[i] > 0 and rel[i] > 0][:topk]
    return order, val, rel


def print_top_errors(ints: Dict[str, int], topk: int):
    idxs, val, rel = top_error_indices(ints, topk)
    print(f"\n[top-error] top {len(idxs)} relative-error cases")
    for rank, idx in enumerate(idxs[:topk], 1):
        a = int(EVAL.a[idx]); b = int(EVAL.b[idx]); ex = int(EVAL.exact[idx]); ap = int(val[idx])
        print(f"[top-error] #{rank:02d} a={a:02d} b={b:02d} exact={ex:04d} approx={ap:04d} abs_err={abs(ap-ex):04d} rel={rel[idx]:.6f}")


def top_error_candidate_bits(ints: Dict[str, int], topk: int, expand_neighbors: bool = True) -> List[Bit]:
    """Collect LUT INIT addresses actually used by top relative-error cases."""
    idxs, _val, _rel = top_error_indices(ints, topk)
    val, bits = hard_values_and_bits(ints)
    cands: Set[Bit] = set()
    for idx in idxs:
        a = int(EVAL.a[idx]); b = int(EVAL.b[idx])
        # approx62 addresses
        for seg, b2 in [("low", b & 3), ("mid", (b >> 2) & 3), ("high", (b >> 4) & 3)]:
            for lut_id in APPROX_IDS:
                a5, a6 = addr_lut62_scalar(seg, lut_id, a, b2)
                cands.add((f"{seg}_{lut_id}", a5))
                cands.add((f"{seg}_{lut_id}", a6))
        plow, pmid, phigh = bits["plow"], bits["pmid"], bits["phigh"]
        # comp23 / comp89 addresses
        base23 = int(plow[2][idx]) + (int(pmid[0][idx]) << 1) + (int(plow[3][idx]) << 2) + (int(pmid[1][idx]) << 3)
        cands.add(("u_comp23", base23))
        cands.add(("u_comp23", base23 + 32))
        base89 = int(pmid[6][idx]) + (int(phigh[4][idx]) << 1) + (int(pmid[7][idx]) << 2) + (int(phigh[5][idx]) << 3)
        cands.add(("u_comp89", base89))
        cands.add(("u_comp89", base89 + 32))
        # comp4/5 same address, comp6/7 same address
        addr45 = int(plow[4][idx]) + (int(pmid[2][idx]) << 1) + (int(phigh[0][idx]) << 2) + (int(plow[5][idx]) << 3) + (int(pmid[3][idx]) << 4) + (int(phigh[1][idx]) << 5)
        addr67 = int(plow[6][idx]) + (int(pmid[4][idx]) << 1) + (int(phigh[2][idx]) << 2) + (int(plow[7][idx]) << 3) + (int(pmid[5][idx]) << 4) + (int(phigh[3][idx]) << 5)
        cands.add(("u_comp4", addr45)); cands.add(("u_comp5", addr45))
        cands.add(("u_comp6", addr67)); cands.add(("u_comp7", addr67))
    # Optionally add +/-1 nearby truth-table entries for small robustness.
    if expand_neighbors:
        extra: Set[Bit] = set()
        for n, b in cands:
            if b - 1 in USED_BITS[n]: extra.add((n, b - 1))
            if b + 1 in USED_BITS[n]: extra.add((n, b + 1))
        cands |= extra
    # Filter only used bits.
    out = sorted([(n, b) for (n, b) in cands if n in USED_BITS and b in USED_BITS[n]], key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
    print(f"[top-candidates] topk={topk} expand_neighbors={expand_neighbors} candidate_bits={len(out)}")
    return out

# ============================================================
# Constrained search primitives
# ============================================================

def passes_constraint(m, max_wce: int, max_er: float, max_med: float) -> bool:
    if max_wce >= 0 and m.WCE > max_wce:
        return False
    if max_er >= 0 and m.ER > max_er:
        return False
    if max_med >= 0 and m.MED > max_med:
        return False
    return True


def bit_list_from_names(names: List[str]) -> List[Bit]:
    return [(n, b) for n in names for b in USED_BITS[n]]


def neutral_bits(ints: Dict[str, int], source_bits: List[Bit], topn: int, margin: float, max_wce: int, max_er: float, max_med: float) -> List[Bit]:
    base_m = EVAL.evaluate_ints(ints)
    scored = []
    print(f"\n[neutral] scan source_bits={len(source_bits)} base_MRED={base_m.MRED:.10f} margin={margin} max_wce={max_wce}")
    for n, b in source_bits:
        trial = dict(ints); trial[n] ^= (1 << b)
        m = EVAL.evaluate_ints(trial)
        delta = m.MRED - base_m.MRED
        ok = passes_constraint(m, max_wce, max_er, max_med)
        if ok and delta <= margin:
            scored.append((delta, n, b, m))
    scored.sort(key=lambda x: x[0])
    out = [(n, b) for _d, n, b, _m in scored[:topn]]
    print(f"[neutral] kept={len(out)} / eligible={len(scored)}")
    for d, n, b, m in scored[:min(20, len(scored))]:
        print(f"[neutral] bit=({n},{b:02d}) delta={d:+.10f} MRED={m.MRED:.10f} MED={m.MED:.4f} ER={m.ER:.4f} WCE={m.WCE}")
    return out


def constrained_single(start: Dict[str, str], bits: List[Bit], rounds: int, mode: str, random_order: bool, max_wce: int, max_er: float, max_med: float, eps: float = 1e-12):
    cur = inits_to_ints(start)
    cur_m = EVAL.evaluate_ints(cur)
    print(f"\n[single-c] start MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE} rounds={rounds} mode={mode} bits={len(bits)} max_wce={max_wce}")
    for r in range(rounds):
        order = list(bits)
        if random_order:
            random.shuffle(order)
        improved = False
        if mode == "best":
            best_trial = None; best_m = cur_m; old = cur_m.MRED
            for n, b in order:
                trial = dict(cur); trial[n] ^= (1 << b)
                m = EVAL.evaluate_ints(trial)
                if passes_constraint(m, max_wce, max_er, max_med) and m.MRED + eps < best_m.MRED:
                    best_trial = (n, b, trial); best_m = m
            if best_trial:
                n, b, cur = best_trial; cur_m = best_m; improved = True
                print(f"[single-c] KEEP-BEST round={r+1} lut={n} bit={b:02d} MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
        else:
            for n, b in order:
                trial = dict(cur); trial[n] ^= (1 << b)
                m = EVAL.evaluate_ints(trial)
                if passes_constraint(m, max_wce, max_er, max_med) and m.MRED + eps < cur_m.MRED:
                    old = cur_m.MRED; cur = trial; cur_m = m; improved = True
                    print(f"[single-c] KEEP round={r+1} lut={n} bit={b:02d} MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
                    break
        if not improved:
            print(f"[single-c] round={r+1} no improvement, stop")
            break
    print(f"[single-c] final MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
    return ints_to_inits(cur), cur_m


def constrained_pair(start: Dict[str, str], bits: List[Bit], rounds: int, mode: str, random_order: bool, max_pairs: int, max_wce: int, max_er: float, max_med: float, eps: float = 1e-12):
    cur = inits_to_ints(start)
    cur_m = EVAL.evaluate_ints(cur)
    nbits = len(bits)
    full_pairs = nbits * (nbits - 1) // 2
    print(f"\n[pair-c] start MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE} rounds={rounds} mode={mode} bits={nbits} full_pairs={full_pairs} max_pairs={max_pairs} max_wce={max_wce}")
    for r in range(rounds):
        pair_idx = [(i, j) for i in range(nbits) for j in range(i + 1, nbits)]
        if random_order:
            random.shuffle(pair_idx)
        if max_pairs > 0 and len(pair_idx) > max_pairs:
            pair_idx = pair_idx[:max_pairs]
        improved = False
        if mode == "best":
            best = None; best_m = cur_m; old = cur_m.MRED
            for i, j in pair_idx:
                n1, b1 = bits[i]; n2, b2 = bits[j]
                trial = dict(cur)
                trial[n1] ^= (1 << b1)
                trial[n2] ^= (1 << b2)
                m = EVAL.evaluate_ints(trial)
                if passes_constraint(m, max_wce, max_er, max_med) and m.MRED + eps < best_m.MRED:
                    best = (n1, b1, n2, b2, trial); best_m = m
            if best:
                n1, b1, n2, b2, cur = best; cur_m = best_m; improved = True
                print(f"[pair-c] KEEP-BEST round={r+1} ({n1},{b1:02d})+({n2},{b2:02d}) MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
        else:
            for i, j in pair_idx:
                n1, b1 = bits[i]; n2, b2 = bits[j]
                trial = dict(cur)
                trial[n1] ^= (1 << b1)
                trial[n2] ^= (1 << b2)
                m = EVAL.evaluate_ints(trial)
                if passes_constraint(m, max_wce, max_er, max_med) and m.MRED + eps < cur_m.MRED:
                    old = cur_m.MRED; cur = trial; cur_m = m; improved = True
                    print(f"[pair-c] KEEP round={r+1} ({n1},{b1:02d})+({n2},{b2:02d}) MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
                    break
        if not improved:
            print(f"[pair-c] round={r+1} no improvement, stop")
            break
    print(f"[pair-c] final MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
    return ints_to_inits(cur), cur_m


def neutral_escape(start: Dict[str, str], neutral: List[Bit], polish_bits: List[Bit], iters: int, kmin: int, kmax: int, single_rounds: int, pair_after: bool, pair_rounds: int, pair_max_pairs: int, max_wce: int, max_er: float, max_med: float):
    best = normalize_inits(start)
    best_m = compute_hard_metrics(best)
    print(f"\n[escape-c] start MRED={best_m.MRED:.10f} MED={best_m.MED:.4f} ER={best_m.ER:.4f} WCE={best_m.WCE} iters={iters} neutral={len(neutral)} k=[{kmin},{kmax}]")
    if not neutral:
        print("[escape-c] no neutral bits; skip")
        return best, best_m

    if len(neutral) < kmin:
        print(f"[escape-c] neutral bits only {len(neutral)} < kmin={kmin}; reduce kmin to {len(neutral)}")
        kmin = len(neutral)

    kmax = min(kmax, len(neutral))

    for it in range(1, iters + 1):
        ints = inits_to_ints(best)
        k = random.randint(kmin, kmax)
        chosen = random.sample(neutral, k)
        for n, b in chosen:
            ints[n] ^= (1 << b)
        pert = ints_to_inits(ints)
        pm = compute_hard_metrics(pert)
        print(f"[escape-c] iter={it}/{iters} perturb k={k} MRED={pm.MRED:.10f} MED={pm.MED:.4f} ER={pm.ER:.4f} WCE={pm.WCE}")
        loc, lm = constrained_single(pert, polish_bits, single_rounds, "first", True, max_wce, max_er, max_med)
        if pair_after:
            loc, lm = constrained_pair(loc, polish_bits, pair_rounds, "first", True, pair_max_pairs, max_wce, max_er, max_med)
        if passes_constraint(lm, max_wce, max_er, max_med) and lm.MRED < best_m.MRED:
            old = best_m.MRED; best = loc; best_m = lm
            print(f"[escape-c] ACCEPT iter={it} MRED {old:.10f}->{best_m.MRED:.10f} MED={best_m.MED:.4f} ER={best_m.ER:.4f} WCE={best_m.WCE}")
        else:
            print(f"[escape-c] reject iter={it} local_MRED={lm.MRED:.10f} best={best_m.MRED:.10f}")
    print(f"[escape-c] final MRED={best_m.MRED:.10f} MED={best_m.MED:.4f} ER={best_m.ER:.4f} WCE={best_m.WCE}")
    return best, best_m

# ============================================================
# Main
# ============================================================

def save_current(out_dir: Path, stage: str, inits: Dict[str, str], m, prefix="best"):
    record = {"stage": stage, "epoch": stage, "loss": None, "metrics": metrics_to_dict(m), "inits": normalize_inits(inits)}
    write_best(out_dir, record, prefix)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-inits-json", required=True)
    ap.add_argument("--out-dir", default="runs_conservative")
    ap.add_argument("--log-file", default="terminal_log.txt")
    ap.add_argument("--stage-name", default="conservative_wce")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-wce", type=int, default=930, help="Reject candidates with WCE above this. Use -1 to disable.")
    ap.add_argument("--max-er", type=float, default=-1.0, help="Optional ER constraint. Use -1 to disable.")
    ap.add_argument("--max-med", type=float, default=-1.0, help="Optional MED constraint. Use -1 to disable.")

    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--print-topk", type=int, default=30)
    ap.add_argument("--candidate-mode", choices=["all", "top", "neutral", "top_neutral"], default="top_neutral")
    ap.add_argument("--lut-names", default="all", help="Base LUT pool for all/neutral candidates.")
    ap.add_argument("--extra-lut-names", default="u_comp23,u_comp4,u_comp5,u_comp6,u_comp7,u_comp89,low_lut1,low_lut2,low_lut3,mid_lut1,mid_lut2,mid_lut3,high_lut2,high_lut3")

    ap.add_argument("--neutral-top", type=int, default=160)
    ap.add_argument("--neutral-margin", type=float, default=0.004)

    ap.add_argument("--single-rounds", type=int, default=30)
    ap.add_argument("--single-mode", choices=["first", "best"], default="best")
    ap.add_argument("--single-random-order", action="store_true")

    ap.add_argument("--pair-rounds", type=int, default=6)
    ap.add_argument("--pair-mode", choices=["first", "best"], default="first")
    ap.add_argument("--pair-random-order", action="store_true")
    ap.add_argument("--pair-max-pairs", type=int, default=120000)

    ap.add_argument("--escape-iters", type=int, default=120)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--escape-single-rounds", type=int, default=10)
    ap.add_argument("--escape-pair-after", action="store_true")
    ap.add_argument("--escape-pair-rounds", type=int, default=2)
    ap.add_argument("--escape-pair-max-pairs", type=int, default=50000)

    ap.add_argument("--do-single", action="store_true")
    ap.add_argument("--do-pair", action="store_true")
    ap.add_argument("--do-escape", action="store_true")
    ap.add_argument("--do-final-single", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    log_handle = install_tee(out_dir / args.log_file)
    try:
        print(f"Log file: {out_dir / args.log_file}")
        print(f"Stage: {args.stage_name}")
        print(f"Args: {vars(args)}")
        inits = load_json_inits(Path(args.base_inits_json))
        m = compute_hard_metrics(inits)
        print(f"Base metrics: MRED={m.MRED:.10f} MED={m.MED:.6f} ER={m.ER:.6f} WCE={m.WCE}")
        if not passes_constraint(m, args.max_wce, args.max_er, args.max_med):
            print("WARNING: base design does not satisfy constraints. Search will only accept constrained improvements.")
        print_top_errors(inits_to_ints(inits), args.print_topk)
        save_current(out_dir, args.stage_name + "+base", inits, m, "best")

        # Build candidates.
        pool_names = parse_lut_names(args.lut_names)
        pool_bits = bit_list_from_names(pool_names)
        extra_names = parse_lut_names(args.extra_lut_names)
        extra_bits = bit_list_from_names(extra_names)
        top_bits = top_error_candidate_bits(inits_to_ints(inits), args.topk, expand_neighbors=True)
        neutral_source = sorted(set(top_bits + extra_bits), key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
        neu_bits = neutral_bits(inits_to_ints(inits), neutral_source, args.neutral_top, args.neutral_margin, args.max_wce, args.max_er, args.max_med)

        if args.candidate_mode == "all":
            bits = pool_bits
        elif args.candidate_mode == "top":
            bits = top_bits
        elif args.candidate_mode == "neutral":
            bits = neu_bits
        else:
            bits = sorted(set(top_bits + neu_bits), key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
        print(f"\n[candidates] mode={args.candidate_mode} final_bits={len(bits)}")
        if len(bits) <= 120:
            print("[candidates] " + ", ".join([f"{n}:{b}" for n,b in bits]))

        cur = inits
        cur_m = m
        if args.do_single:
            cur, cur_m = constrained_single(cur, bits, args.single_rounds, args.single_mode, args.single_random_order, args.max_wce, args.max_er, args.max_med)
            save_current(out_dir, args.stage_name + "+single", cur, cur_m, "best")
            print_top_errors(inits_to_ints(cur), args.print_topk)
        if args.do_pair:
            cur, cur_m = constrained_pair(cur, bits, args.pair_rounds, args.pair_mode, args.pair_random_order, args.pair_max_pairs, args.max_wce, args.max_er, args.max_med)
            save_current(out_dir, args.stage_name + "+pair", cur, cur_m, "best")
            print_top_errors(inits_to_ints(cur), args.print_topk)
        if args.do_escape:
            # Recompute neutral bits from current best; local landscape may have changed.
            top_bits2 = top_error_candidate_bits(inits_to_ints(cur), args.topk, expand_neighbors=True)
            neutral_source2 = sorted(set(top_bits2 + extra_bits), key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
            neu_bits2 = neutral_bits(inits_to_ints(cur), neutral_source2, args.neutral_top, args.neutral_margin, args.max_wce, args.max_er, args.max_med)
            polish_bits = sorted(set(top_bits2 + neu_bits2), key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
            cur, cur_m = neutral_escape(cur, neu_bits2, polish_bits, args.escape_iters, args.kmin, args.kmax, args.escape_single_rounds, args.escape_pair_after, args.escape_pair_rounds, args.escape_pair_max_pairs, args.max_wce, args.max_er, args.max_med)
            save_current(out_dir, args.stage_name + "+escape", cur, cur_m, "best")
            print_top_errors(inits_to_ints(cur), args.print_topk)
        if args.do_final_single:
            # Final global constrained polish over selected important LUTs + top bits.
            top_bits3 = top_error_candidate_bits(inits_to_ints(cur), max(args.topk, 120), expand_neighbors=True)
            final_bits = sorted(set(top_bits3 + extra_bits), key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
            cur, cur_m = constrained_single(cur, final_bits, args.single_rounds, "best", False, args.max_wce, args.max_er, args.max_med)
            save_current(out_dir, args.stage_name + "+final_single", cur, cur_m, "best")
            print_top_errors(inits_to_ints(cur), args.print_topk)

        print("\nFinished conservative WCE search")
        print(f"Final best: MRED={cur_m.MRED:.10f} MED={cur_m.MED:.6f} ER={cur_m.ER:.6f} WCE={cur_m.WCE}")
        print(f"Best JSON: {out_dir / 'best_approx66_inits.json'}")
        print(f"Best Verilog: {out_dir / 'best_approx66_unshared_paircomp.v'}")
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        log_handle.flush(); log_handle.close()

if __name__ == "__main__":
    main()
