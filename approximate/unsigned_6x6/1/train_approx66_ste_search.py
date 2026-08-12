#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STE-based LUT INIT search for the 6x6 approximate multiplier.

Key features:
  1) One epoch uses all 4096 input pairs as a full batch.
  2) INIT bits are trainable continuous logits.
  3) INIT/output sharpening uses f_c(x)=x^c/(x^c+(1-x)^c).
  4) Middle LUT outputs are binarized with STE.
  5) Terminal output is mirrored exactly to the txt log file.
  6) Supports exploration from random INITs and fine search from a saved best JSON.
  7) Optional best-restart and greedy bit-flip refinement.

Example:
  python3 train_approx66_ste_search.py --init-mode random --epochs 1000 --out-dir runs/explore
  python3 train_approx66_ste_search.py --init-mode json --base-inits-json runs/explore/best_approx66_inits.json --epochs 800 --out-dir runs/refine
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


# ============================================================
# 1. Default hand-designed INITs from the current Verilog files
# ============================================================
DEFAULT_APPROX62 = {
    "lut1": "EAC00000A0A00000",
    "lut2": "EEAACC00EAC0EAC0",
    "lut3": "E6AACC006AC0EAC0",
    "lut4": "800000004C000000",
}

DEFAULT_COMP66 = {
    "u_or23": "0000FFF80000FEE6",
    "u_or89": "00005F5800005E4E",
}

TRAINABLE_LUT_NAMES = ["lut1", "lut2", "lut3", "lut4", "u_or23", "u_or89"]


# ============================================================
# 2. Logging: make terminal output exactly equal to txt output
# ============================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def install_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    tee = Tee(sys.__stdout__, f)
    sys.stdout = tee
    sys.stderr = tee
    return f


# ============================================================
# 3. Basic utility functions
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_hex(init_hex: str) -> str:
    s = str(init_hex).strip()
    s = s.replace("64'h", "").replace("64'H", "").replace("0x", "").replace("0X", "")
    s = s.replace("_", "")
    if len(s) > 16:
        s = s[-16:]
    return s.zfill(16).upper()


def hex_to_bits64(init_hex: str) -> List[int]:
    value = int(clean_hex(init_hex), 16)
    return [(value >> i) & 1 for i in range(64)]


def bits_to_hex64(bits: Iterable[int]) -> str:
    value = 0
    for i, b in enumerate(bits):
        if int(b) != 0:
            value |= (1 << i)
    return f"64'h{value:016X}"


def normalize_inits(inits: Dict[str, str]) -> Dict[str, str]:
    return {name: bits_to_hex64(hex_to_bits64(inits[name])) for name in TRAINABLE_LUT_NAMES}


def default_inits() -> Dict[str, str]:
    inits = {}
    inits.update(DEFAULT_APPROX62)
    inits.update(DEFAULT_COMP66)
    return normalize_inits(inits)


def random_inits(prob_one: float) -> Dict[str, str]:
    prob_one = min(max(float(prob_one), 0.0), 1.0)
    out = {}
    for name in TRAINABLE_LUT_NAMES:
        bits = [1 if random.random() < prob_one else 0 for _ in range(64)]
        out[name] = bits_to_hex64(bits)
    return out


def load_inits_json(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "inits" in obj:
        obj = obj["inits"]
    return normalize_inits(obj)


def sharp01(x: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    """f_c(x)=x^c/(x^c+(1-x)^c)."""
    x = torch.clamp(x, eps, 1.0 - eps)
    xc = torch.pow(x, c)
    yc = torch.pow(1.0 - x, c)
    return xc / (xc + yc + eps)


def ste_binarize(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    hard = (x >= threshold).to(x.dtype)
    return hard.detach() - x.detach() + x


def int_bits(x: torch.Tensor, width: int) -> List[torch.Tensor]:
    return [((x >> i) & 1).to(torch.float32) for i in range(width)]


# ============================================================
# 4. Differentiable hard-forward LUT model
# ============================================================
class TrainableLUT6_2(nn.Module):
    def __init__(self, init_hex: str, init_p: float = 0.70, noise_std: float = 0.25):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_p=init_p, noise_std=noise_std)

    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_p: float = 0.70, noise_std: float = 0.25) -> None:
        bits = torch.tensor(hex_to_bits64(init_hex), dtype=torch.float32, device=self.logits.device)
        init_p = float(init_p)
        init_p = min(max(init_p, 1e-4), 1.0 - 1e-4)
        p1 = torch.full_like(bits, init_p)
        p0 = torch.full_like(bits, 1.0 - init_p)
        p = torch.where(bits > 0.5, p1, p0)
        logits = torch.log(p / (1.0 - p))
        if noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std
        self.logits.copy_(logits)

    def table_prob(self, c_init: float) -> torch.Tensor:
        raw = torch.sigmoid(self.logits)
        return sharp01(raw, c_init)

    @staticmethod
    def _soft_lut(inputs: List[torch.Tensor], table: torch.Tensor) -> torch.Tensor:
        n_inputs = len(inputs)
        n_addr = 1 << n_inputs
        out = torch.zeros_like(inputs[0])
        for addr in range(n_addr):
            w = torch.ones_like(inputs[0])
            for i, xi in enumerate(inputs):
                if (addr >> i) & 1:
                    w = w * xi
                else:
                    w = w * (1.0 - xi)
            out = out + w * table[addr]
        return out

    def forward(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        I2: torch.Tensor,
        I3: torch.Tensor,
        I4: torch.Tensor,
        I5: torch.Tensor,
        *,
        c_init: float,
        c_out: float,
        hard_middle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        table = self.table_prob(c_init)

        if hard_middle:
            I0 = ste_binarize(I0)
            I1 = ste_binarize(I1)
            I2 = ste_binarize(I2)
            I3 = ste_binarize(I3)
            I4 = ste_binarize(I4)
            I5 = ste_binarize(I5)

        o5 = self._soft_lut([I0, I1, I2, I3, I4], table[:32])
        o6 = self._soft_lut([I0, I1, I2, I3, I4, I5], table)

        o5 = sharp01(o5, c_out)
        o6 = sharp01(o6, c_out)

        if hard_middle:
            o5 = ste_binarize(o5)
            o6 = ste_binarize(o6)

        return o5, o6

    def hard_bits(self, c_init: float) -> List[int]:
        with torch.no_grad():
            p = self.table_prob(c_init)
            return (p >= 0.5).to(torch.int64).cpu().tolist()

    def hard_hex(self, c_init: float) -> str:
        return bits_to_hex64(self.hard_bits(c_init))

    def binary_regularization(self, c_init: float) -> torch.Tensor:
        p = self.table_prob(c_init)
        return torch.mean(p * (1.0 - p))


class Approx66STE(nn.Module):
    def __init__(self, base_inits: Dict[str, str], init_p: float = 0.70, noise_std: float = 0.25):
        super().__init__()
        base_inits = normalize_inits(base_inits)
        self.luts = nn.ModuleDict({
            name: TrainableLUT6_2(base_inits[name], init_p, noise_std)
            for name in TRAINABLE_LUT_NAMES
        })

    @torch.no_grad()
    def reset_from_inits(self, base_inits: Dict[str, str], init_p: float, noise_std: float) -> None:
        base_inits = normalize_inits(base_inits)
        for name in TRAINABLE_LUT_NAMES:
            self.luts[name].reset_from_hex(base_inits[name], init_p=init_p, noise_std=noise_std)

    @staticmethod
    def const_like(x: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full_like(x, float(value))

    @staticmethod
    def fixed_or3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, hard_middle: bool) -> torch.Tensor:
        out = 1.0 - (1.0 - x) * (1.0 - y) * (1.0 - z)
        if hard_middle:
            out = ste_binarize(out)
        return out

    def approx62(self, a: torch.Tensor, b2: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool) -> List[torch.Tensor]:
        a_bits = int_bits(a, 6)
        b_bits = int_bits(b2, 2)
        o = self.const_like(a_bits[0], 1.0)

        p0, p1 = self.luts["lut1"](
            b_bits[0], b_bits[1], a_bits[0], a_bits[1], o, o,
            c_init=c_init, c_out=c_out, hard_middle=hard_middle,
        )
        p2, p3 = self.luts["lut2"](
            b_bits[0], b_bits[1], a_bits[1], a_bits[2], a_bits[3], o,
            c_init=c_init, c_out=c_out, hard_middle=hard_middle,
        )
        p4, p5 = self.luts["lut3"](
            b_bits[0], b_bits[1], a_bits[3], a_bits[4], a_bits[5], o,
            c_init=c_init, c_out=c_out, hard_middle=hard_middle,
        )
        p6, p7 = self.luts["lut4"](
            b_bits[0], b_bits[1], a_bits[4], a_bits[5], o, o,
            c_init=c_init, c_out=c_out, hard_middle=hard_middle,
        )
        return [p0, p1, p2, p3, p4, p5, p6, p7]

    def comp66(self, plow: List[torch.Tensor], pmid: List[torch.Tensor], phigh: List[torch.Tensor], *, c_init: float, c_out: float, hard_middle: bool) -> List[torch.Tensor]:
        z = self.const_like(plow[0], 0.0)
        o = self.const_like(plow[0], 1.0)
        prod: List[torch.Tensor] = [z for _ in range(12)]

        prod[0] = plow[0]
        prod[1] = plow[1]
        prod[10] = phigh[6]
        prod[11] = phigh[7]

        prod[2], prod[3] = self.luts["u_or23"](
            plow[2], pmid[0], plow[3], pmid[1], z, o,
            c_init=c_init, c_out=c_out, hard_middle=hard_middle,
        )

        prod[4] = self.fixed_or3(plow[4], pmid[2], phigh[0], hard_middle)
        prod[5] = self.fixed_or3(plow[5], pmid[3], phigh[1], hard_middle)
        prod[6] = self.fixed_or3(plow[6], pmid[4], phigh[2], hard_middle)
        prod[7] = self.fixed_or3(plow[7], pmid[5], phigh[3], hard_middle)

        prod[8], prod[9] = self.luts["u_or89"](
            pmid[6], phigh[4], pmid[7], phigh[5], z, o,
            c_init=c_init, c_out=c_out, hard_middle=hard_middle,
        )

        return prod

    def forward(self, a: torch.Tensor, b: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        b_low = b & 0b11
        b_mid = (b >> 2) & 0b11
        b_high = (b >> 4) & 0b11

        plow = self.approx62(a, b_low, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        pmid = self.approx62(a, b_mid, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        phigh = self.approx62(a, b_high, c_init=c_init, c_out=c_out, hard_middle=hard_middle)

        prod_bits = self.comp66(plow, pmid, phigh, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod_stack = torch.stack(prod_bits, dim=1)
        weights = torch.tensor([1 << i for i in range(12)], dtype=torch.float32, device=prod_stack.device)
        approx = torch.sum(prod_stack * weights, dim=1)
        return approx, prod_stack

    def hard_inits(self, c_init: float) -> Dict[str, str]:
        return {name: self.luts[name].hard_hex(c_init) for name in TRAINABLE_LUT_NAMES}

    def binary_regularization(self, c_init: float) -> torch.Tensor:
        regs = [self.luts[name].binary_regularization(c_init) for name in TRAINABLE_LUT_NAMES]
        return torch.mean(torch.stack(regs))


# ============================================================
# 5. Verilog-equivalent hard simulator for metric evaluation
# ============================================================
def lut6_2_hard(bits: List[int], I0: int, I1: int, I2: int, I3: int, I4: int, I5: int) -> Tuple[int, int]:
    addr5 = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4)
    addr6 = addr5 + (I5 << 5)
    return bits[addr5], bits[addr6]


def approx62_hard(a: int, b2: int, init_bits: Dict[str, List[int]]) -> List[int]:
    ab = [(a >> i) & 1 for i in range(6)]
    bb = [(b2 >> i) & 1 for i in range(2)]

    p0, p1 = lut6_2_hard(init_bits["lut1"], bb[0], bb[1], ab[0], ab[1], 1, 1)
    p2, p3 = lut6_2_hard(init_bits["lut2"], bb[0], bb[1], ab[1], ab[2], ab[3], 1)
    p4, p5 = lut6_2_hard(init_bits["lut3"], bb[0], bb[1], ab[3], ab[4], ab[5], 1)
    p6, p7 = lut6_2_hard(init_bits["lut4"], bb[0], bb[1], ab[4], ab[5], 1, 1)
    return [p0, p1, p2, p3, p4, p5, p6, p7]


def comp66_hard(plow: List[int], pmid: List[int], phigh: List[int], init_bits: Dict[str, List[int]]) -> int:
    prod = [0] * 12
    prod[0] = plow[0]
    prod[1] = plow[1]
    prod[10] = phigh[6]
    prod[11] = phigh[7]

    prod[2], prod[3] = lut6_2_hard(init_bits["u_or23"], plow[2], pmid[0], plow[3], pmid[1], 0, 1)

    prod[4] = plow[4] | pmid[2] | phigh[0]
    prod[5] = plow[5] | pmid[3] | phigh[1]
    prod[6] = plow[6] | pmid[4] | phigh[2]
    prod[7] = plow[7] | pmid[5] | phigh[3]

    prod[8], prod[9] = lut6_2_hard(init_bits["u_or89"], pmid[6], phigh[4], pmid[7], phigh[5], 0, 1)

    value = 0
    for i, bit in enumerate(prod):
        value |= (int(bit) << i)
    return value


def approx66_hard(a: int, b: int, init_bits: Dict[str, List[int]]) -> int:
    plow = approx62_hard(a, b & 0b11, init_bits)
    pmid = approx62_hard(a, (b >> 2) & 0b11, init_bits)
    phigh = approx62_hard(a, (b >> 4) & 0b11, init_bits)
    return comp66_hard(plow, pmid, phigh, init_bits)


@dataclass
class Metrics:
    total_cases: int
    error_cases: int
    ER: float
    MED: float
    NED: float
    MRED: float
    WCE: int


def compute_hard_metrics(init_hex: Dict[str, str]) -> Metrics:
    init_hex = normalize_inits(init_hex)
    init_bits = {name: hex_to_bits64(init_hex[name]) for name in TRAINABLE_LUT_NAMES}

    total = 0
    err_cases = 0
    abs_sum = 0.0
    rel_sum = 0.0
    rel_count = 0
    wce = 0

    for a in range(64):
        for b in range(64):
            exact = a * b
            approx = approx66_hard(a, b, init_bits)
            err = abs(approx - exact)
            total += 1
            if err != 0:
                err_cases += 1
            abs_sum += err
            wce = max(wce, err)
            if exact != 0:
                rel_sum += err / exact
                rel_count += 1

    med = abs_sum / total
    ned = med / (63 * 63)
    mred = rel_sum / total if total > 0 else 0.0
    er = err_cases / total
    return Metrics(total, err_cases, er, med, ned, mred, wce)


def make_dataset(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aa, bb, exact = [], [], []
    for a in range(64):
        for b in range(64):
            aa.append(a)
            bb.append(b)
            exact.append(a * b)
    return (
        torch.tensor(aa, dtype=torch.long, device=device),
        torch.tensor(bb, dtype=torch.long, device=device),
        torch.tensor(exact, dtype=torch.float32, device=device),
    )


def loss_fn(
    approx: torch.Tensor,
    exact: torch.Tensor,
    bin_reg: torch.Tensor,
    *,
    zero_weight: float,
    med_weight: float,
    bin_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = exact > 0
    abs_err = torch.abs(approx - exact)
    mred = torch.sum(abs_err[mask] / exact[mask]) / float(exact.numel())

    if torch.any(~mask):
        zero_loss = torch.mean(abs_err[~mask]) / 4096.0
    else:
        zero_loss = torch.tensor(0.0, device=approx.device)

    med_norm = torch.mean(abs_err) / 4096.0
    loss = mred + zero_weight * zero_loss + med_weight * med_norm + bin_weight * bin_reg
    parts = {
        "mred_loss": float(mred.detach().cpu()),
        "zero_loss": float(zero_loss.detach().cpu()),
        "med_norm": float(med_norm.detach().cpu()),
        "bin_reg": float(bin_reg.detach().cpu()),
    }
    return loss, parts


def metrics_to_dict(m: Metrics) -> Dict:
    return {
        "total_cases": m.total_cases,
        "error_cases": m.error_cases,
        "ER": m.ER,
        "MED": m.MED,
        "NED": m.NED,
        "MRED": m.MRED,
        "WCE": m.WCE,
    }


def write_best_files(out_dir: Path, best: Dict, c_init: float, prefix: str = "best") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{prefix}_approx66_inits.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    # Also maintain conventional names for scripts that chain stages.
    if prefix == "best":
        conventional_json = out_dir / "best_approx66_inits.json"
        if conventional_json != json_path:
            with conventional_json.open("w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)

    snippet_path = out_dir / f"{prefix}_approx66_verilog_snippet.v"
    inits = best["inits"]
    with snippet_path.open("w", encoding="utf-8") as f:
        f.write("// Best INITs found by train_approx66_ste_search.py\n")
        f.write(f"// stage = {best.get('stage', '')}\n")
        f.write(f"// epoch = {best['epoch']}\n")
        f.write(f"// MRED  = {best['metrics']['MRED']:.10f}\n")
        f.write(f"// MED   = {best['metrics']['MED']:.10f}\n")
        f.write(f"// ER    = {best['metrics']['ER']:.10f}\n")
        f.write(f"// WCE   = {best['metrics']['WCE']}\n")
        f.write(f"// c_init used for threshold = {c_init}\n\n")
        f.write("// approx62.v\n")
        f.write(f"// LUT6_inst1 INIT = {inits['lut1']}\n")
        f.write(f"// LUT6_inst2 INIT = {inits['lut2']}\n")
        f.write(f"// LUT6_inst3 INIT = {inits['lut3']}\n")
        f.write(f"// LUT6_inst4 INIT = {inits['lut4']}\n\n")
        f.write("// comp66_3_opt.v\n")
        f.write(f"// u_or23 INIT     = {inits['u_or23']}\n")
        f.write(f"// u_or89 INIT     = {inits['u_or89']}\n")

    if prefix == "best":
        conventional_snippet = out_dir / "best_approx66_verilog_snippet.v"
        if conventional_snippet != snippet_path:
            conventional_snippet.write_text(snippet_path.read_text(encoding="utf-8"), encoding="utf-8")


def greedy_bitflip(
    start_inits: Dict[str, str],
    *,
    max_rounds: int = 3,
    eps: float = 1e-12,
) -> Tuple[Dict[str, str], Metrics]:
    current = normalize_inits(start_inits)
    current_metrics = compute_hard_metrics(current)
    print("\n[bitflip] start "
          f"MRED={current_metrics.MRED:.10f} MED={current_metrics.MED:.4f} "
          f"ER={current_metrics.ER:.4f} WCE={current_metrics.WCE}")

    for r in range(max_rounds):
        improved_this_round = False
        print(f"[bitflip] round {r + 1}/{max_rounds} begin")
        for name in TRAINABLE_LUT_NAMES:
            bits = hex_to_bits64(current[name])
            for bit_idx in range(64):
                trial_bits = bits.copy()
                trial_bits[bit_idx] ^= 1
                trial = dict(current)
                trial[name] = bits_to_hex64(trial_bits)
                trial_metrics = compute_hard_metrics(trial)
                if trial_metrics.MRED + eps < current_metrics.MRED:
                    current = trial
                    bits = trial_bits
                    old_mred = current_metrics.MRED
                    current_metrics = trial_metrics
                    improved_this_round = True
                    print(
                        f"[bitflip] KEEP round={r + 1} lut={name} bit={bit_idx:02d} "
                        f"MRED {old_mred:.10f} -> {current_metrics.MRED:.10f} "
                        f"MED={current_metrics.MED:.4f} ER={current_metrics.ER:.4f} WCE={current_metrics.WCE}"
                    )
        if not improved_this_round:
            print(f"[bitflip] round {r + 1} no improvement, stop")
            break
    print("[bitflip] final "
          f"MRED={current_metrics.MRED:.10f} MED={current_metrics.MED:.4f} "
          f"ER={current_metrics.ER:.4f} WCE={current_metrics.WCE}")
    return current, current_metrics


def make_base_inits(args) -> Dict[str, str]:
    if args.init_mode == "manual":
        return default_inits()
    if args.init_mode == "random":
        return random_inits(args.random_init_prob)
    if args.init_mode == "json":
        if not args.base_inits_json:
            raise ValueError("--init-mode json requires --base-inits-json")
        return load_inits_json(Path(args.base_inits_json))
    raise ValueError(f"Unknown init mode: {args.init_mode}")


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--init-mode", type=str, default="manual", choices=["manual", "random", "json"])
    parser.add_argument("--base-inits-json", type=str, default="")
    parser.add_argument("--random-init-prob", type=float, default=0.5)
    parser.add_argument("--init-p", type=float, default=0.70)
    parser.add_argument("--noise-std", type=float, default=0.25)

    parser.add_argument("--c-init", type=float, default=2.0)
    parser.add_argument("--c-out", type=float, default=2.0)
    parser.add_argument("--c-anneal", action="store_true")
    parser.add_argument("--zero-weight", type=float, default=0.01)
    parser.add_argument("--med-weight", type=float, default=0.0)
    parser.add_argument("--bin-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--restart-from-best-every", type=int, default=0,
                        help="If >0, when no best for this many epochs, reload best INITs with small noise.")
    parser.add_argument("--restart-init-p", type=float, default=0.85)
    parser.add_argument("--restart-noise-std", type=float, default=0.05)
    parser.add_argument("--restart-lr-decay", type=float, default=0.70)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--bitflip-after", action="store_true")
    parser.add_argument("--bitflip-rounds", type=int, default=3)

    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--log-file", type=str, default="terminal_log.txt")
    parser.add_argument("--stage-name", type=str, default="train")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_file
    log_handle = install_tee(log_path)

    try:
        print(f"Log file: {log_path}")
        print(f"Stage: {args.stage_name}")
        print(f"Device: {device}")
        print(f"Args: {vars(args)}")

        base_inits = make_base_inits(args)
        base_metrics = compute_hard_metrics(base_inits)
        manual_metrics = compute_hard_metrics(default_inits())

        print("Base INIT metrics: "
              f"MRED={base_metrics.MRED:.10f}, MED={base_metrics.MED:.6f}, "
              f"ER={base_metrics.ER:.6f}, WCE={base_metrics.WCE}")
        print("Manual Verilog reference metrics: "
              f"MRED={manual_metrics.MRED:.10f}, MED={manual_metrics.MED:.6f}, "
              f"ER={manual_metrics.ER:.6f}, WCE={manual_metrics.WCE}")
        print("Base INITs:")
        for k in TRAINABLE_LUT_NAMES:
            print(f"  {k:7s} = {base_inits[k]}")

        model = Approx66STE(base_inits=base_inits, init_p=args.init_p, noise_std=args.noise_std).to(device)
        optimizer = build_optimizer(model, args.lr)
        a, b, exact = make_dataset(device)

        best_mred = base_metrics.MRED
        best = {
            "stage": args.stage_name,
            "epoch": -1,
            "loss": math.inf,
            "metrics": metrics_to_dict(base_metrics),
            "inits": base_inits,
        }
        write_best_files(out_dir, best, args.c_init, prefix="best")
        epochs_since_best = 0

        print("\nTraining begin")
        for epoch in range(args.epochs):
            if args.c_anneal:
                t = 0.0 if args.epochs <= 1 else epoch / (args.epochs - 1)
                c_init = 1.0 + t * (args.c_init - 1.0)
                c_out = 1.0 + t * (args.c_out - 1.0)
            else:
                c_init = args.c_init
                c_out = args.c_out

            model.train()
            optimizer.zero_grad(set_to_none=True)
            approx, _ = model(a, b, c_init=c_init, c_out=c_out, hard_middle=True)
            bin_reg = model.binary_regularization(c_init)
            loss, parts = loss_fn(
                approx,
                exact,
                bin_reg,
                zero_weight=args.zero_weight,
                med_weight=args.med_weight,
                bin_weight=args.bin_weight,
            )
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            model.eval()
            current_inits = model.hard_inits(c_init)
            hard_metrics = compute_hard_metrics(current_inits)

            improved = hard_metrics.MRED < best_mred
            if improved:
                best_mred = hard_metrics.MRED
                best = {
                    "stage": args.stage_name,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "metrics": metrics_to_dict(hard_metrics),
                    "inits": current_inits,
                }
                write_best_files(out_dir, best, c_init, prefix="best")
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            mark = " *BEST*" if improved else ""
            print(
                f"[epoch {epoch:05d}] "
                f"lr={current_lr(optimizer):.6g} "
                f"loss={float(loss.detach().cpu()):.8f} "
                f"train_mred={parts['mred_loss']:.8f} "
                f"zero={parts['zero_loss']:.6f} "
                f"med_norm={parts['med_norm']:.6f} "
                f"bin={parts['bin_reg']:.6f} "
                f"hard_MRED={hard_metrics.MRED:.8f} "
                f"MED={hard_metrics.MED:.4f} "
                f"ER={hard_metrics.ER:.4f} "
                f"WCE={hard_metrics.WCE} "
                f"best={best_mred:.8f}{mark}"
            )

            if args.restart_from_best_every > 0 and epochs_since_best >= args.restart_from_best_every:
                old_lr = current_lr(optimizer)
                new_lr = max(args.min_lr, old_lr * args.restart_lr_decay)
                print(
                    f"[restart] no new best for {epochs_since_best} epochs; "
                    f"reload best INITs, lr {old_lr:.6g} -> {new_lr:.6g}, "
                    f"restart_init_p={args.restart_init_p}, restart_noise_std={args.restart_noise_std}"
                )
                model.reset_from_inits(best["inits"], init_p=args.restart_init_p, noise_std=args.restart_noise_std)
                optimizer = build_optimizer(model, new_lr)
                epochs_since_best = 0

        if args.bitflip_after:
            flipped_inits, flipped_metrics = greedy_bitflip(best["inits"], max_rounds=args.bitflip_rounds)
            if flipped_metrics.MRED < best_mred:
                best_mred = flipped_metrics.MRED
                best = {
                    "stage": args.stage_name + "+bitflip",
                    "epoch": "bitflip",
                    "loss": best.get("loss", math.inf),
                    "metrics": metrics_to_dict(flipped_metrics),
                    "inits": flipped_inits,
                }
                write_best_files(out_dir, best, args.c_init, prefix="best")
                write_best_files(out_dir, best, args.c_init, prefix="bitflip_best")
                print("[bitflip] improved final best and wrote best files")
            else:
                print("[bitflip] no improvement over training best")

        write_best_files(out_dir, best, args.c_init, prefix="best")
        print("\nTraining finished.")
        print(f"Best epoch: {best['epoch']}")
        print(f"Best hard MRED: {best['metrics']['MRED']:.10f}")
        print(f"Best hard MED : {best['metrics']['MED']:.10f}")
        print(f"Best hard ER  : {best['metrics']['ER']:.10f}")
        print(f"Best hard WCE : {best['metrics']['WCE']}")
        print("Best INITs:")
        for k, v in best["inits"].items():
            print(f"  {k:7s} = {v}")
        print(f"Log saved to: {log_path}")
        print(f"Best JSON saved to: {out_dir / 'best_approx66_inits.json'}")
        print(f"Best Verilog snippet saved to: {out_dir / 'best_approx66_verilog_snippet.v'}")

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()
