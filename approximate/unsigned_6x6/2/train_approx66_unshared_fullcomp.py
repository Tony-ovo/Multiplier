#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STE-based LUT INIT search for 6x6 approximate multiplier, unshared/full-comp version.

Compared with train_approx66_ste_search.py:
  1) approx62 is no longer shared by LOW/MID/HIGH.
     - low_lut1..low_lut4
     - mid_lut1..mid_lut4
     - high_lut1..high_lut4
  2) comp66 fixed OR columns prod[4:7] are replaced by trainable LUT6:
     - u_or4, u_or5, u_or6, u_or7
     u_or23 and u_or89 remain trainable LUT6_2.
  3) Old JSON is compatible. If it contains shared lut1..lut4, these are expanded to low/mid/high.
  4) Terminal output is mirrored exactly to the txt log file.
  5) Bit-flip supports used-bit masks, random scan, and optional best-improvement search.

Typical usage:
  python3 train_approx66_unshared_fullcomp.py \
    --init-mode json --base-inits-json old_best/best_approx66_inits.json \
    --epochs 1000 --lr 0.001 --bitflip-after --bitflip-rounds 20 \
    --out-dir runs/unshared_fullcomp
"""

from __future__ import annotations

import argparse
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
# 1. INIT names and defaults
# ============================================================
DEFAULT_APPROX62 = {
    "lut1": "EAC00000A0A00000",
    "lut2": "EEAACC00EAC0EAC0",
    "lut3": "E6AACC006AC0EAC0",
    "lut4": "800000004C000000",
}

DEFAULT_COMP66 = {
    "u_or23": "0000FFF80000FEE6",
    "u_or4":  "00000000000000FE",
    "u_or5":  "00000000000000FE",
    "u_or6":  "00000000000000FE",
    "u_or7":  "00000000000000FE",
    "u_or89": "00005F5800005E4E",
}

SEGMENTS = ["low", "mid", "high"]
APPROX_LUT_IDS = ["lut1", "lut2", "lut3", "lut4"]
APPROX_LUT_NAMES = [f"{seg}_{lut}" for seg in SEGMENTS for lut in APPROX_LUT_IDS]
COMP_LUT6_2_NAMES = ["u_or23", "u_or89"]
COMP_LUT6_NAMES = ["u_or4", "u_or5", "u_or6", "u_or7"]
TRAINABLE_LUT_NAMES = APPROX_LUT_NAMES + COMP_LUT6_2_NAMES + COMP_LUT6_NAMES

# Only these bit positions can affect hardware because some LUT inputs are constants.
USED_BITS: Dict[str, List[int]] = {}
for seg in SEGMENTS:
    # lut1/lut4: O5 uses I4=1 => 16..31; O6 uses I4=1,I5=1 => 48..63.
    USED_BITS[f"{seg}_lut1"] = list(range(16, 32)) + list(range(48, 64))
    USED_BITS[f"{seg}_lut2"] = list(range(64))
    USED_BITS[f"{seg}_lut3"] = list(range(64))
    USED_BITS[f"{seg}_lut4"] = list(range(16, 32)) + list(range(48, 64))
# u_or23/u_or89: O5 uses I4=0 => 0..15; O6 uses I4=0,I5=1 => 32..47.
USED_BITS["u_or23"] = list(range(0, 16)) + list(range(32, 48))
USED_BITS["u_or89"] = list(range(0, 16)) + list(range(32, 48))
# LUT6 OR replacements: I3=I4=I5=0, so only 0..7 are addressable.
for name in COMP_LUT6_NAMES:
    USED_BITS[name] = list(range(8))


# ============================================================
# 2. Logging
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
# 3. Utilities
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


def sharp01(x: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)
    xc = torch.pow(x, c)
    yc = torch.pow(1.0 - x, c)
    return xc / (xc + yc + eps)


def ste_binarize(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    hard = (x >= threshold).to(x.dtype)
    return hard.detach() - x.detach() + x


def int_bits(x: torch.Tensor, width: int) -> List[torch.Tensor]:
    return [((x >> i) & 1).to(torch.float32) for i in range(width)]


def normalize_inits(obj: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize to the new unshared/full-comp key set.
    Compatible with old shared JSON containing lut1..lut4,u_or23,u_or89.
    """
    if "inits" in obj:
        obj = obj["inits"]

    out: Dict[str, str] = {}

    # approx62: use explicit unshared key if present; otherwise copy old shared lut1..lut4; otherwise default.
    for seg in SEGMENTS:
        for lut in APPROX_LUT_IDS:
            new_key = f"{seg}_{lut}"
            if new_key in obj:
                out[new_key] = bits_to_hex64(hex_to_bits64(obj[new_key]))
            elif lut in obj:
                out[new_key] = bits_to_hex64(hex_to_bits64(obj[lut]))
            else:
                out[new_key] = bits_to_hex64(hex_to_bits64(DEFAULT_APPROX62[lut]))

    # comp66
    for name in COMP_LUT6_2_NAMES + COMP_LUT6_NAMES:
        if name in obj:
            out[name] = bits_to_hex64(hex_to_bits64(obj[name]))
        else:
            out[name] = bits_to_hex64(hex_to_bits64(DEFAULT_COMP66[name]))

    return out


def default_inits() -> Dict[str, str]:
    return normalize_inits({})


def random_inits(prob_one: float) -> Dict[str, str]:
    prob_one = min(max(float(prob_one), 0.0), 1.0)
    out = {}
    for name in TRAINABLE_LUT_NAMES:
        bits = [1 if random.random() < prob_one else 0 for _ in range(64)]
        out[name] = bits_to_hex64(bits)
    return normalize_inits(out)


def load_inits_json(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return normalize_inits(obj)


# ============================================================
# 4. Differentiable LUTs
# ============================================================
class TrainableLUT6_2(nn.Module):
    def __init__(self, init_hex: str, init_p: float = 0.90, noise_std: float = 0.03):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_p=init_p, noise_std=noise_std)

    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_p: float = 0.90, noise_std: float = 0.03) -> None:
        bits = torch.tensor(hex_to_bits64(init_hex), dtype=torch.float32, device=self.logits.device)
        init_p = min(max(float(init_p), 1e-4), 1.0 - 1e-4)
        p = torch.where(bits > 0.5, torch.full_like(bits, init_p), torch.full_like(bits, 1.0 - init_p))
        logits = torch.log(p / (1.0 - p))
        if noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std
        self.logits.copy_(logits)

    def table_prob(self, c_init: float) -> torch.Tensor:
        return sharp01(torch.sigmoid(self.logits), c_init)

    @staticmethod
    def _soft_lut(inputs: List[torch.Tensor], table: torch.Tensor) -> torch.Tensor:
        n_addr = 1 << len(inputs)
        out = torch.zeros_like(inputs[0])
        for addr in range(n_addr):
            w = torch.ones_like(inputs[0])
            for i, xi in enumerate(inputs):
                w = w * (xi if ((addr >> i) & 1) else (1.0 - xi))
            out = out + w * table[addr]
        return out

    def forward(self, I0, I1, I2, I3, I4, I5, *, c_init: float, c_out: float, hard_middle: bool = True):
        table = self.table_prob(c_init)
        if hard_middle:
            I0, I1, I2, I3, I4, I5 = [ste_binarize(x) for x in [I0, I1, I2, I3, I4, I5]]
        o5 = self._soft_lut([I0, I1, I2, I3, I4], table[:32])
        o6 = self._soft_lut([I0, I1, I2, I3, I4, I5], table)
        o5 = sharp01(o5, c_out)
        o6 = sharp01(o6, c_out)
        if hard_middle:
            o5, o6 = ste_binarize(o5), ste_binarize(o6)
        return o5, o6

    def hard_bits(self, c_init: float) -> List[int]:
        with torch.no_grad():
            return (self.table_prob(c_init) >= 0.5).to(torch.int64).cpu().tolist()

    def hard_hex(self, c_init: float) -> str:
        return bits_to_hex64(self.hard_bits(c_init))

    def binary_regularization(self, c_init: float) -> torch.Tensor:
        p = self.table_prob(c_init)
        return torch.mean(p * (1.0 - p))


class TrainableLUT6(nn.Module):
    def __init__(self, init_hex: str, init_p: float = 0.90, noise_std: float = 0.03):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_p=init_p, noise_std=noise_std)

    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_p: float = 0.90, noise_std: float = 0.03) -> None:
        bits = torch.tensor(hex_to_bits64(init_hex), dtype=torch.float32, device=self.logits.device)
        init_p = min(max(float(init_p), 1e-4), 1.0 - 1e-4)
        p = torch.where(bits > 0.5, torch.full_like(bits, init_p), torch.full_like(bits, 1.0 - init_p))
        logits = torch.log(p / (1.0 - p))
        if noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std
        self.logits.copy_(logits)

    def table_prob(self, c_init: float) -> torch.Tensor:
        return sharp01(torch.sigmoid(self.logits), c_init)

    @staticmethod
    def _soft_lut(inputs: List[torch.Tensor], table: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(inputs[0])
        for addr in range(64):
            w = torch.ones_like(inputs[0])
            for i, xi in enumerate(inputs):
                w = w * (xi if ((addr >> i) & 1) else (1.0 - xi))
            out = out + w * table[addr]
        return out

    def forward(self, I0, I1, I2, I3, I4, I5, *, c_init: float, c_out: float, hard_middle: bool = True):
        table = self.table_prob(c_init)
        if hard_middle:
            I0, I1, I2, I3, I4, I5 = [ste_binarize(x) for x in [I0, I1, I2, I3, I4, I5]]
        o = self._soft_lut([I0, I1, I2, I3, I4, I5], table)
        o = sharp01(o, c_out)
        if hard_middle:
            o = ste_binarize(o)
        return o

    def hard_bits(self, c_init: float) -> List[int]:
        with torch.no_grad():
            return (self.table_prob(c_init) >= 0.5).to(torch.int64).cpu().tolist()

    def hard_hex(self, c_init: float) -> str:
        return bits_to_hex64(self.hard_bits(c_init))

    def binary_regularization(self, c_init: float) -> torch.Tensor:
        p = self.table_prob(c_init)
        return torch.mean(p * (1.0 - p))


class Approx66UnsharedFullComp(nn.Module):
    def __init__(self, base_inits: Dict[str, str], init_p: float = 0.90, noise_std: float = 0.03):
        super().__init__()
        base_inits = normalize_inits(base_inits)
        self.lut6_2 = nn.ModuleDict({
            name: TrainableLUT6_2(base_inits[name], init_p, noise_std)
            for name in APPROX_LUT_NAMES + COMP_LUT6_2_NAMES
        })
        self.lut6 = nn.ModuleDict({
            name: TrainableLUT6(base_inits[name], init_p, noise_std)
            for name in COMP_LUT6_NAMES
        })

    @torch.no_grad()
    def reset_from_inits(self, base_inits: Dict[str, str], init_p: float, noise_std: float) -> None:
        base_inits = normalize_inits(base_inits)
        for name in APPROX_LUT_NAMES + COMP_LUT6_2_NAMES:
            self.lut6_2[name].reset_from_hex(base_inits[name], init_p=init_p, noise_std=noise_std)
        for name in COMP_LUT6_NAMES:
            self.lut6[name].reset_from_hex(base_inits[name], init_p=init_p, noise_std=noise_std)

    @staticmethod
    def const_like(x: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full_like(x, float(value))

    def approx62_segment(self, seg: str, a: torch.Tensor, b2: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool) -> List[torch.Tensor]:
        a_bits = int_bits(a, 6)
        b_bits = int_bits(b2, 2)
        one = self.const_like(a_bits[0], 1.0)
        p0, p1 = self.lut6_2[f"{seg}_lut1"](b_bits[0], b_bits[1], a_bits[0], a_bits[1], one, one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p2, p3 = self.lut6_2[f"{seg}_lut2"](b_bits[0], b_bits[1], a_bits[1], a_bits[2], a_bits[3], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p4, p5 = self.lut6_2[f"{seg}_lut3"](b_bits[0], b_bits[1], a_bits[3], a_bits[4], a_bits[5], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p6, p7 = self.lut6_2[f"{seg}_lut4"](b_bits[0], b_bits[1], a_bits[4], a_bits[5], one, one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return [p0, p1, p2, p3, p4, p5, p6, p7]

    def comp66(self, plow, pmid, phigh, *, c_init: float, c_out: float, hard_middle: bool) -> List[torch.Tensor]:
        z = self.const_like(plow[0], 0.0)
        o = self.const_like(plow[0], 1.0)
        prod: List[torch.Tensor] = [z for _ in range(12)]
        prod[0] = plow[0]
        prod[1] = plow[1]
        prod[10] = phigh[6]
        prod[11] = phigh[7]

        prod[2], prod[3] = self.lut6_2["u_or23"](plow[2], pmid[0], plow[3], pmid[1], z, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[4] = self.lut6["u_or4"](plow[4], pmid[2], phigh[0], z, z, z, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[5] = self.lut6["u_or5"](plow[5], pmid[3], phigh[1], z, z, z, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[6] = self.lut6["u_or6"](plow[6], pmid[4], phigh[2], z, z, z, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[7] = self.lut6["u_or7"](plow[7], pmid[5], phigh[3], z, z, z, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[8], prod[9] = self.lut6_2["u_or89"](pmid[6], phigh[4], pmid[7], phigh[5], z, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return prod

    def forward(self, a: torch.Tensor, b: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool = True):
        plow = self.approx62_segment("low", a, b & 0b11, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        pmid = self.approx62_segment("mid", a, (b >> 2) & 0b11, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        phigh = self.approx62_segment("high", a, (b >> 4) & 0b11, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod_bits = self.comp66(plow, pmid, phigh, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod_stack = torch.stack(prod_bits, dim=1)
        weights = torch.tensor([1 << i for i in range(12)], dtype=torch.float32, device=prod_stack.device)
        approx = torch.sum(prod_stack * weights, dim=1)
        return approx, prod_stack

    def hard_inits(self, c_init: float) -> Dict[str, str]:
        out = {}
        for name in APPROX_LUT_NAMES + COMP_LUT6_2_NAMES:
            out[name] = self.lut6_2[name].hard_hex(c_init)
        for name in COMP_LUT6_NAMES:
            out[name] = self.lut6[name].hard_hex(c_init)
        return normalize_inits(out)

    def binary_regularization(self, c_init: float) -> torch.Tensor:
        regs = []
        for name in APPROX_LUT_NAMES + COMP_LUT6_2_NAMES:
            regs.append(self.lut6_2[name].binary_regularization(c_init))
        for name in COMP_LUT6_NAMES:
            regs.append(self.lut6[name].binary_regularization(c_init))
        return torch.mean(torch.stack(regs))


# ============================================================
# 5. Verilog-equivalent hard simulator
# ============================================================
def lut6_2_hard(bits: List[int], I0: int, I1: int, I2: int, I3: int, I4: int, I5: int) -> Tuple[int, int]:
    addr5 = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4)
    addr6 = addr5 + (I5 << 5)
    return bits[addr5], bits[addr6]


def lut6_hard(bits: List[int], I0: int, I1: int, I2: int, I3: int, I4: int, I5: int) -> int:
    addr = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4) + (I5 << 5)
    return bits[addr]


def approx62_hard_segment(a: int, b2: int, seg: str, init_bits: Dict[str, List[int]]) -> List[int]:
    ab = [(a >> i) & 1 for i in range(6)]
    bb = [(b2 >> i) & 1 for i in range(2)]
    p0, p1 = lut6_2_hard(init_bits[f"{seg}_lut1"], bb[0], bb[1], ab[0], ab[1], 1, 1)
    p2, p3 = lut6_2_hard(init_bits[f"{seg}_lut2"], bb[0], bb[1], ab[1], ab[2], ab[3], 1)
    p4, p5 = lut6_2_hard(init_bits[f"{seg}_lut3"], bb[0], bb[1], ab[3], ab[4], ab[5], 1)
    p6, p7 = lut6_2_hard(init_bits[f"{seg}_lut4"], bb[0], bb[1], ab[4], ab[5], 1, 1)
    return [p0, p1, p2, p3, p4, p5, p6, p7]


def comp66_hard(plow: List[int], pmid: List[int], phigh: List[int], init_bits: Dict[str, List[int]]) -> int:
    prod = [0] * 12
    prod[0] = plow[0]
    prod[1] = plow[1]
    prod[10] = phigh[6]
    prod[11] = phigh[7]
    prod[2], prod[3] = lut6_2_hard(init_bits["u_or23"], plow[2], pmid[0], plow[3], pmid[1], 0, 1)
    prod[4] = lut6_hard(init_bits["u_or4"], plow[4], pmid[2], phigh[0], 0, 0, 0)
    prod[5] = lut6_hard(init_bits["u_or5"], plow[5], pmid[3], phigh[1], 0, 0, 0)
    prod[6] = lut6_hard(init_bits["u_or6"], plow[6], pmid[4], phigh[2], 0, 0, 0)
    prod[7] = lut6_hard(init_bits["u_or7"], plow[7], pmid[5], phigh[3], 0, 0, 0)
    prod[8], prod[9] = lut6_2_hard(init_bits["u_or89"], pmid[6], phigh[4], pmid[7], phigh[5], 0, 1)
    value = 0
    for i, bit in enumerate(prod):
        value |= (int(bit) << i)
    return value


def approx66_hard(a: int, b: int, init_bits: Dict[str, List[int]]) -> int:
    plow = approx62_hard_segment(a, b & 0b11, "low", init_bits)
    pmid = approx62_hard_segment(a, (b >> 2) & 0b11, "mid", init_bits)
    phigh = approx62_hard_segment(a, (b >> 4) & 0b11, "high", init_bits)
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


def make_dataset(device: torch.device):
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


def loss_fn(approx, exact, bin_reg, *, zero_weight: float, med_weight: float, bin_weight: float):
    mask = exact > 0
    abs_err = torch.abs(approx - exact)
    mred = torch.sum(abs_err[mask] / exact[mask]) / float(exact.numel())
    zero_loss = torch.mean(abs_err[~mask]) / 4096.0 if torch.any(~mask) else torch.tensor(0.0, device=approx.device)
    med_norm = torch.mean(abs_err) / 4096.0
    loss = mred + zero_weight * zero_loss + med_weight * med_norm + bin_weight * bin_reg
    return loss, {
        "mred_loss": float(mred.detach().cpu()),
        "zero_loss": float(zero_loss.detach().cpu()),
        "med_norm": float(med_norm.detach().cpu()),
        "bin_reg": float(bin_reg.detach().cpu()),
    }


def metrics_to_dict(m: Metrics) -> Dict:
    return {"total_cases": m.total_cases, "error_cases": m.error_cases, "ER": m.ER, "MED": m.MED, "NED": m.NED, "MRED": m.MRED, "WCE": m.WCE}


# ============================================================
# 6. Output files
# ============================================================
def write_best_files(out_dir: Path, best: Dict, c_init: float, prefix: str = "best") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}_approx66_inits.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    if prefix == "best":
        conventional_json = out_dir / "best_approx66_inits.json"
        if conventional_json != json_path:
            conventional_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    inits = normalize_inits(best["inits"])
    snippet_path = out_dir / f"{prefix}_approx66_unshared_fullcomp.v"
    with snippet_path.open("w", encoding="utf-8") as f:
        f.write("// Full Verilog template for unshared approx62 + trainable full comp66.\n")
        f.write("// Generated by train_approx66_unshared_fullcomp.py\n")
        f.write(f"// stage = {best.get('stage', '')}\n")
        f.write(f"// epoch = {best['epoch']}\n")
        f.write(f"// MRED  = {best['metrics']['MRED']:.10f}\n")
        f.write(f"// MED   = {best['metrics']['MED']:.10f}\n")
        f.write(f"// ER    = {best['metrics']['ER']:.10f}\n")
        f.write(f"// WCE   = {best['metrics']['WCE']}\n")
        f.write(f"// c_init threshold = {c_init}\n\n")
        f.write("module approx66_unshared_fullcomp (\n")
        f.write("    input wire [5:0] a,\n    input wire [5:0] b,\n    output wire [11:0] prod\n);\n")
        f.write("wire [7:0] plow, pmid, phigh;\n")
        f.write("approx62_low  U_LOW  (.a(a), .b(b[1:0]), .prod(plow));\n")
        f.write("approx62_mid  U_MID  (.a(a), .b(b[3:2]), .prod(pmid));\n")
        f.write("approx62_high U_HIGH (.a(a), .b(b[5:4]), .prod(phigh));\n")
        f.write("comp66_full U_COMP (.plow(plow), .pmid(pmid), .phigh(phigh), .prod(prod));\n")
        f.write("endmodule\n\n")

        for seg in SEGMENTS:
            mod = f"approx62_{seg}"
            f.write(f"module {mod} (input wire [5:0] a, input wire [1:0] b, output wire [7:0] prod);\n")
            key = f"{seg}_lut1"
            f.write(f"LUT6_2 #(.INIT({inits[key]})) LUT6_inst1 (.I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]), .I4(1'b1), .I5(1'b1), .O5(prod[0]), .O6(prod[1]));\n")
            key = f"{seg}_lut2"
            f.write(f"LUT6_2 #(.INIT({inits[key]})) LUT6_inst2 (.I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]), .I4(a[3]), .I5(1'b1), .O5(prod[2]), .O6(prod[3]));\n")
            key = f"{seg}_lut3"
            f.write(f"LUT6_2 #(.INIT({inits[key]})) LUT6_inst3 (.I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]), .I4(a[5]), .I5(1'b1), .O5(prod[4]), .O6(prod[5]));\n")
            key = f"{seg}_lut4"
            f.write(f"LUT6_2 #(.INIT({inits[key]})) LUT6_inst4 (.I0(b[0]), .I1(b[1]), .I2(a[4]), .I3(a[5]), .I4(1'b1), .I5(1'b1), .O5(prod[6]), .O6(prod[7]));\n")
            f.write("endmodule\n\n")

        f.write("module comp66_full (input wire [7:0] plow, input wire [7:0] pmid, input wire [7:0] phigh, output wire [11:0] prod);\n")
        f.write("assign prod[0] = plow[0];\nassign prod[1] = plow[1];\nassign prod[10] = phigh[6];\nassign prod[11] = phigh[7];\n")
        f.write(f"LUT6_2 #(.INIT({inits['u_or23']})) u_or23 (.I0(plow[2]), .I1(pmid[0]), .I2(plow[3]), .I3(pmid[1]), .I4(1'b0), .I5(1'b1), .O5(prod[2]), .O6(prod[3]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_or4']})) u_or4 (.I0(plow[4]), .I1(pmid[2]), .I2(phigh[0]), .I3(1'b0), .I4(1'b0), .I5(1'b0), .O(prod[4]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_or5']})) u_or5 (.I0(plow[5]), .I1(pmid[3]), .I2(phigh[1]), .I3(1'b0), .I4(1'b0), .I5(1'b0), .O(prod[5]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_or6']})) u_or6 (.I0(plow[6]), .I1(pmid[4]), .I2(phigh[2]), .I3(1'b0), .I4(1'b0), .I5(1'b0), .O(prod[6]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_or7']})) u_or7 (.I0(plow[7]), .I1(pmid[5]), .I2(phigh[3]), .I3(1'b0), .I4(1'b0), .I5(1'b0), .O(prod[7]));\n")
        f.write(f"LUT6_2 #(.INIT({inits['u_or89']})) u_or89 (.I0(pmid[6]), .I1(phigh[4]), .I2(pmid[7]), .I3(phigh[5]), .I4(1'b0), .I5(1'b1), .O5(prod[8]), .O6(prod[9]));\n")
        f.write("endmodule\n")

    if prefix == "best":
        conventional = out_dir / "best_approx66_unshared_fullcomp.v"
        if conventional != snippet_path:
            conventional.write_text(snippet_path.read_text(encoding="utf-8"), encoding="utf-8")


# ============================================================
# 7. Bit-flip search
# ============================================================
def bit_pairs(names: List[str], random_order: bool) -> List[Tuple[str, int]]:
    pairs = [(name, bit) for name in names for bit in USED_BITS[name]]
    if random_order:
        random.shuffle(pairs)
    return pairs


def greedy_bitflip(start_inits: Dict[str, str], *, max_rounds: int = 20, eps: float = 1e-12, random_order: bool = False, mode: str = "first"):
    current = normalize_inits(start_inits)
    current_metrics = compute_hard_metrics(current)
    print("\n[bitflip] start "
          f"MRED={current_metrics.MRED:.10f} MED={current_metrics.MED:.4f} "
          f"ER={current_metrics.ER:.4f} WCE={current_metrics.WCE} mode={mode} random_order={random_order}")

    names = TRAINABLE_LUT_NAMES
    for r in range(max_rounds):
        improved_this_round = False
        print(f"[bitflip] round {r + 1}/{max_rounds} begin")

        if mode == "best":
            best_trial = None
            best_trial_metrics = current_metrics
            best_old_mred = current_metrics.MRED
            for name, bit_idx in bit_pairs(names, random_order):
                bits = hex_to_bits64(current[name])
                bits[bit_idx] ^= 1
                trial = dict(current)
                trial[name] = bits_to_hex64(bits)
                trial_metrics = compute_hard_metrics(trial)
                if trial_metrics.MRED + eps < best_trial_metrics.MRED:
                    best_trial = (name, bit_idx, trial)
                    best_trial_metrics = trial_metrics
            if best_trial is not None:
                name, bit_idx, trial = best_trial
                current = trial
                current_metrics = best_trial_metrics
                improved_this_round = True
                print(f"[bitflip] KEEP-BEST round={r + 1} lut={name} bit={bit_idx:02d} "
                      f"MRED {best_old_mred:.10f} -> {current_metrics.MRED:.10f} "
                      f"MED={current_metrics.MED:.4f} ER={current_metrics.ER:.4f} WCE={current_metrics.WCE}")
        else:
            for name, bit_idx in bit_pairs(names, random_order):
                bits = hex_to_bits64(current[name])
                trial_bits = bits.copy()
                trial_bits[bit_idx] ^= 1
                trial = dict(current)
                trial[name] = bits_to_hex64(trial_bits)
                trial_metrics = compute_hard_metrics(trial)
                if trial_metrics.MRED + eps < current_metrics.MRED:
                    old = current_metrics.MRED
                    current = trial
                    current_metrics = trial_metrics
                    improved_this_round = True
                    print(f"[bitflip] KEEP round={r + 1} lut={name} bit={bit_idx:02d} "
                          f"MRED {old:.10f} -> {current_metrics.MRED:.10f} "
                          f"MED={current_metrics.MED:.4f} ER={current_metrics.ER:.4f} WCE={current_metrics.WCE}")
        if not improved_this_round:
            print(f"[bitflip] round {r + 1} no improvement, stop")
            break
    print("[bitflip] final "
          f"MRED={current_metrics.MRED:.10f} MED={current_metrics.MED:.4f} "
          f"ER={current_metrics.ER:.4f} WCE={current_metrics.WCE}")
    return current, current_metrics


# ============================================================
# 8. Main training
# ============================================================
def make_base_inits(args) -> Dict[str, str]:
    if args.init_mode == "manual":
        return default_inits()
    if args.init_mode == "random":
        return random_inits(args.random_init_prob)
    if args.init_mode == "json":
        if not args.base_inits_json:
            raise ValueError("--init-mode json requires --base-inits-json")
        return load_inits_json(Path(args.base_inits_json))
    raise ValueError(args.init_mode)


def build_optimizer(model: nn.Module, lr: float):
    return torch.optim.Adam(model.parameters(), lr=lr)


def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--init-mode", type=str, default="json", choices=["manual", "random", "json"])
    parser.add_argument("--base-inits-json", type=str, default="")
    parser.add_argument("--random-init-prob", type=float, default=0.5)
    parser.add_argument("--init-p", type=float, default=0.90)
    parser.add_argument("--noise-std", type=float, default=0.03)

    parser.add_argument("--c-init", type=float, default=2.0)
    parser.add_argument("--c-out", type=float, default=2.0)
    parser.add_argument("--c-anneal", action="store_true")
    parser.add_argument("--zero-weight", type=float, default=0.01)
    parser.add_argument("--med-weight", type=float, default=0.0)
    parser.add_argument("--bin-weight", type=float, default=2e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--restart-from-best-every", type=int, default=150)
    parser.add_argument("--restart-init-p", type=float, default=0.92)
    parser.add_argument("--restart-noise-std", type=float, default=0.02)
    parser.add_argument("--restart-lr-decay", type=float, default=0.85)
    parser.add_argument("--min-lr", type=float, default=1e-5)

    parser.add_argument("--bitflip-after", action="store_true")
    parser.add_argument("--bitflip-rounds", type=int, default=20)
    parser.add_argument("--bitflip-random-order", action="store_true")
    parser.add_argument("--bitflip-mode", choices=["first", "best"], default="first")
    parser.add_argument("--bitflip-only", action="store_true")

    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--log-file", type=str, default="terminal_log.txt")
    parser.add_argument("--stage-name", type=str, default="unshared_fullcomp")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
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
              f"MRED={base_metrics.MRED:.10f}, MED={base_metrics.MED:.6f}, ER={base_metrics.ER:.6f}, WCE={base_metrics.WCE}")
        print("Manual expanded reference metrics: "
              f"MRED={manual_metrics.MRED:.10f}, MED={manual_metrics.MED:.6f}, ER={manual_metrics.ER:.6f}, WCE={manual_metrics.WCE}")
        print("Trainable LUT count:", len(TRAINABLE_LUT_NAMES), "used bit count:", sum(len(USED_BITS[n]) for n in TRAINABLE_LUT_NAMES))
        print("Base INITs:")
        for k in TRAINABLE_LUT_NAMES:
            print(f"  {k:10s} = {base_inits[k]}")

        best_mred = base_metrics.MRED
        best = {"stage": args.stage_name, "epoch": -1, "loss": math.inf, "metrics": metrics_to_dict(base_metrics), "inits": base_inits}
        write_best_files(out_dir, best, args.c_init, prefix="best")

        if args.bitflip_only:
            flipped_inits, flipped_metrics = greedy_bitflip(best["inits"], max_rounds=args.bitflip_rounds, random_order=args.bitflip_random_order, mode=args.bitflip_mode)
            best = {"stage": args.stage_name + "+bitflip_only", "epoch": "bitflip_only", "loss": math.inf, "metrics": metrics_to_dict(flipped_metrics), "inits": flipped_inits}
            write_best_files(out_dir, best, args.c_init, prefix="best")
            print("[bitflip-only] wrote best files")
            return

        model = Approx66UnsharedFullComp(base_inits=base_inits, init_p=args.init_p, noise_std=args.noise_std).to(device)
        optimizer = build_optimizer(model, args.lr)
        a, b, exact = make_dataset(device)
        epochs_since_best = 0

        print("\nTraining begin")
        for epoch in range(args.epochs):
            if args.c_anneal:
                t = 0.0 if args.epochs <= 1 else epoch / (args.epochs - 1)
                c_init = 1.0 + t * (args.c_init - 1.0)
                c_out = 1.0 + t * (args.c_out - 1.0)
            else:
                c_init, c_out = args.c_init, args.c_out

            model.train()
            optimizer.zero_grad(set_to_none=True)
            approx, _ = model(a, b, c_init=c_init, c_out=c_out, hard_middle=True)
            bin_reg = model.binary_regularization(c_init)
            loss, parts = loss_fn(approx, exact, bin_reg, zero_weight=args.zero_weight, med_weight=args.med_weight, bin_weight=args.bin_weight)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            current_inits = model.hard_inits(c_init)
            hard_metrics = compute_hard_metrics(current_inits)
            improved = hard_metrics.MRED < best_mred
            if improved:
                best_mred = hard_metrics.MRED
                best = {"stage": args.stage_name, "epoch": epoch, "loss": float(loss.detach().cpu()), "metrics": metrics_to_dict(hard_metrics), "inits": current_inits}
                write_best_files(out_dir, best, c_init, prefix="best")
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            mark = " *BEST*" if improved else ""
            print(f"[epoch {epoch:05d}] lr={current_lr(optimizer):.6g} loss={float(loss.detach().cpu()):.8f} "
                  f"train_mred={parts['mred_loss']:.8f} zero={parts['zero_loss']:.6f} med_norm={parts['med_norm']:.6f} "
                  f"bin={parts['bin_reg']:.6f} hard_MRED={hard_metrics.MRED:.8f} MED={hard_metrics.MED:.4f} "
                  f"ER={hard_metrics.ER:.4f} WCE={hard_metrics.WCE} best={best_mred:.8f}{mark}")

            if args.restart_from_best_every > 0 and epochs_since_best >= args.restart_from_best_every:
                old_lr = current_lr(optimizer)
                new_lr = max(args.min_lr, old_lr * args.restart_lr_decay)
                print(f"[restart] no new best for {epochs_since_best} epochs; reload best INITs, "
                      f"lr {old_lr:.6g} -> {new_lr:.6g}, init_p={args.restart_init_p}, noise={args.restart_noise_std}")
                model.reset_from_inits(best["inits"], init_p=args.restart_init_p, noise_std=args.restart_noise_std)
                optimizer = build_optimizer(model, new_lr)
                epochs_since_best = 0

        if args.bitflip_after:
            flipped_inits, flipped_metrics = greedy_bitflip(best["inits"], max_rounds=args.bitflip_rounds, random_order=args.bitflip_random_order, mode=args.bitflip_mode)
            if flipped_metrics.MRED < best_mred:
                best_mred = flipped_metrics.MRED
                best = {"stage": args.stage_name + "+bitflip", "epoch": "bitflip", "loss": best.get("loss", math.inf), "metrics": metrics_to_dict(flipped_metrics), "inits": flipped_inits}
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
        for k in TRAINABLE_LUT_NAMES:
            print(f"  {k:10s} = {best['inits'][k]}")
        print(f"Log saved to: {log_path}")
        print(f"Best JSON saved to: {out_dir / 'best_approx66_inits.json'}")
        print(f"Best Verilog saved to: {out_dir / 'best_approx66_unshared_fullcomp.v'}")
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        log_handle.flush(); log_handle.close()


if __name__ == "__main__":
    main()
