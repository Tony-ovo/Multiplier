#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-by-step INIT search for approx88 with a trainable LUT cascade.

Architecture being optimized:
  approx88(a,b) = comp88_lut(LL(al,bl), HL(bl,ah), LH(al,bh), HH(ah,bh))

where:
  al=a[5:0], ah=a[7:6], bl=b[5:0], bh=b[7:6]

What is trainable:
  - LL: the optimized approx66 low block, using the unshared + pair-aware comp66 structure.
  - HL: a 6x2 approx62-like block, initialized from the existing approx62_opt INIT by default.
  - LH: another independent 6x2 approx62-like block, also initialized from approx62_opt by default.
  - comp88_lut: the top 8x8 compressor LUTs u88_gp0..u88_gp8.

What is NOT optimized:
  - hh = ah*bh remains exact.
  - CARRY4-style carry propagation remains structural logic; its driving LUT INITs are trainable.

This script is intended for step-by-step experiments. The sibling shell scripts provide a pipeline.
It supports:
  --eval-only
  normal STE training with --train-scope low|cross|top|cross_top|low_top|all
  --single-only / --pair-only / --escape-only constrained discrete search

MRED definition default matches tb_88.v style:
  RED is accumulated for exact!=0, then divided by TOTAL=65536.
Use --mred-denom nonzero to divide by 65025 instead.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Names / defaults
# ============================================================
DEFAULT_SHARED = {
    "lut1": "EAC00000A0A00000",
    "lut2": "EEAACC00EAC0EAC0",
    "lut3": "E6AACC006AC0EAC0",
    "lut4": "800000004C000000",
    "u_or23": "0000FFF80000FEE6",
    "u_or89": "00005F5800005E4E",
}
DEFAULT_COMP_OLD = {
    "u_or4": "00000000000000FE",
    "u_or5": "00000000000000FE",
    "u_or6": "00000000000000FE",
    "u_or7": "00000000000000FE",
}
SEGMENTS = ["low", "mid", "high"]
APPROX_IDS = ["lut1", "lut2", "lut3", "lut4"]
LOW66_NAMES = [f"{s}_{l}" for s in SEGMENTS for l in APPROX_IDS] + [
    "u_comp23", "u_comp89", "u_comp4", "u_comp5", "u_comp6", "u_comp7"
]
CROSS_NAMES = [f"hl_{l}" for l in APPROX_IDS] + [f"lh_{l}" for l in APPROX_IDS]
TOP88_NAMES = [f"u88_gp{i}" for i in range(9)]
TRAIN_NAMES = LOW66_NAMES + CROSS_NAMES + TOP88_NAMES
DEFAULT_TOP88 = {
    "u88_gp0": "96969696E8E8E8E8",
    **{f"u88_gp{i}": "69966996E8E8E8E8" for i in range(1, 8)},
    "u88_gp8": "96669666A000A000",
}

USED_BITS: Dict[str, List[int]] = {}
for seg in SEGMENTS:
    USED_BITS[f"{seg}_lut1"] = list(range(16, 32)) + list(range(48, 64))
    USED_BITS[f"{seg}_lut2"] = list(range(64))
    USED_BITS[f"{seg}_lut3"] = list(range(64))
    USED_BITS[f"{seg}_lut4"] = list(range(16, 32)) + list(range(48, 64))
USED_BITS["u_comp23"] = list(range(0, 16)) + list(range(32, 48))
USED_BITS["u_comp89"] = list(range(0, 16)) + list(range(32, 48))
for n in ["u_comp4", "u_comp5", "u_comp6", "u_comp7"]:
    USED_BITS[n] = list(range(64))
for prefix in ["hl", "lh"]:
    USED_BITS[f"{prefix}_lut1"] = list(range(16, 32)) + list(range(48, 64))
    USED_BITS[f"{prefix}_lut2"] = list(range(64))
    USED_BITS[f"{prefix}_lut3"] = list(range(64))
    USED_BITS[f"{prefix}_lut4"] = list(range(16, 32)) + list(range(48, 64))
for n in TOP88_NAMES:
    USED_BITS[n] = list(range(16, 32)) + list(range(48, 64))

# ============================================================
# Logging / utilities
# ============================================================
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
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = sys.stdout
    return f

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_hex(s: str) -> str:
    s = str(s).strip().replace("64'h", "").replace("64'H", "").replace("0x", "").replace("0X", "").replace("_", "")
    if len(s) > 16:
        s = s[-16:]
    return s.zfill(16).upper()

def hex_int(s: str) -> int:
    return int(clean_hex(s), 16)

def int_to_hex(v: int) -> str:
    return f"64'h{int(v) & ((1 << 64) - 1):016X}"

def int_to_bits(v: int) -> List[int]:
    return [(int(v) >> i) & 1 for i in range(64)]

def bits_to_int(bits: Iterable[int]) -> int:
    v = 0
    for i, b in enumerate(bits):
        if int(b):
            v |= (1 << i)
    return v

def bits_to_hex(bits: Iterable[int]) -> str:
    return int_to_hex(bits_to_int(bits))

def sharp01(x: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)
    xc = torch.pow(x, c)
    yc = torch.pow(1.0 - x, c)
    return xc / (xc + yc + eps)

def ste_binarize(x: torch.Tensor) -> torch.Tensor:
    h = (x >= 0.5).to(x.dtype)
    return h.detach() - x.detach() + x

def int_bits_t(x: torch.Tensor, w: int) -> List[torch.Tensor]:
    return [((x >> i) & 1).to(torch.float32) for i in range(w)]

# ============================================================
# INIT projection / normalization
# ============================================================
def approx62_addr(lut_id: str, a: int, b2: int, o6: bool) -> int:
    b0 = b2 & 1
    b1 = (b2 >> 1) & 1
    ab = [(a >> i) & 1 for i in range(6)]
    if lut_id == "lut1":
        addr5 = b0 + (b1 << 1) + (ab[0] << 2) + (ab[1] << 3) + (1 << 4)
    elif lut_id == "lut2":
        addr5 = b0 + (b1 << 1) + (ab[1] << 2) + (ab[2] << 3) + (ab[3] << 4)
    elif lut_id == "lut3":
        addr5 = b0 + (b1 << 1) + (ab[3] << 2) + (ab[4] << 3) + (ab[5] << 4)
    elif lut_id == "lut4":
        addr5 = b0 + (b1 << 1) + (ab[4] << 2) + (ab[5] << 3) + (1 << 4)
    else:
        raise ValueError(lut_id)
    return addr5 + (32 if o6 else 0)

def projected_accurate62_inits() -> Dict[str, str]:
    """Best-effort projection of exact 6x2 truth table onto the existing approx62 local-window LUT structure.

    Exact 6x2 is not fully representable by this four-LUT local-window structure because some output bits
    depend on lower carry information not present in the LUT inputs. For each reachable INIT address, this
    chooses the majority exact output bit over all (a,b2) cases that hit that address.
    """
    out: Dict[str, List[int]] = {lut: [0] * 64 for lut in APPROX_IDS}
    cnt1: Dict[Tuple[str, int], int] = {}
    cnt0: Dict[Tuple[str, int], int] = {}
    bit_map = {
        "lut1": (0, 1),
        "lut2": (2, 3),
        "lut3": (4, 5),
        "lut4": (6, 7),
    }
    for a in range(64):
        for b2 in range(4):
            prod = a * b2
            for lut, (bit_o5, bit_o6) in bit_map.items():
                a5 = approx62_addr(lut, a, b2, False)
                a6 = approx62_addr(lut, a, b2, True)
                for addr, bit in [(a5, bit_o5), (a6, bit_o6)]:
                    y = (prod >> bit) & 1
                    key = (lut, addr)
                    if y:
                        cnt1[key] = cnt1.get(key, 0) + 1
                    else:
                        cnt0[key] = cnt0.get(key, 0) + 1
    for lut in APPROX_IDS:
        for addr in range(64):
            c1 = cnt1.get((lut, addr), 0)
            c0 = cnt0.get((lut, addr), 0)
            out[lut][addr] = 1 if c1 >= c0 and (c1 + c0) > 0 else 0
    return {lut: bits_to_hex(bits) for lut, bits in out.items()}

PROJECTED62 = projected_accurate62_inits()

def lut6_table_int_from_old_lut3(old_hex: str, input_map: Tuple[int, int, int]) -> int:
    old = int_to_bits(hex_int(old_hex))
    out = [0] * 64
    for addr in range(64):
        old_addr = 0
        for old_i, new_pos in enumerate(input_map):
            old_addr |= (((addr >> new_pos) & 1) << old_i)
        out[addr] = old[old_addr]
    return bits_to_int(out)

def normalize_inits(obj: Dict, cross_init_mode: str = "approx62") -> Dict[str, str]:
    if "inits" in obj:
        obj = obj["inits"]
    out: Dict[str, str] = {}
    # Low 6x6 pair-aware approx66.
    for seg in SEGMENTS:
        for lut in APPROX_IDS:
            nk = f"{seg}_{lut}"
            if nk in obj:
                out[nk] = int_to_hex(hex_int(obj[nk]))
            elif lut in obj:
                out[nk] = int_to_hex(hex_int(obj[lut]))
            else:
                out[nk] = int_to_hex(hex_int(DEFAULT_SHARED[lut]))
    if "u_comp23" in obj:
        out["u_comp23"] = int_to_hex(hex_int(obj["u_comp23"]))
    elif "u_or23" in obj:
        out["u_comp23"] = int_to_hex(hex_int(obj["u_or23"]))
    else:
        out["u_comp23"] = int_to_hex(hex_int(DEFAULT_SHARED["u_or23"]))
    if "u_comp89" in obj:
        out["u_comp89"] = int_to_hex(hex_int(obj["u_comp89"]))
    elif "u_or89" in obj:
        out["u_comp89"] = int_to_hex(hex_int(obj["u_or89"]))
    else:
        out["u_comp89"] = int_to_hex(hex_int(DEFAULT_SHARED["u_or89"]))
    if "u_comp4" in obj:
        out["u_comp4"] = int_to_hex(hex_int(obj["u_comp4"]))
    else:
        out["u_comp4"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or4", DEFAULT_COMP_OLD["u_or4"]), (0, 1, 2)))
    if "u_comp5" in obj:
        out["u_comp5"] = int_to_hex(hex_int(obj["u_comp5"]))
    else:
        out["u_comp5"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or5", DEFAULT_COMP_OLD["u_or5"]), (3, 4, 5)))
    if "u_comp6" in obj:
        out["u_comp6"] = int_to_hex(hex_int(obj["u_comp6"]))
    else:
        out["u_comp6"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or6", DEFAULT_COMP_OLD["u_or6"]), (0, 1, 2)))
    if "u_comp7" in obj:
        out["u_comp7"] = int_to_hex(hex_int(obj["u_comp7"]))
    else:
        out["u_comp7"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or7", DEFAULT_COMP_OLD["u_or7"]), (3, 4, 5)))

    # Cross 6x2 HL/LH. If absent, initialize conservatively.
    if cross_init_mode == "projected":
        base62 = PROJECTED62
    elif cross_init_mode == "approx62":
        base62 = {lut: DEFAULT_SHARED[lut] for lut in APPROX_IDS}
    elif cross_init_mode == "zero":
        base62 = {lut: "0000000000000000" for lut in APPROX_IDS}
    else:
        raise ValueError(cross_init_mode)
    for prefix in ["hl", "lh"]:
        for lut in APPROX_IDS:
            nk = f"{prefix}_{lut}"
            if nk in obj:
                out[nk] = int_to_hex(hex_int(obj[nk]))
            else:
                out[nk] = int_to_hex(hex_int(base62[lut]))
    # Top 8x8 compressor LUTs. If absent, start from the exact comp88 GP/CARRY implementation.
    for n in TOP88_NAMES:
        out[n] = int_to_hex(hex_int(obj.get(n, DEFAULT_TOP88[n])))
    return out

def load_json_inits(path: Path, cross_init_mode: str = "approx62") -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return normalize_inits(obj, cross_init_mode)

def inits_to_ints(inits: Dict[str, str]) -> Dict[str, int]:
    return {k: hex_int(v) for k, v in normalize_inits(inits).items()}

def ints_to_inits(ints: Dict[str, int]) -> Dict[str, str]:
    return normalize_inits({k: int_to_hex(v) for k, v in ints.items()})

# ============================================================
# Torch trainable LUTs / model
# ============================================================
class TrainableLUT6_2(nn.Module):
    def __init__(self, init_hex: str, init_p: float, noise_std: float):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_p, noise_std)
    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_p: float, noise_std: float):
        bits = torch.tensor(int_to_bits(hex_int(init_hex)), dtype=torch.float32, device=self.logits.device)
        p1 = min(max(float(init_p), 1e-4), 1 - 1e-4)
        p = torch.where(bits > 0.5, torch.full_like(bits, p1), torch.full_like(bits, 1 - p1))
        z = torch.log(p / (1 - p))
        if noise_std > 0:
            z = z + torch.randn_like(z) * noise_std
        self.logits.copy_(z)
    def table_prob(self, c):
        return sharp01(torch.sigmoid(self.logits), c)
    @staticmethod
    def soft_lut(inputs: List[torch.Tensor], table: torch.Tensor):
        out = torch.zeros_like(inputs[0])
        for addr in range(1 << len(inputs)):
            w = torch.ones_like(inputs[0])
            for i, xi in enumerate(inputs):
                w = w * (xi if ((addr >> i) & 1) else (1.0 - xi))
            out = out + w * table[addr]
        return out
    def forward(self, I0, I1, I2, I3, I4, I5, *, c_init, c_out, hard_middle=True):
        table = self.table_prob(c_init)
        if hard_middle:
            I0, I1, I2, I3, I4, I5 = [ste_binarize(x) for x in [I0, I1, I2, I3, I4, I5]]
        o5 = self.soft_lut([I0, I1, I2, I3, I4], table[:32])
        o6 = self.soft_lut([I0, I1, I2, I3, I4, I5], table)
        o5, o6 = sharp01(o5, c_out), sharp01(o6, c_out)
        if hard_middle:
            o5, o6 = ste_binarize(o5), ste_binarize(o6)
        return o5, o6
    def hard_hex(self, c_init):
        bits = (self.table_prob(c_init) >= 0.5).to(torch.int64).detach().cpu().tolist()
        return bits_to_hex(bits)
    def bin_reg(self, c_init):
        p = self.table_prob(c_init)
        return torch.mean(p * (1 - p))

class TrainableLUT6(nn.Module):
    def __init__(self, init_hex: str, init_p: float, noise_std: float):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_p, noise_std)
    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_p: float, noise_std: float):
        bits = torch.tensor(int_to_bits(hex_int(init_hex)), dtype=torch.float32, device=self.logits.device)
        p1 = min(max(float(init_p), 1e-4), 1 - 1e-4)
        p = torch.where(bits > 0.5, torch.full_like(bits, p1), torch.full_like(bits, 1 - p1))
        z = torch.log(p / (1 - p))
        if noise_std > 0:
            z = z + torch.randn_like(z) * noise_std
        self.logits.copy_(z)
    def table_prob(self, c):
        return sharp01(torch.sigmoid(self.logits), c)
    @staticmethod
    def soft_lut(inputs: List[torch.Tensor], table: torch.Tensor):
        out = torch.zeros_like(inputs[0])
        for addr in range(64):
            w = torch.ones_like(inputs[0])
            for i, xi in enumerate(inputs):
                w = w * (xi if ((addr >> i) & 1) else (1.0 - xi))
            out = out + w * table[addr]
        return out
    def forward(self, I0, I1, I2, I3, I4, I5, *, c_init, c_out, hard_middle=True):
        table = self.table_prob(c_init)
        if hard_middle:
            I0, I1, I2, I3, I4, I5 = [ste_binarize(x) for x in [I0, I1, I2, I3, I4, I5]]
        o = self.soft_lut([I0, I1, I2, I3, I4, I5], table)
        o = sharp01(o, c_out)
        if hard_middle:
            o = ste_binarize(o)
        return o
    def hard_hex(self, c_init):
        bits = (self.table_prob(c_init) >= 0.5).to(torch.int64).detach().cpu().tolist()
        return bits_to_hex(bits)
    def bin_reg(self, c_init):
        p = self.table_prob(c_init)
        return torch.mean(p * (1 - p))

class Approx88CrossModel(nn.Module):
    def __init__(self, base_inits: Dict[str, str], init_p: float, noise_std: float):
        super().__init__()
        base = normalize_inits(base_inits)
        lut62_names = [n for n in LOW66_NAMES if n not in ["u_comp4", "u_comp5", "u_comp6", "u_comp7"]] + CROSS_NAMES
        self.lut62 = nn.ModuleDict({n: TrainableLUT6_2(base[n], init_p, noise_std) for n in lut62_names})
        self.lut6 = nn.ModuleDict({n: TrainableLUT6(base[n], init_p, noise_std) for n in ["u_comp4", "u_comp5", "u_comp6", "u_comp7"]})
        self.top88 = nn.ModuleDict({n: TrainableLUT6_2(base[n], init_p, noise_std) for n in TOP88_NAMES})
    @staticmethod
    def const(x, v):
        return torch.full_like(x, float(v))
    @staticmethod
    def bits_value(bits: List[torch.Tensor], width: int):
        s = torch.stack(bits, dim=1)
        w = torch.tensor([1 << i for i in range(width)], device=s.device, dtype=torch.float32)
        return torch.sum(s * w, dim=1)
    def soft_xor2(self, a, b, *, c_out, hard_middle):
        y = a + b - 2.0 * a * b
        y = sharp01(y, c_out)
        return ste_binarize(y) if hard_middle else y
    def soft_muxcy(self, s, ci, di, *, c_out, hard_middle):
        # Xilinx MUXCY: CO = S ? CI : DI.
        y = s * ci + (1.0 - s) * di
        y = sharp01(y, c_out)
        return ste_binarize(y) if hard_middle else y
    def approx62_named(self, prefix: str, a: torch.Tensor, b2: torch.Tensor, *, c_init, c_out, hard_middle):
        ab = int_bits_t(a, 6)
        bb = int_bits_t(b2, 2)
        one = self.const(ab[0], 1.0)
        p0, p1 = self.lut62[f"{prefix}_lut1"](bb[0], bb[1], ab[0], ab[1], one, one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p2, p3 = self.lut62[f"{prefix}_lut2"](bb[0], bb[1], ab[1], ab[2], ab[3], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p4, p5 = self.lut62[f"{prefix}_lut3"](bb[0], bb[1], ab[3], ab[4], ab[5], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p6, p7 = self.lut62[f"{prefix}_lut4"](bb[0], bb[1], ab[4], ab[5], one, one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return [p0, p1, p2, p3, p4, p5, p6, p7]
    def comp66_pair(self, plow, pmid, phigh, *, c_init, c_out, hard_middle):
        z = self.const(plow[0], 0.0)
        o = self.const(plow[0], 1.0)
        prod = [z for _ in range(12)]
        prod[0], prod[1], prod[10], prod[11] = plow[0], plow[1], phigh[6], phigh[7]
        prod[2], prod[3] = self.lut62["u_comp23"](plow[2], pmid[0], plow[3], pmid[1], z, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[4] = self.lut6["u_comp4"](plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1], c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[5] = self.lut6["u_comp5"](plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1], c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[6] = self.lut6["u_comp6"](plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3], c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[7] = self.lut6["u_comp7"](plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3], c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod[8], prod[9] = self.lut62["u_comp89"](pmid[6], phigh[4], pmid[7], phigh[5], z, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return prod
    def approx66_low(self, al, bl, *, c_init, c_out, hard_middle):
        plow = self.approx62_named("low", al, bl & 3, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        pmid = self.approx62_named("mid", al, (bl >> 2) & 3, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        phigh = self.approx62_named("high", al, (bl >> 4) & 3, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        bits = self.comp66_pair(plow, pmid, phigh, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return self.bits_value(bits, 12)
    def approx66_low_bits(self, al, bl, *, c_init, c_out, hard_middle):
        plow = self.approx62_named("low", al, bl & 3, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        pmid = self.approx62_named("mid", al, (bl >> 2) & 3, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        phigh = self.approx62_named("high", al, (bl >> 4) & 3, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return self.comp66_pair(plow, pmid, phigh, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
    def approx62_value(self, prefix, a6, b2, *, c_init, c_out, hard_middle):
        bits = self.approx62_named(prefix, a6, b2, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return self.bits_value(bits, 8)
    def comp88_top_bits(self, ll_bits, hl_bits, lh_bits, ah, bh, *, c_init, c_out, hard_middle):
        ah_bits = int_bits_t(ah, 2)
        bh_bits = int_bits_t(bh, 2)
        x1 = ah_bits[1] * bh_bits[0]
        x2 = ah_bits[0] * bh_bits[1]
        hh_bits = [
            ah_bits[0] * bh_bits[0],
            x1 + x2 - 2.0 * x1 * x2,
            x1 * x2,
            bh_bits[1],
            ah_bits[1],
        ]
        z = self.const(ll_bits[0], 0.0)
        o = self.const(ll_bits[0], 1.0)
        a_reg = ll_bits[6:12] + hh_bits[0:3]
        b_reg = lh_bits
        c_reg = hl_bits

        p = [z for _ in range(12)]
        g = [z for _ in range(12)]
        g[9], g[10], g[11] = z, z, z

        g[0], p[0] = self.top88["u88_gp0"](c_reg[0], b_reg[0], a_reg[0], o, o, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        for j in range(1, 8):
            g[j], p[j] = self.top88[f"u88_gp{j}"](c_reg[j], b_reg[j], a_reg[j], g[j - 1], o, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        g[8], p[8] = self.top88["u88_gp8"](a_reg[8], g[7], hh_bits[4], hh_bits[3], o, o, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p[9], p[10], p[11] = g[8], z, z

        carry = z
        sums = []
        for i in range(12):
            sums.append(self.soft_xor2(p[i], carry, c_out=c_out, hard_middle=hard_middle))
            di = z if i == 0 else g[i - 1]
            carry = self.soft_muxcy(p[i], carry, di, c_out=c_out, hard_middle=hard_middle)
        return ll_bits[0:6] + sums[0:10]
    def forward(self, a, b, *, c_init, c_out, hard_middle=True):
        al = a & 63
        ah = (a >> 6) & 3
        bl = b & 63
        bh = (b >> 6) & 3
        ll_bits = self.approx66_low_bits(al, bl, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        hl_bits = self.approx62_named("hl", bl, ah, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        lh_bits = self.approx62_named("lh", al, bh, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        prod_bits = self.comp88_top_bits(ll_bits, hl_bits, lh_bits, ah, bh, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return self.bits_value(prod_bits, 16)
    def hard_inits(self, c_init):
        out = {}
        for n, m in self.lut62.items():
            out[n] = m.hard_hex(c_init)
        for n, m in self.lut6.items():
            out[n] = m.hard_hex(c_init)
        for n, m in self.top88.items():
            out[n] = m.hard_hex(c_init)
        return normalize_inits(out)
    def bin_reg(self, c_init):
        regs = [m.bin_reg(c_init) for m in list(self.lut62.values()) + list(self.lut6.values()) + list(self.top88.values())]
        return torch.mean(torch.stack(regs))
    def set_train_scope(self, scope: str):
        scopes = {
            "low": set(LOW66_NAMES),
            "cross": set(CROSS_NAMES),
            "top": set(TOP88_NAMES),
            "cross_top": set(CROSS_NAMES + TOP88_NAMES),
            "low_top": set(LOW66_NAMES + TOP88_NAMES),
            "all": set(TRAIN_NAMES),
        }
        active = scopes[scope]
        for n, m in self.lut62.items():
            req = n in active
            for p in m.parameters():
                p.requires_grad = req
        for n, m in self.lut6.items():
            req = n in active
            for p in m.parameters():
                p.requires_grad = req
        for n, m in self.top88.items():
            req = n in active
            for p in m.parameters():
                p.requires_grad = req

# ============================================================
# Fast hard evaluator
# ============================================================
@dataclass
class Metrics:
    total_cases: int
    error_cases: int
    ER: float
    MED: float
    NED: float
    MRED: float
    WCE: int

def metrics_to_dict(m: Metrics):
    return {"total_cases": m.total_cases, "error_cases": m.error_cases, "ER": m.ER, "MED": m.MED, "NED": m.NED, "MRED": m.MRED, "WCE": m.WCE}

def metric_str(m: Metrics) -> str:
    return f"MRED={m.MRED:.10f} MED={m.MED:.4f} ER={m.ER:.4f} WCE={m.WCE} err={m.error_cases}/{m.total_cases}"

class Eval88Cross:
    def __init__(self, mred_denom: str = "total"):
        A, B = np.meshgrid(np.arange(256, dtype=np.int64), np.arange(256, dtype=np.int64), indexing="ij")
        self.a = A.reshape(-1)
        self.b = B.reshape(-1)
        self.exact = (self.a * self.b).astype(np.int64)
        self.mask = self.exact > 0
        self.mred_denom = mred_denom
        self.al = self.a & 63
        self.ah = (self.a >> 6) & 3
        self.bl = self.b & 63
        self.bh = (self.b >> 6) & 3
        self.al_bits = [((self.al >> i) & 1).astype(np.uint8) for i in range(6)]
        self.bl_bits = [((self.bl >> i) & 1).astype(np.uint8) for i in range(6)]
        self.z = np.zeros_like(self.al_bits[0], dtype=np.uint8)
        self.o = np.ones_like(self.al_bits[0], dtype=np.uint8)
    @staticmethod
    def bits(v: int) -> np.ndarray:
        return np.array([(int(v) >> i) & 1 for i in range(64)], dtype=np.uint8)
    @staticmethod
    def lut62(tab, I0, I1, I2, I3, I4, I5):
        addr5 = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4)
        addr6 = addr5 + (I5 << 5)
        return tab[addr5], tab[addr6]
    @staticmethod
    def lut6(tab, I0, I1, I2, I3, I4, I5):
        addr = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4) + (I5 << 5)
        return tab[addr]
    def tables(self, ints):
        return {n: self.bits(ints[n]) for n in TRAIN_NAMES}
    def approx62(self, prefix, a_bits, b2, tabs):
        bb0 = (b2 & 1).astype(np.uint8)
        bb1 = ((b2 >> 1) & 1).astype(np.uint8)
        p0, p1 = self.lut62(tabs[f"{prefix}_lut1"], bb0, bb1, a_bits[0], a_bits[1], self.o, self.o)
        p2, p3 = self.lut62(tabs[f"{prefix}_lut2"], bb0, bb1, a_bits[1], a_bits[2], a_bits[3], self.o)
        p4, p5 = self.lut62(tabs[f"{prefix}_lut3"], bb0, bb1, a_bits[3], a_bits[4], a_bits[5], self.o)
        p6, p7 = self.lut62(tabs[f"{prefix}_lut4"], bb0, bb1, a_bits[4], a_bits[5], self.o, self.o)
        return [p0, p1, p2, p3, p4, p5, p6, p7]
    def value_from_bits(self, bits):
        v = np.zeros_like(self.exact)
        for i, p in enumerate(bits):
            v += p.astype(np.int64) << i
        return v
    def low66_bits(self, tabs):
        lowb0 = self.bl & 3
        midb = (self.bl >> 2) & 3
        highb = (self.bl >> 4) & 3
        plow = self.approx62("low", self.al_bits, lowb0, tabs)
        pmid = self.approx62("mid", self.al_bits, midb, tabs)
        phigh = self.approx62("high", self.al_bits, highb, tabs)
        prod = [None] * 12
        prod[0], prod[1], prod[10], prod[11] = plow[0], plow[1], phigh[6], phigh[7]
        prod[2], prod[3] = self.lut62(tabs["u_comp23"], plow[2], pmid[0], plow[3], pmid[1], self.z, self.o)
        prod[4] = self.lut6(tabs["u_comp4"], plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
        prod[5] = self.lut6(tabs["u_comp5"], plow[4], pmid[2], phigh[0], plow[5], pmid[3], phigh[1])
        prod[6] = self.lut6(tabs["u_comp6"], plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
        prod[7] = self.lut6(tabs["u_comp7"], plow[6], pmid[4], phigh[2], plow[7], pmid[5], phigh[3])
        prod[8], prod[9] = self.lut62(tabs["u_comp89"], pmid[6], phigh[4], pmid[7], phigh[5], self.z, self.o)
        return prod
    def comp88_bits(self, ll_bits, hl_bits, lh_bits, tabs):
        ah0 = (self.ah & 1).astype(np.uint8)
        ah1 = ((self.ah >> 1) & 1).astype(np.uint8)
        bh0 = (self.bh & 1).astype(np.uint8)
        bh1 = ((self.bh >> 1) & 1).astype(np.uint8)
        x1 = (ah1 & bh0).astype(np.uint8)
        x2 = (ah0 & bh1).astype(np.uint8)
        hh_bits = [
            (ah0 & bh0).astype(np.uint8),
            np.bitwise_xor(x1, x2).astype(np.uint8),
            (x1 & x2).astype(np.uint8),
            bh1,
            ah1,
        ]
        a_reg = ll_bits[6:12] + hh_bits[0:3]
        b_reg = lh_bits
        c_reg = hl_bits

        p = [self.z for _ in range(12)]
        g = [self.z for _ in range(12)]
        g[0], p[0] = self.lut62(tabs["u88_gp0"], c_reg[0], b_reg[0], a_reg[0], self.o, self.o, self.o)
        for j in range(1, 8):
            g[j], p[j] = self.lut62(tabs[f"u88_gp{j}"], c_reg[j], b_reg[j], a_reg[j], g[j - 1], self.o, self.o)
        g[8], p[8] = self.lut62(tabs["u88_gp8"], a_reg[8], g[7], hh_bits[4], hh_bits[3], self.o, self.o)
        g[9], g[10], g[11] = self.z, self.z, self.z
        p[9], p[10], p[11] = g[8], self.z, self.z

        carry = self.z
        sums = []
        for i in range(12):
            sums.append(np.bitwise_xor(p[i], carry).astype(np.uint8))
            di = self.z if i == 0 else g[i - 1]
            carry = np.where(p[i].astype(bool), carry, di).astype(np.uint8)
        return ll_bits[0:6] + sums[0:10]
    def evaluate_ints(self, ints: Dict[str, int]) -> Metrics:
        tabs = self.tables(ints)
        ll_bits = self.low66_bits(tabs)
        hl_bits = self.approx62("hl", self.bl_bits, self.ah, tabs)
        lh_bits = self.approx62("lh", self.al_bits, self.bh, tabs)
        val = self.value_from_bits(self.comp88_bits(ll_bits, hl_bits, lh_bits, tabs))
        err = np.abs(val - self.exact)
        total = err.size
        err_cases = int(np.count_nonzero(err))
        med = float(err.mean())
        wce = int(err.max())
        red_sum = float(np.sum(err[self.mask] / self.exact[self.mask]))
        denom = total if self.mred_denom == "total" else int(np.count_nonzero(self.mask))
        mred = red_sum / denom
        return Metrics(total, err_cases, err_cases / total, med, med / (255 * 255), mred, wce)
    def evaluate(self, inits):
        return self.evaluate_ints(inits_to_ints(inits))
    def approx_values(self, ints):
        tabs = self.tables(ints)
        ll_bits = self.low66_bits(tabs)
        hl_bits = self.approx62("hl", self.bl_bits, self.ah, tabs)
        lh_bits = self.approx62("lh", self.al_bits, self.bh, tabs)
        return self.value_from_bits(self.comp88_bits(ll_bits, hl_bits, lh_bits, tabs))

_EVAL_CACHE: Dict[str, Eval88Cross] = {}
def get_eval(mred_denom: str) -> Eval88Cross:
    if mred_denom not in _EVAL_CACHE:
        _EVAL_CACHE[mred_denom] = Eval88Cross(mred_denom)
    return _EVAL_CACHE[mred_denom]

def compute_hard_metrics(inits, mred_denom):
    return get_eval(mred_denom).evaluate(inits)

# ============================================================
# Dataset / loss
# ============================================================
def make_dataset(device):
    aa, bb, exact = [], [], []
    for a in range(256):
        for b in range(256):
            aa.append(a); bb.append(b); exact.append(a*b)
    return (torch.tensor(aa, dtype=torch.long, device=device),
            torch.tensor(bb, dtype=torch.long, device=device),
            torch.tensor(exact, dtype=torch.float32, device=device))

def loss_fn(approx, exact, bin_reg, zero_weight, med_weight, bin_weight, mred_denom):
    mask = exact > 0
    abs_err = torch.abs(approx - exact)
    red_sum = torch.sum(abs_err[mask] / exact[mask])
    denom = exact.numel() if mred_denom == "total" else torch.count_nonzero(mask).item()
    mred = red_sum / float(denom)
    zero = torch.mean(abs_err[~mask]) / 65536.0 if torch.any(~mask) else torch.tensor(0.0, device=approx.device)
    medn = torch.mean(abs_err) / 65536.0
    loss = mred + zero_weight * zero + med_weight * medn + bin_weight * bin_reg
    return loss, float(mred.detach().cpu()), float(zero.detach().cpu()), float(medn.detach().cpu()), float(bin_reg.detach().cpu())

# ============================================================
# Search helpers
# ============================================================
def names_for_scope(scope: str) -> List[str]:
    if scope == "low": return LOW66_NAMES
    if scope == "cross": return CROSS_NAMES
    if scope == "top": return TOP88_NAMES
    if scope == "cross_top": return CROSS_NAMES + TOP88_NAMES
    if scope == "low_top": return LOW66_NAMES + TOP88_NAMES
    if scope == "all": return TRAIN_NAMES
    raise ValueError(scope)

def parse_lut_names(s: str) -> List[str]:
    if s == "all":
        return TRAIN_NAMES
    out: List[str] = []
    for tok in [x.strip() for x in s.split(",") if x.strip()]:
        if tok == "all":
            out.extend(TRAIN_NAMES)
        elif tok in ["low", "cross", "top", "cross_top", "low_top"]:
            out.extend(names_for_scope(tok))
        else:
            if tok not in TRAIN_NAMES:
                raise ValueError(f"Unknown LUT name {tok}")
            out.append(tok)
    # preserve order, remove duplicates
    seen = set()
    dedup = []
    for n in out:
        if n not in seen:
            seen.add(n); dedup.append(n)
    return dedup

def candidate_bits(names: List[str], random_order=False) -> List[Tuple[str, int]]:
    bits = [(n, b) for n in names for b in USED_BITS[n]]
    if random_order:
        random.shuffle(bits)
    return bits

def normalize_candidate_bit_list(bits: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    seen = set()
    for n, b in bits:
        b = int(b)
        if n in USED_BITS and b in USED_BITS[n] and (n, b) not in seen:
            seen.add((n, b))
            out.append((n, b))
    return sorted(out, key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))

def passes(m: Metrics, max_wce: int, max_er: float, max_med: float) -> bool:
    if max_wce >= 0 and m.WCE > max_wce: return False
    if max_er >= 0 and m.ER > max_er: return False
    if max_med >= 0 and m.MED > max_med: return False
    return True

def flip_bit_ints(ints, n, b):
    trial = dict(ints)
    trial[n] ^= (1 << b)
    return trial

def greedy_single_bits(start, mred_denom, rounds, mode, bits, random_order, max_wce, max_er, max_med):
    cur = inits_to_ints(start); cur_m = get_eval(mred_denom).evaluate_ints(cur)
    bits = normalize_candidate_bit_list(bits)
    print(f"\n[single] start {metric_str(cur_m)} rounds={rounds} mode={mode} bits={len(bits)} max_wce={max_wce}")
    for r in range(rounds):
        order = list(bits)
        if random_order: random.shuffle(order)
        improved = False
        if mode == "best":
            best_t = None; best_m = cur_m
            for n, b in order:
                tm = get_eval(mred_denom).evaluate_ints(flip_bit_ints(cur, n, b))
                if passes(tm, max_wce, max_er, max_med) and tm.MRED < best_m.MRED - 1e-14:
                    best_t = (n, b); best_m = tm
            if best_t:
                n, b = best_t; old = cur_m.MRED; cur = flip_bit_ints(cur, n, b); cur_m = best_m; improved = True
                print(f"[single] KEEP-BEST r={r+1} {n}:{b:02d} {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} WCE={cur_m.WCE}")
        else:
            for n, b in order:
                trial = flip_bit_ints(cur, n, b); tm = get_eval(mred_denom).evaluate_ints(trial)
                if passes(tm, max_wce, max_er, max_med) and tm.MRED < cur_m.MRED - 1e-14:
                    old = cur_m.MRED; cur = trial; cur_m = tm; improved = True
                    print(f"[single] KEEP r={r+1} {n}:{b:02d} {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} WCE={cur_m.WCE}")
                    break
        if not improved:
            print(f"[single] r={r+1} no improvement, stop")
            break
    return ints_to_inits(cur), cur_m

def greedy_single(start, mred_denom, rounds, mode, names, random_order, max_wce, max_er, max_med):
    return greedy_single_bits(start,mred_denom,rounds,mode,candidate_bits(names, random_order=False),random_order,max_wce,max_er,max_med)

def greedy_pair_bits(start, mred_denom, rounds, mode, bits, random_order, max_pairs, max_wce, max_er, max_med):
    cur = inits_to_ints(start); cur_m = get_eval(mred_denom).evaluate_ints(cur)
    bits = normalize_candidate_bit_list(bits)
    pairs = [(i, j) for i in range(len(bits)) for j in range(i+1, len(bits))]
    print(f"\n[pair] start {metric_str(cur_m)} rounds={rounds} mode={mode} bits={len(bits)} full_pairs={len(pairs)} max_pairs={max_pairs}")
    for r in range(rounds):
        order = list(pairs)
        if random_order: random.shuffle(order)
        if max_pairs > 0 and len(order) > max_pairs: order = order[:max_pairs]
        improved = False
        if mode == "best":
            best = None; best_m = cur_m
            for i, j in order:
                n1, b1 = bits[i]; n2, b2 = bits[j]
                trial = dict(cur); trial[n1] ^= (1 << b1); trial[n2] ^= (1 << b2)
                tm = get_eval(mred_denom).evaluate_ints(trial)
                if passes(tm, max_wce, max_er, max_med) and tm.MRED < best_m.MRED - 1e-14:
                    best = (n1, b1, n2, b2, trial); best_m = tm
            if best:
                n1, b1, n2, b2, cur = best; old = cur_m.MRED; cur_m = best_m; improved = True
                print(f"[pair] KEEP-BEST r={r+1} {n1}:{b1:02d}+{n2}:{b2:02d} {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} WCE={cur_m.WCE}")
        else:
            for i, j in order:
                n1, b1 = bits[i]; n2, b2 = bits[j]
                trial = dict(cur); trial[n1] ^= (1 << b1); trial[n2] ^= (1 << b2)
                tm = get_eval(mred_denom).evaluate_ints(trial)
                if passes(tm, max_wce, max_er, max_med) and tm.MRED < cur_m.MRED - 1e-14:
                    old = cur_m.MRED; cur = trial; cur_m = tm; improved = True
                    print(f"[pair] KEEP r={r+1} {n1}:{b1:02d}+{n2}:{b2:02d} {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} WCE={cur_m.WCE}")
                    break
        if not improved:
            print(f"[pair] r={r+1} no improvement, stop")
            break
    return ints_to_inits(cur), cur_m

def greedy_pair(start, mred_denom, rounds, mode, names, random_order, max_pairs, max_wce, max_er, max_med):
    return greedy_pair_bits(start,mred_denom,rounds,mode,candidate_bits(names, random_order=False),random_order,max_pairs,max_wce,max_er,max_med)

def top_error_candidate_bits(inits, mred_denom, topk, scope_names):
    E = get_eval(mred_denom)
    ints = inits_to_ints(inits)
    val = E.approx_values(ints)
    err = np.abs(val - E.exact)
    rel = np.zeros_like(E.exact, dtype=np.float64)
    rel[E.mask] = err[E.mask] / E.exact[E.mask]
    idxs = np.argsort(-rel)[:topk]
    cands: Set[Tuple[str, int]] = set()
    for idx in idxs:
        a = int(E.a[idx]); b = int(E.b[idx])
        al = a & 63; ah = (a >> 6) & 3; bl = b & 63; bh = (b >> 6) & 3
        # low approx66 addresses
        for seg, b2, aa in [("low", bl & 3, al), ("mid", (bl >> 2) & 3, al), ("high", (bl >> 4) & 3, al)]:
            for lut in APPROX_IDS:
                cands.add((f"{seg}_{lut}", approx62_addr(lut, aa, b2, False)))
                cands.add((f"{seg}_{lut}", approx62_addr(lut, aa, b2, True)))
        for prefix, aa, b2 in [("hl", bl, ah), ("lh", al, bh)]:
            for lut in APPROX_IDS:
                cands.add((f"{prefix}_{lut}", approx62_addr(lut, aa, b2, False)))
                cands.add((f"{prefix}_{lut}", approx62_addr(lut, aa, b2, True)))
        # For comp LUTs, include all used bits if low comp is in scope. This is conservative but still targeted by scope.
        for n in ["u_comp23", "u_comp89", "u_comp4", "u_comp5", "u_comp6", "u_comp7"]:
            if n in scope_names:
                for bit in USED_BITS[n]: cands.add((n, bit))
        for n in TOP88_NAMES:
            if n in scope_names:
                for bit in USED_BITS[n]: cands.add((n, bit))
    scope_set = set(scope_names)
    out = [(n,b) for n,b in cands if n in scope_set and b in USED_BITS[n]]
    out = sorted(set(out), key=lambda x: (TRAIN_NAMES.index(x[0]), x[1]))
    print(f"[top-candidates] topk={topk} scope={','.join(scope_names)} bits={len(out)}")
    return out

def neutral_bits(start, mred_denom, source_bits, topn, margin, max_wce, max_er, max_med):
    cur = inits_to_ints(start); base_m = get_eval(mred_denom).evaluate_ints(cur)
    scored = []
    print(f"\n[neutral] scan source={len(source_bits)} base={metric_str(base_m)} margin={margin}")
    for n, b in source_bits:
        trial = flip_bit_ints(cur, n, b)
        tm = get_eval(mred_denom).evaluate_ints(trial)
        d = tm.MRED - base_m.MRED
        if passes(tm, max_wce, max_er, max_med) and d <= margin:
            scored.append((d, n, b))
    scored.sort(key=lambda x: x[0])
    out = [(n,b) for d,n,b in scored[:topn]]
    print(f"[neutral] kept={len(out)}")
    for d,n,b in scored[:min(20,len(scored))]:
        print(f"  {n}:{b:02d} delta={d:+.10f}")
    return out

def escape_search(start, mred_denom, source_bits, neutral, iters, kmin, kmax, single_rounds, pair_after, pair_rounds, pair_max_pairs, max_wce, max_er, max_med):
    best = normalize_inits(start); best_m = compute_hard_metrics(best, mred_denom)
    if not neutral:
        print("[escape] no neutral bits; skip")
        return best, best_m
    kmax = min(kmax, len(neutral))
    kmin = min(kmin, kmax)
    print(f"\n[escape] start {metric_str(best_m)} iters={iters} k=[{kmin},{kmax}] neutral={len(neutral)} polish_bits={len(source_bits)}")
    for it in range(1, iters+1):
        ints = inits_to_ints(best)
        chosen = random.sample(neutral, random.randint(kmin, kmax))
        for n,b in chosen: ints[n] ^= (1 << b)
        pert = ints_to_inits(ints)
        pm = compute_hard_metrics(pert, mred_denom)
        print(f"[escape] iter={it}/{iters} perturb {metric_str(pm)}")
        loc, lm = greedy_single_bits(pert, mred_denom, single_rounds, "first", source_bits, True, max_wce, max_er, max_med)
        if pair_after:
            loc, lm = greedy_pair_bits(loc, mred_denom, pair_rounds, "first", source_bits, True, pair_max_pairs, max_wce, max_er, max_med)
        if lm.MRED < best_m.MRED - 1e-14 and passes(lm, max_wce, max_er, max_med):
            print(f"[escape] ACCEPT {best_m.MRED:.10f}->{lm.MRED:.10f}")
            best, best_m = loc, lm
    return best, best_m

# ============================================================
# Output
# ============================================================
def write_best(out_dir: Path, best: Dict, prefix="best"):
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{prefix}_approx88_cascade_inits.json"
    with jp.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False, allow_nan=False)
    if prefix == "best":
        (out_dir / "best_approx88_cascade_inits.json").write_text(jp.read_text(encoding="utf-8"), encoding="utf-8")
        (out_dir / "best_approx88_cross62_inits.json").write_text(jp.read_text(encoding="utf-8"), encoding="utf-8")
        # convenience name for chaining
        (out_dir / "best_approx88_inits.json").write_text(jp.read_text(encoding="utf-8"), encoding="utf-8")
    write_verilog(out_dir / f"{prefix}_approx88_cascade.v", best)
    if prefix == "best":
        (out_dir / "best_approx88_cascade.v").write_text((out_dir / f"{prefix}_approx88_cascade.v").read_text(encoding="utf-8"), encoding="utf-8")
        (out_dir / "best_approx88_cross62.v").write_text((out_dir / f"{prefix}_approx88_cascade.v").read_text(encoding="utf-8"), encoding="utf-8")

def write_verilog(path: Path, best: Dict):
    I = normalize_inits(best["inits"])
    with path.open("w", encoding="utf-8") as f:
        f.write("// Generated by train_approx88_cascade.py\n")
        f.write(f"// MRED={best['metrics']['MRED']:.10f} MED={best['metrics']['MED']:.10f} ER={best['metrics']['ER']:.10f} WCE={best['metrics']['WCE']}\n\n")
        f.write("module approx88_cascade(input wire [7:0] a, input wire [7:0] b, output wire [15:0] prod);\n")
        f.write("wire [1:0] ah=a[7:6], bh=b[7:6]; wire [5:0] al=a[5:0], bl=b[5:0];\n")
        f.write("wire [11:0] ll; wire [7:0] hl, lh; wire hh_0, hh_1, hh_2; wire [4:0] hh;\n")
        f.write("approx66_cross_low U_LL(.a(al),.b(bl),.prod(ll));\n")
        f.write("approx62_hl U_HL(.a(bl),.b(ah),.prod(hl));\n")
        f.write("approx62_lh U_LH(.a(al),.b(bh),.prod(lh));\n")
        f.write("LUT6 #(.INIT(64'h8000000000000000)) LUT_HH0(.I0(a[6]),.I1(b[6]),.I2(1'b1),.I3(1'b1),.I4(1'b1),.I5(1'b1),.O(hh_0));\n")
        f.write("LUT6_2 #(.INIT(64'h8000800078887888)) LUT_HH12(.I0(a[7]),.I1(b[6]),.I2(a[6]),.I3(b[7]),.I4(1'b1),.I5(1'b1),.O6(hh_2),.O5(hh_1));\n")
        f.write("assign hh={a[7],b[7],hh_2,hh_1,hh_0};\n")
        f.write("comp88_cascade U_COMP(.hh(hh),.hl(hl),.lh(lh),.ll(ll),.prod(prod));\n")
        f.write("endmodule\n\n")
        f.write("module approx66_cross_low(input wire [5:0] a,input wire [5:0] b,output wire [11:0] prod);\n")
        f.write("wire [7:0] plow,pmid,phigh;\n")
        f.write("approx62_low U_LOW(.a(a),.b(b[1:0]),.prod(plow));\n")
        f.write("approx62_mid U_MID(.a(a),.b(b[3:2]),.prod(pmid));\n")
        f.write("approx62_high U_HIGH(.a(a),.b(b[5:4]),.prod(phigh));\n")
        f.write("comp66_pair_cross U_COMP(.plow(plow),.pmid(pmid),.phigh(phigh),.prod(prod));\nendmodule\n\n")
        for seg in SEGMENTS:
            f.write(f"module approx62_{seg}(input wire [5:0] a,input wire [1:0] b,output wire [7:0] prod);\n")
            for idx,lut in enumerate(APPROX_IDS,1):
                init=I[f"{seg}_{lut}"]
                if lut=="lut1": conn=".I0(b[0]),.I1(b[1]),.I2(a[0]),.I3(a[1]),.I4(1'b1),.I5(1'b1),.O5(prod[0]),.O6(prod[1])"
                elif lut=="lut2": conn=".I0(b[0]),.I1(b[1]),.I2(a[1]),.I3(a[2]),.I4(a[3]),.I5(1'b1),.O5(prod[2]),.O6(prod[3])"
                elif lut=="lut3": conn=".I0(b[0]),.I1(b[1]),.I2(a[3]),.I3(a[4]),.I4(a[5]),.I5(1'b1),.O5(prod[4]),.O6(prod[5])"
                else: conn=".I0(b[0]),.I1(b[1]),.I2(a[4]),.I3(a[5]),.I4(1'b1),.I5(1'b1),.O5(prod[6]),.O6(prod[7])"
                f.write(f"LUT6_2 #(.INIT({init})) LUT6_inst{idx}({conn});\n")
            f.write("endmodule\n\n")
        for prefix in ["hl","lh"]:
            f.write(f"module approx62_{prefix}(input wire [5:0] a,input wire [1:0] b,output wire [7:0] prod);\n")
            for idx,lut in enumerate(APPROX_IDS,1):
                init=I[f"{prefix}_{lut}"]
                if lut=="lut1": conn=".I0(b[0]),.I1(b[1]),.I2(a[0]),.I3(a[1]),.I4(1'b1),.I5(1'b1),.O5(prod[0]),.O6(prod[1])"
                elif lut=="lut2": conn=".I0(b[0]),.I1(b[1]),.I2(a[1]),.I3(a[2]),.I4(a[3]),.I5(1'b1),.O5(prod[2]),.O6(prod[3])"
                elif lut=="lut3": conn=".I0(b[0]),.I1(b[1]),.I2(a[3]),.I3(a[4]),.I4(a[5]),.I5(1'b1),.O5(prod[4]),.O6(prod[5])"
                else: conn=".I0(b[0]),.I1(b[1]),.I2(a[4]),.I3(a[5]),.I4(1'b1),.I5(1'b1),.O5(prod[6]),.O6(prod[7])"
                f.write(f"LUT6_2 #(.INIT({init})) LUT6_inst{idx}({conn});\n")
            f.write("endmodule\n\n")
        f.write("module comp66_pair_cross(input wire [7:0] plow,input wire [7:0] pmid,input wire [7:0] phigh,output wire [11:0] prod);\n")
        f.write("assign prod[0]=plow[0]; assign prod[1]=plow[1]; assign prod[10]=phigh[6]; assign prod[11]=phigh[7];\n")
        f.write(f"LUT6_2 #(.INIT({I['u_comp23']})) u_comp23(.I0(plow[2]),.I1(pmid[0]),.I2(plow[3]),.I3(pmid[1]),.I4(1'b0),.I5(1'b1),.O5(prod[2]),.O6(prod[3]));\n")
        f.write(f"LUT6 #(.INIT({I['u_comp4']})) u_comp4(.I0(plow[4]),.I1(pmid[2]),.I2(phigh[0]),.I3(plow[5]),.I4(pmid[3]),.I5(phigh[1]),.O(prod[4]));\n")
        f.write(f"LUT6 #(.INIT({I['u_comp5']})) u_comp5(.I0(plow[4]),.I1(pmid[2]),.I2(phigh[0]),.I3(plow[5]),.I4(pmid[3]),.I5(phigh[1]),.O(prod[5]));\n")
        f.write(f"LUT6 #(.INIT({I['u_comp6']})) u_comp6(.I0(plow[6]),.I1(pmid[4]),.I2(phigh[2]),.I3(plow[7]),.I4(pmid[5]),.I5(phigh[3]),.O(prod[6]));\n")
        f.write(f"LUT6 #(.INIT({I['u_comp7']})) u_comp7(.I0(plow[6]),.I1(pmid[4]),.I2(phigh[2]),.I3(plow[7]),.I4(pmid[5]),.I5(phigh[3]),.O(prod[7]));\n")
        f.write(f"LUT6_2 #(.INIT({I['u_comp89']})) u_comp89(.I0(pmid[6]),.I1(phigh[4]),.I2(pmid[7]),.I3(phigh[5]),.I4(1'b0),.I5(1'b1),.O5(prod[8]),.O6(prod[9]));\n")
        f.write("endmodule\n")
        f.write("\nmodule comp88_cascade(input wire [4:0] hh,input wire [7:0] hl,input wire [7:0] lh,input wire [11:0] ll,output wire [15:0] prod);\n")
        f.write("wire [8:0] a_reg; wire [7:0] b_reg,c_reg; wire [11:0] p,g; wire [12:0] c_i; wire [11:0] sum;\n")
        f.write("assign a_reg[5:0]=ll[11:6]; assign a_reg[8:6]=hh[2:0]; assign b_reg=lh; assign c_reg=hl;\n")
        f.write("assign prod[5:0]=ll[5:0]; assign g[11:9]=3'b000; assign p[11:9]={2'b00,g[8]}; assign c_i[0]=1'b0;\n")
        f.write(f"LUT6_2 #(.INIT({I['u88_gp0']})) u88_gp0(.I0(c_reg[0]),.I1(b_reg[0]),.I2(a_reg[0]),.I3(1'b1),.I4(1'b1),.I5(1'b1),.O6(p[0]),.O5(g[0]));\n")
        for j in range(1, 8):
            f.write(f"LUT6_2 #(.INIT({I[f'u88_gp{j}']})) u88_gp{j}(.I0(c_reg[{j}]),.I1(b_reg[{j}]),.I2(a_reg[{j}]),.I3(g[{j-1}]),.I4(1'b1),.I5(1'b1),.O6(p[{j}]),.O5(g[{j}]));\n")
        f.write(f"LUT6_2 #(.INIT({I['u88_gp8']})) u88_gp8(.I0(a_reg[8]),.I1(g[7]),.I2(hh[4]),.I3(hh[3]),.I4(1'b1),.I5(1'b1),.O6(p[8]),.O5(g[8]));\n")
        f.write("CARRY4 CARRY4_0(.CO(c_i[4:1]),.O(sum[3:0]),.CI(c_i[0]),.CYINIT(1'b0),.DI({g[2:0],1'b0}),.S(p[3:0]));\n")
        f.write("CARRY4 CARRY4_1(.CO(c_i[8:5]),.O(sum[7:4]),.CI(c_i[4]),.CYINIT(1'b0),.DI(g[6:3]),.S(p[7:4]));\n")
        f.write("CARRY4 CARRY4_2(.CO(c_i[12:9]),.O(sum[11:8]),.CI(c_i[8]),.CYINIT(1'b0),.DI(g[10:7]),.S(p[11:8]));\n")
        f.write("assign prod[15:6]=sum[9:0];\nendmodule\n")

# ============================================================
# Main
# ============================================================
def make_base(args):
    if args.init_mode == "json":
        return load_json_inits(Path(args.base_inits_json), args.cross_init_mode)
    if args.init_mode == "default":
        return normalize_inits({}, args.cross_init_mode)
    if args.init_mode == "random":
        d = {}
        for n in TRAIN_NAMES:
            p = args.random_prob
            bits = [1 if random.random() < p else 0 for _ in range(64)]
            d[n] = bits_to_hex(bits)
        return normalize_inits(d, args.cross_init_mode)
    raise ValueError(args.init_mode)

def set_requires_grad(model: Approx88CrossModel, train_scope: str):
    model.set_train_scope(train_scope)
    active = [n for n,p in model.named_parameters() if p.requires_grad]
    print(f"[train-scope] {train_scope}, trainable parameter tensors={len(active)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-name", default="approx88_cascade")
    ap.add_argument("--init-mode", choices=["json","default","random"], default="json")
    ap.add_argument("--base-inits-json", default="best_approx88_inits.json")
    ap.add_argument("--cross-init-mode", choices=["projected","approx62","zero"], default="approx62")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mred-denom", choices=["total","nonzero"], default="total")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--random-prob", type=float, default=0.5)
    # training
    ap.add_argument("--train-scope", choices=["low","cross","top","cross_top","low_top","all"], default="cross_top")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--init-p", type=float, default=0.95)
    ap.add_argument("--noise-std", type=float, default=0.01)
    ap.add_argument("--c-init", type=float, default=2.0)
    ap.add_argument("--c-out", type=float, default=2.0)
    ap.add_argument("--c-anneal", action="store_true")
    ap.add_argument("--zero-weight", type=float, default=0.01)
    ap.add_argument("--med-weight", type=float, default=0.0)
    ap.add_argument("--bin-weight", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--restart-from-best-every", type=int, default=0)
    ap.add_argument("--restart-init-p", type=float, default=0.96)
    ap.add_argument("--restart-noise-std", type=float, default=0.005)
    ap.add_argument("--restart-lr-decay", type=float, default=0.8)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    # constraints
    ap.add_argument("--max-wce", type=int, default=-1)
    ap.add_argument("--max-er", type=float, default=-1.0)
    ap.add_argument("--max-med", type=float, default=-1.0)
    # post/search
    ap.add_argument("--single-after", action="store_true")
    ap.add_argument("--single-only", action="store_true")
    ap.add_argument("--single-rounds", type=int, default=20)
    ap.add_argument("--single-mode", choices=["first","best"], default="best")
    ap.add_argument("--single-lut-names", default="all")
    ap.add_argument("--single-random-order", action="store_true")
    ap.add_argument("--pair-after", action="store_true")
    ap.add_argument("--pair-only", action="store_true")
    ap.add_argument("--pair-rounds", type=int, default=4)
    ap.add_argument("--pair-mode", choices=["first","best"], default="first")
    ap.add_argument("--pair-lut-names", default="cross")
    ap.add_argument("--pair-random-order", action="store_true")
    ap.add_argument("--pair-max-pairs", type=int, default=80000)
    ap.add_argument("--escape-only", action="store_true")
    ap.add_argument("--topk", type=int, default=120)
    ap.add_argument("--candidate-mode", choices=["all","top","neutral","top_neutral"], default="top_neutral")
    ap.add_argument("--neutral-top", type=int, default=180)
    ap.add_argument("--neutral-margin", type=float, default=0.004)
    ap.add_argument("--escape-iters", type=int, default=120)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--escape-single-rounds", type=int, default=8)
    ap.add_argument("--escape-pair-after", action="store_true")
    ap.add_argument("--escape-pair-rounds", type=int, default=2)
    ap.add_argument("--escape-pair-max-pairs", type=int, default=50000)
    ap.add_argument("--do-single", action="store_true")
    ap.add_argument("--do-pair", action="store_true")
    ap.add_argument("--do-escape", action="store_true")
    ap.add_argument("--do-final-single", action="store_true")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--log-file", default="terminal_log.txt")
    args = ap.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    log_handle = install_tee(out_dir / args.log_file)
    try:
        print(f"Log file: {out_dir/args.log_file}\nStage: {args.stage_name}\nArgs: {vars(args)}")
        print(f"Projected-accurate approx62 INIT: {PROJECTED62}")
        base = make_base(args)
        base_m = compute_hard_metrics(base, args.mred_denom)
        print(f"Base metrics: {metric_str(base_m)} mred_denom={args.mred_denom}")
        for n in TRAIN_NAMES:
            print(f"  {n:10s} = {base[n]}")
        best = {"stage": args.stage_name, "epoch": -1, "loss": None, "metrics": metrics_to_dict(base_m), "inits": base}
        best_mred = base_m.MRED
        write_best(out_dir, best, "best")

        if args.eval_only:
            print("[eval-only] done")
            return

        single_names = parse_lut_names(args.single_lut_names)
        pair_names = parse_lut_names(args.pair_lut_names)

        if args.single_only:
            bi,bm = greedy_single(base,args.mred_denom,args.single_rounds,args.single_mode,single_names,args.single_random_order,args.max_wce,args.max_er,args.max_med)
            best={"stage":args.stage_name+"+single_only","epoch":"single_only","loss":None,"metrics":metrics_to_dict(bm),"inits":bi}; write_best(out_dir,best,"best"); return
        if args.pair_only:
            bi,bm = greedy_pair(base,args.mred_denom,args.pair_rounds,args.pair_mode,pair_names,args.pair_random_order,args.pair_max_pairs,args.max_wce,args.max_er,args.max_med)
            best={"stage":args.stage_name+"+pair_only","epoch":"pair_only","loss":None,"metrics":metrics_to_dict(bm),"inits":bi}; write_best(out_dir,best,"best"); return

        if args.escape_only or args.do_single or args.do_pair or args.do_escape or args.do_final_single:
            scope_names = parse_lut_names(args.single_lut_names if args.single_lut_names != "all" else "all")
            all_bits = candidate_bits(scope_names)
            top_bits = top_error_candidate_bits(base,args.mred_denom,args.topk,scope_names)
            source = all_bits if args.candidate_mode == "all" else top_bits
            if args.candidate_mode in ["neutral","top_neutral"]:
                neutral = neutral_bits(base,args.mred_denom,source,args.neutral_top,args.neutral_margin,args.max_wce,args.max_er,args.max_med)
                source = list(dict.fromkeys(top_bits + neutral)) if args.candidate_mode == "top_neutral" else neutral
            cur=base; cur_m=base_m
            if args.do_single:
                cur,cur_m=greedy_single_bits(cur,args.mred_denom,args.single_rounds,args.single_mode,source,True,args.max_wce,args.max_er,args.max_med)
            if args.do_pair:
                cur,cur_m=greedy_pair_bits(cur,args.mred_denom,args.pair_rounds,args.pair_mode,source,args.pair_random_order,args.pair_max_pairs,args.max_wce,args.max_er,args.max_med)
            if args.do_escape or args.escape_only:
                neutral=neutral_bits(cur,args.mred_denom,source,args.neutral_top,args.neutral_margin,args.max_wce,args.max_er,args.max_med)
                cur,cur_m=escape_search(cur,args.mred_denom,source,neutral,args.escape_iters,args.kmin,args.kmax,args.escape_single_rounds,args.escape_pair_after,args.escape_pair_rounds,args.escape_pair_max_pairs,args.max_wce,args.max_er,args.max_med)
            if args.do_final_single:
                cur,cur_m=greedy_single(cur,args.mred_denom,args.single_rounds,args.single_mode,TRAIN_NAMES,False,args.max_wce,args.max_er,args.max_med)
            best={"stage":args.stage_name,"epoch":args.stage_name,"loss":None,"metrics":metrics_to_dict(cur_m),"inits":cur}; write_best(out_dir,best,"best"); return

        if args.epochs > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            a,b,exact = make_dataset(device)
            model = Approx88CrossModel(base,args.init_p,args.noise_std).to(device)
            set_requires_grad(model,args.train_scope)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            since=0
            for epoch in range(args.epochs):
                c_init=args.c_init; c_out=args.c_out
                if args.c_anneal:
                    scale=min(8.0,1.0+7.0*epoch/max(1,args.epochs-1)); c_init=args.c_init*scale; c_out=args.c_out*scale
                opt.zero_grad(set_to_none=True)
                approx = model(a,b,c_init=c_init,c_out=c_out,hard_middle=True)
                reg = model.bin_reg(c_init)
                loss,mred,zero,medn,binr = loss_fn(approx,exact,reg,args.zero_weight,args.med_weight,args.bin_weight,args.mred_denom)
                loss.backward()
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                opt.step()
                hard = model.hard_inits(args.c_init)
                hm = compute_hard_metrics(hard,args.mred_denom)
                ok = passes(hm,args.max_wce,args.max_er,args.max_med)
                improved = ok and hm.MRED < best_mred
                if improved:
                    best_mred = hm.MRED; best={"stage":args.stage_name,"epoch":epoch,"loss":float(loss.detach().cpu()),"metrics":metrics_to_dict(hm),"inits":hard}; write_best(out_dir,best,"best"); since=0
                else:
                    since += 1
                print(f"[epoch {epoch:05d}] lr={opt.param_groups[0]['lr']:.6g} loss={float(loss.detach().cpu()):.8f} train_mred={mred:.8f} zero={zero:.6f} med_norm={medn:.6f} bin={binr:.6f} hard_MRED={hm.MRED:.8f} MED={hm.MED:.4f} ER={hm.ER:.4f} WCE={hm.WCE} ok={int(ok)} best={best_mred:.8f}{' *BEST*' if improved else ''}")
                if args.restart_from_best_every>0 and since>=args.restart_from_best_every:
                    oldlr=opt.param_groups[0]['lr']; newlr=max(args.min_lr,oldlr*args.restart_lr_decay)
                    print(f"[restart] reload best after {since} epochs lr {oldlr:.6g}->{newlr:.6g}")
                    model = Approx88CrossModel(best["inits"],args.restart_init_p,args.restart_noise_std).to(device)
                    set_requires_grad(model,args.train_scope)
                    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=newlr)
                    since=0
        cur=best["inits"]; cur_m=compute_hard_metrics(cur,args.mred_denom)
        if args.single_after:
            cur,cur_m=greedy_single(cur,args.mred_denom,args.single_rounds,args.single_mode,single_names,args.single_random_order,args.max_wce,args.max_er,args.max_med)
        if args.pair_after:
            cur,cur_m=greedy_pair(cur,args.mred_denom,args.pair_rounds,args.pair_mode,pair_names,args.pair_random_order,args.pair_max_pairs,args.max_wce,args.max_er,args.max_med)
        if cur_m.MRED < best_mred or args.single_after or args.pair_after:
            best={"stage":args.stage_name+"+post","epoch":"post","loss":best.get("loss"),"metrics":metrics_to_dict(cur_m),"inits":cur}; write_best(out_dir,best,"best")
        print("\nFinished")
        print(f"Best {metric_str(compute_hard_metrics(best['inits'],args.mred_denom))}")
        print(f"Best JSON: {out_dir/'best_approx88_cascade_inits.json'}")
        print(f"Best Verilog: {out_dir/'best_approx88_cascade.v'}")
    finally:
        sys.stdout.flush(); sys.stderr.flush(); sys.stdout=sys.__stdout__; sys.stderr=sys.__stderr__; log_handle.flush(); log_handle.close()

if __name__ == "__main__":
    main()
