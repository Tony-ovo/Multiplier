#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next-step INIT search for 6x6 approximate multiplier.

Architecture:
  - approx62 LOW/MID/HIGH are unshared.
  - comp66 prod[4:5] and prod[6:7] use pair-aware LUT6 columns:
      u_comp4/u_comp5 both see {plow4,pmid2,phigh0,plow5,pmid3,phigh1}
      u_comp6/u_comp7 both see {plow6,pmid4,phigh2,plow7,pmid5,phigh3}
    This keeps four LUT6 resources but gives each output access to the adjacent column.
  - u_comp23 and u_comp89 remain LUT6_2.

Search extras:
  - single-bit flip, best/first improvement
  - pair-bit flip on selected LUTs
  - basin hopping: random multi-bit perturbation + single-bit local refinement

Terminal output is mirrored exactly to the txt log.
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
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Names/defaults
# ============================================================
DEFAULT_APPROX62 = {
    "lut1": "EAC00000A0A00000",
    "lut2": "EEAACC00EAC0EAC0",
    "lut3": "E6AACC006AC0EAC0",
    "lut4": "800000004C000000",
}
DEFAULT_COMP_OLD = {
    "u_or23": "0000FFF80000FEE6",
    "u_or4":  "00000000000000FE",
    "u_or5":  "00000000000000FE",
    "u_or6":  "00000000000000FE",
    "u_or7":  "00000000000000FE",
    "u_or89": "00005F5800005E4E",
}
SEGMENTS = ["low", "mid", "high"]
APPROX_IDS = ["lut1", "lut2", "lut3", "lut4"]
APPROX_NAMES = [f"{s}_{l}" for s in SEGMENTS for l in APPROX_IDS]
COMP_LUT6_2 = ["u_comp23", "u_comp89"]
COMP_LUT6 = ["u_comp4", "u_comp5", "u_comp6", "u_comp7"]
TRAIN_NAMES = APPROX_NAMES + COMP_LUT6_2 + COMP_LUT6

USED_BITS: Dict[str, List[int]] = {}
for seg in SEGMENTS:
    USED_BITS[f"{seg}_lut1"] = list(range(16, 32)) + list(range(48, 64))
    USED_BITS[f"{seg}_lut2"] = list(range(64))
    USED_BITS[f"{seg}_lut3"] = list(range(64))
    USED_BITS[f"{seg}_lut4"] = list(range(16, 32)) + list(range(48, 64))
USED_BITS["u_comp23"] = list(range(0, 16)) + list(range(32, 48))
USED_BITS["u_comp89"] = list(range(0, 16)) + list(range(32, 48))
for n in COMP_LUT6:
    USED_BITS[n] = list(range(64))

# ============================================================
# Logging
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

# ============================================================
# Utilities
# ============================================================
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

def lut6_table_int_from_old_lut3(old_hex: str, input_map: Tuple[int, int, int]) -> int:
    """Create a 6-input LUT int from an old 3-input LUT6 that used I0,I1,I2 and constants 0,0,0.
    input_map says which new input positions feed old I0/I1/I2.
    """
    old = int_to_bits(hex_int(old_hex))
    out = [0] * 64
    for addr in range(64):
        old_addr = 0
        for old_i, new_pos in enumerate(input_map):
            old_addr |= (((addr >> new_pos) & 1) << old_i)
        out[addr] = old[old_addr]
    return bits_to_int(out)

def normalize_inits(obj: Dict) -> Dict[str, str]:
    if "inits" in obj:
        obj = obj["inits"]
    out: Dict[str, str] = {}
    # approx LUTs: explicit unshared > old shared > default
    for seg in SEGMENTS:
        for lut in APPROX_IDS:
            nk = f"{seg}_{lut}"
            if nk in obj:
                out[nk] = int_to_hex(hex_int(obj[nk]))
            elif lut in obj:
                out[nk] = int_to_hex(hex_int(obj[lut]))
            else:
                out[nk] = int_to_hex(hex_int(DEFAULT_APPROX62[lut]))
    # comp23/89: explicit new > old name > default
    if "u_comp23" in obj:
        out["u_comp23"] = int_to_hex(hex_int(obj["u_comp23"]))
    elif "u_or23" in obj:
        out["u_comp23"] = int_to_hex(hex_int(obj["u_or23"]))
    else:
        out["u_comp23"] = int_to_hex(hex_int(DEFAULT_COMP_OLD["u_or23"]))

    if "u_comp89" in obj:
        out["u_comp89"] = int_to_hex(hex_int(obj["u_comp89"]))
    elif "u_or89" in obj:
        out["u_comp89"] = int_to_hex(hex_int(obj["u_or89"]))
    else:
        out["u_comp89"] = int_to_hex(hex_int(DEFAULT_COMP_OLD["u_or89"]))

    # pair-aware comp LUT6s.
    # u_comp4 depends on new inputs [plow4,pmid2,phigh0,plow5,pmid3,phigh1], old u_or4 used positions 0,1,2.
    # u_comp5 should initially reproduce old u_or5, which depends on positions 3,4,5.
    # u_comp6 depends on [plow6,pmid4,phigh2,plow7,pmid5,phigh3], old u_or6 positions 0,1,2.
    # u_comp7 initially reproduces old u_or7 positions 3,4,5.
    default_old = DEFAULT_COMP_OLD
    if "u_comp4" in obj:
        out["u_comp4"] = int_to_hex(hex_int(obj["u_comp4"]))
    else:
        out["u_comp4"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or4", default_old["u_or4"]), (0, 1, 2)))
    if "u_comp5" in obj:
        out["u_comp5"] = int_to_hex(hex_int(obj["u_comp5"]))
    else:
        out["u_comp5"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or5", default_old["u_or5"]), (3, 4, 5)))
    if "u_comp6" in obj:
        out["u_comp6"] = int_to_hex(hex_int(obj["u_comp6"]))
    else:
        out["u_comp6"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or6", default_old["u_or6"]), (0, 1, 2)))
    if "u_comp7" in obj:
        out["u_comp7"] = int_to_hex(hex_int(obj["u_comp7"]))
    else:
        out["u_comp7"] = int_to_hex(lut6_table_int_from_old_lut3(obj.get("u_or7", default_old["u_or7"]), (3, 4, 5)))
    return out

def load_json_inits(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return normalize_inits(obj)

def default_inits() -> Dict[str, str]:
    return normalize_inits({})

def random_inits(p: float) -> Dict[str, str]:
    d = {}
    for n in TRAIN_NAMES:
        bits = [1 if random.random() < p else 0 for _ in range(64)]
        d[n] = bits_to_hex(bits)
    return normalize_inits(d)

def inits_to_ints(inits: Dict[str, str]) -> Dict[str, int]:
    return {k: hex_int(v) for k, v in normalize_inits(inits).items()}

def ints_to_inits(ints: Dict[str, int]) -> Dict[str, str]:
    return normalize_inits({k: int_to_hex(v) for k, v in ints.items()})

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
# Torch LUTs/model
# ============================================================
class TrainableLUT6_2(nn.Module):
    def __init__(self, init_hex: str, init_p: float, noise_std: float):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_p, noise_std)
    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_p: float, noise_std: float):
        bits = torch.tensor(int_to_bits(hex_int(init_hex)), dtype=torch.float32, device=self.logits.device)
        p1 = min(max(init_p, 1e-4), 1 - 1e-4)
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
    def forward(self, I0,I1,I2,I3,I4,I5, *, c_init, c_out, hard_middle=True):
        table = self.table_prob(c_init)
        if hard_middle:
            I0,I1,I2,I3,I4,I5 = [ste_binarize(x) for x in [I0,I1,I2,I3,I4,I5]]
        o5 = self.soft_lut([I0,I1,I2,I3,I4], table[:32])
        o6 = self.soft_lut([I0,I1,I2,I3,I4,I5], table)
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
        p1 = min(max(init_p, 1e-4), 1 - 1e-4)
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
    def forward(self, I0,I1,I2,I3,I4,I5, *, c_init, c_out, hard_middle=True):
        table = self.table_prob(c_init)
        if hard_middle:
            I0,I1,I2,I3,I4,I5 = [ste_binarize(x) for x in [I0,I1,I2,I3,I4,I5]]
        o = self.soft_lut([I0,I1,I2,I3,I4,I5], table)
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

class Approx66PairComp(nn.Module):
    def __init__(self, base_inits: Dict[str, str], init_p: float, noise_std: float):
        super().__init__()
        base = normalize_inits(base_inits)
        self.lut62 = nn.ModuleDict({n: TrainableLUT6_2(base[n], init_p, noise_std) for n in APPROX_NAMES + COMP_LUT6_2})
        self.lut6 = nn.ModuleDict({n: TrainableLUT6(base[n], init_p, noise_std) for n in COMP_LUT6})
    @torch.no_grad()
    def reset_from_inits(self, inits, init_p, noise_std):
        base = normalize_inits(inits)
        for n in APPROX_NAMES + COMP_LUT6_2:
            self.lut62[n].reset_from_hex(base[n], init_p, noise_std)
        for n in COMP_LUT6:
            self.lut6[n].reset_from_hex(base[n], init_p, noise_std)
    @staticmethod
    def const(x, v):
        return torch.full_like(x, float(v))
    def approx62(self, seg, a, b2, *, c_init, c_out, hard_middle):
        ab = int_bits_t(a, 6)
        bb = int_bits_t(b2, 2)
        one = self.const(ab[0], 1.0)
        p0,p1 = self.lut62[f"{seg}_lut1"](bb[0],bb[1],ab[0],ab[1],one,one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p2,p3 = self.lut62[f"{seg}_lut2"](bb[0],bb[1],ab[1],ab[2],ab[3],one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p4,p5 = self.lut62[f"{seg}_lut3"](bb[0],bb[1],ab[3],ab[4],ab[5],one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        p6,p7 = self.lut62[f"{seg}_lut4"](bb[0],bb[1],ab[4],ab[5],one,one,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        return [p0,p1,p2,p3,p4,p5,p6,p7]
    def comp(self, plow, pmid, phigh, *, c_init, c_out, hard_middle):
        z = self.const(plow[0], 0.0)
        o = self.const(plow[0], 1.0)
        prod = [z for _ in range(12)]
        prod[0], prod[1], prod[10], prod[11] = plow[0], plow[1], phigh[6], phigh[7]
        prod[2], prod[3] = self.lut62["u_comp23"](plow[2],pmid[0],plow[3],pmid[1],z,o,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[4] = self.lut6["u_comp4"](plow[4],pmid[2],phigh[0],plow[5],pmid[3],phigh[1],c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[5] = self.lut6["u_comp5"](plow[4],pmid[2],phigh[0],plow[5],pmid[3],phigh[1],c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[6] = self.lut6["u_comp6"](plow[6],pmid[4],phigh[2],plow[7],pmid[5],phigh[3],c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[7] = self.lut6["u_comp7"](plow[6],pmid[4],phigh[2],plow[7],pmid[5],phigh[3],c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        prod[8], prod[9] = self.lut62["u_comp89"](pmid[6],phigh[4],pmid[7],phigh[5],z,o,c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        return prod
    def forward(self, a, b, *, c_init, c_out, hard_middle=True):
        plow = self.approx62("low", a, b & 3, c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        pmid = self.approx62("mid", a, (b >> 2) & 3, c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        phigh = self.approx62("high", a, (b >> 4) & 3, c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        bits = self.comp(plow, pmid, phigh, c_init=c_init,c_out=c_out,hard_middle=hard_middle)
        s = torch.stack(bits, dim=1)
        w = torch.tensor([1 << i for i in range(12)], device=s.device, dtype=torch.float32)
        return torch.sum(s * w, dim=1), s
    def hard_inits(self, c_init):
        out = {}
        for n in APPROX_NAMES + COMP_LUT6_2:
            out[n] = self.lut62[n].hard_hex(c_init)
        for n in COMP_LUT6:
            out[n] = self.lut6[n].hard_hex(c_init)
        return normalize_inits(out)
    def bin_reg(self, c_init):
        regs = []
        for n in APPROX_NAMES + COMP_LUT6_2:
            regs.append(self.lut62[n].bin_reg(c_init))
        for n in COMP_LUT6:
            regs.append(self.lut6[n].bin_reg(c_init))
        return torch.mean(torch.stack(regs))

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

class FastEvaluator:
    def __init__(self):
        A, B = np.meshgrid(np.arange(64, dtype=np.int64), np.arange(64, dtype=np.int64), indexing="ij")
        self.a = A.reshape(-1)
        self.b = B.reshape(-1)
        self.exact = (self.a * self.b).astype(np.int64)
        self.mask = self.exact > 0
        self.ab = [((self.a >> i) & 1).astype(np.uint8) for i in range(6)]
        self.lowb = [((self.b >> i) & 1).astype(np.uint8) for i in range(2)]
        self.midb = [(((self.b >> 2) >> i) & 1).astype(np.uint8) for i in range(2)]
        self.highb = [(((self.b >> 4) >> i) & 1).astype(np.uint8) for i in range(2)]
        self.z = np.zeros_like(self.ab[0], dtype=np.uint8)
        self.o = np.ones_like(self.ab[0], dtype=np.uint8)
        self.weights = np.array([1 << i for i in range(12)], dtype=np.int64)
        self.bits_cache: Dict[Tuple[Tuple[str,int],...], Dict[str,np.ndarray]] = {}
    @staticmethod
    def bits(v: int) -> np.ndarray:
        return np.array([(int(v) >> i) & 1 for i in range(64)], dtype=np.uint8)
    def tables(self, ints: Dict[str, int]) -> Dict[str, np.ndarray]:
        # no cache by default; cache can grow too large during pair scans
        return {n: self.bits(ints[n]) for n in TRAIN_NAMES}
    @staticmethod
    def lut62(tab, I0,I1,I2,I3,I4,I5):
        addr5 = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4)
        addr6 = addr5 + (I5 << 5)
        return tab[addr5], tab[addr6]
    @staticmethod
    def lut6(tab, I0,I1,I2,I3,I4,I5):
        addr = I0 + (I1 << 1) + (I2 << 2) + (I3 << 3) + (I4 << 4) + (I5 << 5)
        return tab[addr]
    def approx62(self, seg, bbits, tabs):
        ab, bb, o = self.ab, bbits, self.o
        p0,p1 = self.lut62(tabs[f"{seg}_lut1"], bb[0],bb[1],ab[0],ab[1],o,o)
        p2,p3 = self.lut62(tabs[f"{seg}_lut2"], bb[0],bb[1],ab[1],ab[2],ab[3],o)
        p4,p5 = self.lut62(tabs[f"{seg}_lut3"], bb[0],bb[1],ab[3],ab[4],ab[5],o)
        p6,p7 = self.lut62(tabs[f"{seg}_lut4"], bb[0],bb[1],ab[4],ab[5],o,o)
        return [p0,p1,p2,p3,p4,p5,p6,p7]
    def evaluate_ints(self, ints: Dict[str, int]) -> Metrics:
        tabs = self.tables(ints)
        plow = self.approx62("low", self.lowb, tabs)
        pmid = self.approx62("mid", self.midb, tabs)
        phigh = self.approx62("high", self.highb, tabs)
        prod = [None] * 12
        prod[0], prod[1], prod[10], prod[11] = plow[0], plow[1], phigh[6], phigh[7]
        prod[2], prod[3] = self.lut62(tabs["u_comp23"], plow[2],pmid[0],plow[3],pmid[1],self.z,self.o)
        prod[4] = self.lut6(tabs["u_comp4"], plow[4],pmid[2],phigh[0],plow[5],pmid[3],phigh[1])
        prod[5] = self.lut6(tabs["u_comp5"], plow[4],pmid[2],phigh[0],plow[5],pmid[3],phigh[1])
        prod[6] = self.lut6(tabs["u_comp6"], plow[6],pmid[4],phigh[2],plow[7],pmid[5],phigh[3])
        prod[7] = self.lut6(tabs["u_comp7"], plow[6],pmid[4],phigh[2],plow[7],pmid[5],phigh[3])
        prod[8], prod[9] = self.lut62(tabs["u_comp89"], pmid[6],phigh[4],pmid[7],phigh[5],self.z,self.o)
        val = np.zeros_like(self.exact)
        for i, p in enumerate(prod):
            val += p.astype(np.int64) << i
        err = np.abs(val - self.exact)
        total = err.size
        err_cases = int(np.count_nonzero(err))
        med = float(err.mean())
        wce = int(err.max())
        mred = float(np.sum(err[self.mask] / self.exact[self.mask]) / total)
        return Metrics(total, err_cases, err_cases / total, med, med / (63 * 63), mred, wce)
    def evaluate(self, inits: Dict[str, str]) -> Metrics:
        return self.evaluate_ints(inits_to_ints(inits))

EVAL = FastEvaluator()

def compute_hard_metrics(inits: Dict[str, str]) -> Metrics:
    return EVAL.evaluate(inits)

# ============================================================
# Output Verilog/JSON
# ============================================================
def write_best(out_dir: Path, best: Dict, prefix="best"):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}_approx66_inits.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False, allow_nan=False)
    if prefix == "best":
        (out_dir / "best_approx66_inits.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    inits = normalize_inits(best["inits"])
    vp = out_dir / f"{prefix}_approx66_unshared_paircomp.v"
    with vp.open("w", encoding="utf-8") as f:
        f.write("// Generated by train_approx66_unshared_paircomp.py\n")
        f.write(f"// MRED={best['metrics']['MRED']:.10f} MED={best['metrics']['MED']:.10f} ER={best['metrics']['ER']:.10f} WCE={best['metrics']['WCE']}\n\n")
        f.write("module approx66_unshared_paircomp(input wire [5:0] a, input wire [5:0] b, output wire [11:0] prod);\n")
        f.write("wire [7:0] plow, pmid, phigh;\n")
        f.write("approx62_low  U_LOW (.a(a), .b(b[1:0]), .prod(plow));\n")
        f.write("approx62_mid  U_MID (.a(a), .b(b[3:2]), .prod(pmid));\n")
        f.write("approx62_high U_HIGH(.a(a), .b(b[5:4]), .prod(phigh));\n")
        f.write("comp66_pair U_COMP(.plow(plow), .pmid(pmid), .phigh(phigh), .prod(prod));\nendmodule\n\n")
        for seg in SEGMENTS:
            f.write(f"module approx62_{seg}(input wire [5:0] a, input wire [1:0] b, output wire [7:0] prod);\n")
            f.write(f"LUT6_2 #(.INIT({inits[f'{seg}_lut1']})) LUT6_inst1(.I0(b[0]),.I1(b[1]),.I2(a[0]),.I3(a[1]),.I4(1'b1),.I5(1'b1),.O5(prod[0]),.O6(prod[1]));\n")
            f.write(f"LUT6_2 #(.INIT({inits[f'{seg}_lut2']})) LUT6_inst2(.I0(b[0]),.I1(b[1]),.I2(a[1]),.I3(a[2]),.I4(a[3]),.I5(1'b1),.O5(prod[2]),.O6(prod[3]));\n")
            f.write(f"LUT6_2 #(.INIT({inits[f'{seg}_lut3']})) LUT6_inst3(.I0(b[0]),.I1(b[1]),.I2(a[3]),.I3(a[4]),.I4(a[5]),.I5(1'b1),.O5(prod[4]),.O6(prod[5]));\n")
            f.write(f"LUT6_2 #(.INIT({inits[f'{seg}_lut4']})) LUT6_inst4(.I0(b[0]),.I1(b[1]),.I2(a[4]),.I3(a[5]),.I4(1'b1),.I5(1'b1),.O5(prod[6]),.O6(prod[7]));\n")
            f.write("endmodule\n\n")
        f.write("module comp66_pair(input wire [7:0] plow, input wire [7:0] pmid, input wire [7:0] phigh, output wire [11:0] prod);\n")
        f.write("assign prod[0]=plow[0]; assign prod[1]=plow[1]; assign prod[10]=phigh[6]; assign prod[11]=phigh[7];\n")
        f.write(f"LUT6_2 #(.INIT({inits['u_comp23']})) u_comp23(.I0(plow[2]),.I1(pmid[0]),.I2(plow[3]),.I3(pmid[1]),.I4(1'b0),.I5(1'b1),.O5(prod[2]),.O6(prod[3]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_comp4']})) u_comp4(.I0(plow[4]),.I1(pmid[2]),.I2(phigh[0]),.I3(plow[5]),.I4(pmid[3]),.I5(phigh[1]),.O(prod[4]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_comp5']})) u_comp5(.I0(plow[4]),.I1(pmid[2]),.I2(phigh[0]),.I3(plow[5]),.I4(pmid[3]),.I5(phigh[1]),.O(prod[5]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_comp6']})) u_comp6(.I0(plow[6]),.I1(pmid[4]),.I2(phigh[2]),.I3(plow[7]),.I4(pmid[5]),.I5(phigh[3]),.O(prod[6]));\n")
        f.write(f"LUT6 #(.INIT({inits['u_comp7']})) u_comp7(.I0(plow[6]),.I1(pmid[4]),.I2(phigh[2]),.I3(plow[7]),.I4(pmid[5]),.I5(phigh[3]),.O(prod[7]));\n")
        f.write(f"LUT6_2 #(.INIT({inits['u_comp89']})) u_comp89(.I0(pmid[6]),.I1(phigh[4]),.I2(pmid[7]),.I3(phigh[5]),.I4(1'b0),.I5(1'b1),.O5(prod[8]),.O6(prod[9]));\n")
        f.write("endmodule\n")
    if prefix == "best":
        (out_dir / "best_approx66_unshared_paircomp.v").write_text(vp.read_text(encoding="utf-8"), encoding="utf-8")

# ============================================================
# Loss/dataset
# ============================================================
def make_dataset(device):
    aa, bb, ex = [], [], []
    for a in range(64):
        for b in range(64):
            aa.append(a); bb.append(b); ex.append(a * b)
    return torch.tensor(aa, dtype=torch.long, device=device), torch.tensor(bb, dtype=torch.long, device=device), torch.tensor(ex, dtype=torch.float32, device=device)

def loss_fn(approx, exact, bin_reg, zero_weight, med_weight, bin_weight):
    err = torch.abs(approx - exact)
    mask = exact > 0
    mred = torch.sum(err[mask] / exact[mask]) / float(exact.numel())
    zero = torch.mean(err[~mask]) / 4096.0 if torch.any(~mask) else torch.tensor(0.0, device=approx.device)
    medn = torch.mean(err) / 4096.0
    loss = mred + zero_weight * zero + med_weight * medn + bin_weight * bin_reg
    return loss, float(mred.detach().cpu()), float(zero.detach().cpu()), float(medn.detach().cpu()), float(bin_reg.detach().cpu())

# ============================================================
# Bit search helpers
# ============================================================
def parse_lut_names(s: str) -> List[str]:
    if not s or s.lower() == "all":
        return TRAIN_NAMES
    names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [n for n in names if n not in TRAIN_NAMES]
    if bad:
        raise ValueError(f"Unknown LUT names in --pair-lut-names/--single-lut-names: {bad}")
    return names

def candidate_bits(names: List[str], random_order=False) -> List[Tuple[str, int]]:
    out = [(n, b) for n in names for b in USED_BITS[n]]
    if random_order:
        random.shuffle(out)
    return out

def eval_ints(ints):
    return EVAL.evaluate_ints(ints)

def greedy_single(start: Dict[str, str], rounds: int, mode: str, names: List[str], random_order: bool, eps=1e-12):
    cur = inits_to_ints(start)
    cur_m = eval_ints(cur)
    print(f"\n[single] start MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE} mode={mode} names={','.join(names)}")
    for r in range(rounds):
        bits = candidate_bits(names, random_order)
        improved = False
        print(f"[single] round {r+1}/{rounds} begin candidates={len(bits)}")
        if mode == "best":
            best = None; best_m = cur_m; old = cur_m.MRED
            for n,b in bits:
                trial = dict(cur); trial[n] ^= (1 << b)
                m = eval_ints(trial)
                if m.MRED + eps < best_m.MRED:
                    best, best_m = (n,b,trial), m
            if best:
                n,b,cur = best; cur_m = best_m; improved=True
                print(f"[single] KEEP-BEST lut={n} bit={b:02d} MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
        else:
            for n,b in bits:
                trial = dict(cur); trial[n] ^= (1 << b)
                m = eval_ints(trial)
                if m.MRED + eps < cur_m.MRED:
                    old=cur_m.MRED; cur=trial; cur_m=m; improved=True
                    print(f"[single] KEEP lut={n} bit={b:02d} MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
        if not improved:
            print(f"[single] round {r+1} no improvement, stop")
            break
    print(f"[single] final MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
    return ints_to_inits(cur), cur_m

def greedy_pair(start: Dict[str, str], rounds: int, mode: str, names: List[str], random_order: bool, max_pairs: int, eps=1e-12):
    cur = inits_to_ints(start)
    cur_m = eval_ints(cur)
    cand = candidate_bits(names, random_order=False)
    pair_count = len(cand) * (len(cand) - 1) // 2
    print(f"\n[pair] start MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE} mode={mode} names={','.join(names)} bits={len(cand)} pairs={pair_count}")
    for r in range(rounds):
        order = [(i,j) for i in range(len(cand)) for j in range(i+1, len(cand))]
        if random_order:
            random.shuffle(order)
        if max_pairs > 0 and len(order) > max_pairs:
            order = order[:max_pairs]
        improved = False
        print(f"[pair] round {r+1}/{rounds} begin trial_pairs={len(order)}")
        if mode == "best":
            best=None; best_m=cur_m; old=cur_m.MRED
            for i,j in order:
                n1,b1 = cand[i]; n2,b2 = cand[j]
                trial = dict(cur)
                trial[n1] ^= (1 << b1)
                trial[n2] ^= (1 << b2)
                m = eval_ints(trial)
                if m.MRED + eps < best_m.MRED:
                    best=(n1,b1,n2,b2,trial); best_m=m
            if best:
                n1,b1,n2,b2,cur = best; cur_m=best_m; improved=True
                print(f"[pair] KEEP-BEST ({n1},{b1:02d})+({n2},{b2:02d}) MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
        else:
            for i,j in order:
                n1,b1 = cand[i]; n2,b2 = cand[j]
                trial = dict(cur)
                trial[n1] ^= (1 << b1)
                trial[n2] ^= (1 << b2)
                m = eval_ints(trial)
                if m.MRED + eps < cur_m.MRED:
                    old=cur_m.MRED; cur=trial; cur_m=m; improved=True
                    print(f"[pair] KEEP ({n1},{b1:02d})+({n2},{b2:02d}) MRED {old:.10f}->{cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
                    if mode == "first":
                        break
        if not improved:
            print(f"[pair] round {r+1} no improvement, stop")
            break
    print(f"[pair] final MRED={cur_m.MRED:.10f} MED={cur_m.MED:.4f} ER={cur_m.ER:.4f} WCE={cur_m.WCE}")
    return ints_to_inits(cur), cur_m

def basin_hop(start: Dict[str, str], iters: int, flip_min: int, flip_max: int, names: List[str], single_rounds: int, single_mode: str, random_order: bool):
    best = normalize_inits(start)
    best_m = compute_hard_metrics(best)
    cand = candidate_bits(names, random_order=False)
    print(f"\n[basin] start MRED={best_m.MRED:.10f} iters={iters} random_flips=[{flip_min},{flip_max}] candidates={len(cand)}")
    for it in range(iters):
        ints = inits_to_ints(best)
        k = random.randint(flip_min, flip_max)
        chosen = random.sample(cand, min(k, len(cand)))
        for n,b in chosen:
            ints[n] ^= (1 << b)
        pert = ints_to_inits(ints)
        pert_m = compute_hard_metrics(pert)
        print(f"[basin] iter={it+1}/{iters} perturb k={k} MRED={pert_m.MRED:.10f}")
        loc, loc_m = greedy_single(pert, single_rounds, single_mode, names, random_order)
        if loc_m.MRED < best_m.MRED:
            old = best_m.MRED; best, best_m = loc, loc_m
            print(f"[basin] ACCEPT iter={it+1} MRED {old:.10f}->{best_m.MRED:.10f} MED={best_m.MED:.4f} ER={best_m.ER:.4f} WCE={best_m.WCE}")
        else:
            print(f"[basin] reject iter={it+1} local_MRED={loc_m.MRED:.10f} best={best_m.MRED:.10f}")
    print(f"[basin] final MRED={best_m.MRED:.10f} MED={best_m.MED:.4f} ER={best_m.ER:.4f} WCE={best_m.WCE}")
    return best, best_m

# ============================================================
# Main
# ============================================================
def make_base(args):
    if args.init_mode == "json":
        if not args.base_inits_json:
            raise ValueError("--init-mode json requires --base-inits-json")
        return load_json_inits(Path(args.base_inits_json))
    if args.init_mode == "manual":
        return default_inits()
    return random_inits(args.random_init_prob)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--init-mode", choices=["json","manual","random"], default="json")
    p.add_argument("--base-inits-json", default="")
    p.add_argument("--random-init-prob", type=float, default=0.5)
    p.add_argument("--init-p", type=float, default=0.92)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--c-init", type=float, default=2.0)
    p.add_argument("--c-out", type=float, default=2.0)
    p.add_argument("--c-anneal", action="store_true")
    p.add_argument("--zero-weight", type=float, default=0.01)
    p.add_argument("--med-weight", type=float, default=0.0)
    p.add_argument("--bin-weight", type=float, default=2e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--restart-from-best-every", type=int, default=150)
    p.add_argument("--restart-init-p", type=float, default=0.94)
    p.add_argument("--restart-noise-std", type=float, default=0.015)
    p.add_argument("--restart-lr-decay", type=float, default=0.8)
    p.add_argument("--min-lr", type=float, default=1e-5)

    p.add_argument("--single-after", action="store_true")
    p.add_argument("--single-only", action="store_true")
    p.add_argument("--single-rounds", type=int, default=20)
    p.add_argument("--single-mode", choices=["first","best"], default="best")
    p.add_argument("--single-random-order", action="store_true")
    p.add_argument("--single-lut-names", default="all")

    p.add_argument("--pair-after", action="store_true")
    p.add_argument("--pair-only", action="store_true")
    p.add_argument("--pair-rounds", type=int, default=5)
    p.add_argument("--pair-mode", choices=["first","best"], default="first")
    p.add_argument("--pair-random-order", action="store_true")
    p.add_argument("--pair-lut-names", default="u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut3,mid_lut3,high_lut3")
    p.add_argument("--pair-max-pairs", type=int, default=100000)

    p.add_argument("--basin-after", action="store_true")
    p.add_argument("--basin-only", action="store_true")
    p.add_argument("--basin-iters", type=int, default=20)
    p.add_argument("--basin-flip-min", type=int, default=2)
    p.add_argument("--basin-flip-max", type=int, default=5)
    p.add_argument("--basin-lut-names", default="u_comp4,u_comp5,u_comp6,u_comp7,u_comp23,u_comp89,low_lut3,mid_lut3,high_lut3")
    p.add_argument("--basin-single-rounds", type=int, default=8)

    p.add_argument("--out-dir", default=".")
    p.add_argument("--log-file", default="terminal_log.txt")
    p.add_argument("--stage-name", default="unshared_paircomp")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    log_handle = install_tee(out_dir / args.log_file)
    try:
        print(f"Log file: {out_dir / args.log_file}")
        print(f"Stage: {args.stage_name}")
        print(f"Args: {vars(args)}")
        base = make_base(args)
        base_m = compute_hard_metrics(base)
        print(f"Base metrics: MRED={base_m.MRED:.10f} MED={base_m.MED:.6f} ER={base_m.ER:.6f} WCE={base_m.WCE}")
        print(f"Trainable LUT count={len(TRAIN_NAMES)} used_bits={sum(len(USED_BITS[n]) for n in TRAIN_NAMES)}")
        for n in TRAIN_NAMES:
            print(f"  {n:10s} = {base[n]}")
        best = {"stage": args.stage_name, "epoch": -1, "loss": None, "metrics": metrics_to_dict(base_m), "inits": base}
        best_mred = base_m.MRED
        write_best(out_dir, best, "best")

        single_names = parse_lut_names(args.single_lut_names)
        pair_names = parse_lut_names(args.pair_lut_names)
        basin_names = parse_lut_names(args.basin_lut_names)

        if args.single_only:
            bi, bm = greedy_single(base, args.single_rounds, args.single_mode, single_names, args.single_random_order)
            best = {"stage": args.stage_name + "+single_only", "epoch": "single_only", "loss": None, "metrics": metrics_to_dict(bm), "inits": bi}
            write_best(out_dir, best, "best")
            return
        if args.pair_only:
            bi, bm = greedy_pair(base, args.pair_rounds, args.pair_mode, pair_names, args.pair_random_order, args.pair_max_pairs)
            best = {"stage": args.stage_name + "+pair_only", "epoch": "pair_only", "loss": None, "metrics": metrics_to_dict(bm), "inits": bi}
            write_best(out_dir, best, "best")
            return
        if args.basin_only:
            bi, bm = basin_hop(base, args.basin_iters, args.basin_flip_min, args.basin_flip_max, basin_names, args.basin_single_rounds, args.single_mode, True)
            best = {"stage": args.stage_name + "+basin_only", "epoch": "basin_only", "loss": None, "metrics": metrics_to_dict(bm), "inits": bi}
            write_best(out_dir, best, "best")
            return

        device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
        print(f"Device: {device}")
        model = Approx66PairComp(base, args.init_p, args.noise_std).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        a,b,exact = make_dataset(device)
        since = 0
        print("\nTraining begin")
        for ep in range(args.epochs):
            if args.c_anneal:
                t = 0 if args.epochs <= 1 else ep / (args.epochs - 1)
                c_init = 1.0 + t * (args.c_init - 1.0)
                c_out = 1.0 + t * (args.c_out - 1.0)
            else:
                c_init, c_out = args.c_init, args.c_out
            model.train(); opt.zero_grad(set_to_none=True)
            approx,_ = model(a,b,c_init=c_init,c_out=c_out,hard_middle=True)
            binreg = model.bin_reg(c_init)
            loss, tr_mred, zero, medn, br = loss_fn(approx, exact, binreg, args.zero_weight, args.med_weight, args.bin_weight)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            cur_inits = model.hard_inits(c_init)
            hm = compute_hard_metrics(cur_inits)
            improved = hm.MRED < best_mred
            if improved:
                best_mred = hm.MRED
                best = {"stage": args.stage_name, "epoch": ep, "loss": float(loss.detach().cpu()), "metrics": metrics_to_dict(hm), "inits": cur_inits}
                write_best(out_dir, best, "best")
                since = 0
            else:
                since += 1
            lr = float(opt.param_groups[0]["lr"])
            print(f"[epoch {ep:05d}] lr={lr:.6g} loss={float(loss.detach().cpu()):.8f} train_mred={tr_mred:.8f} zero={zero:.6f} med_norm={medn:.6f} bin={br:.6f} hard_MRED={hm.MRED:.8f} MED={hm.MED:.4f} ER={hm.ER:.4f} WCE={hm.WCE} best={best_mred:.8f}{' *BEST*' if improved else ''}")
            if args.restart_from_best_every > 0 and since >= args.restart_from_best_every:
                oldlr = float(opt.param_groups[0]["lr"])
                newlr = max(args.min_lr, oldlr * args.restart_lr_decay)
                print(f"[restart] no best for {since} epochs; reload best, lr {oldlr:.6g}->{newlr:.6g}")
                model.reset_from_inits(best["inits"], args.restart_init_p, args.restart_noise_std)
                opt = torch.optim.Adam(model.parameters(), lr=newlr)
                since = 0

        if args.single_after:
            bi,bm = greedy_single(best["inits"], args.single_rounds, args.single_mode, single_names, args.single_random_order)
            if bm.MRED < best_mred:
                best_mred = bm.MRED
                best = {"stage": args.stage_name + "+single", "epoch": "single", "loss": best.get("loss"), "metrics": metrics_to_dict(bm), "inits": bi}
                write_best(out_dir, best, "best")
        if args.pair_after:
            bi,bm = greedy_pair(best["inits"], args.pair_rounds, args.pair_mode, pair_names, args.pair_random_order, args.pair_max_pairs)
            if bm.MRED < best_mred:
                best_mred = bm.MRED
                best = {"stage": args.stage_name + "+pair", "epoch": "pair", "loss": best.get("loss"), "metrics": metrics_to_dict(bm), "inits": bi}
                write_best(out_dir, best, "best")
        if args.basin_after:
            bi,bm = basin_hop(best["inits"], args.basin_iters, args.basin_flip_min, args.basin_flip_max, basin_names, args.basin_single_rounds, args.single_mode, True)
            if bm.MRED < best_mred:
                best_mred = bm.MRED
                best = {"stage": args.stage_name + "+basin", "epoch": "basin", "loss": best.get("loss"), "metrics": metrics_to_dict(bm), "inits": bi}
                write_best(out_dir, best, "best")

        write_best(out_dir, best, "best")
        print("\nFinished")
        print(f"Best MRED={best['metrics']['MRED']:.10f} MED={best['metrics']['MED']:.6f} ER={best['metrics']['ER']:.6f} WCE={best['metrics']['WCE']}")
        print(f"Best JSON: {out_dir / 'best_approx66_inits.json'}")
        print(f"Best Verilog: {out_dir / 'best_approx66_unshared_paircomp.v'}")
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        log_handle.flush(); log_handle.close()

if __name__ == "__main__":
    main()
