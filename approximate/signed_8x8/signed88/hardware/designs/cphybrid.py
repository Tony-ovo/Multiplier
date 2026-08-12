from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from ...common import hex_to_int
from ...lut import TrainableLUT6_2, int_bits_t, lut62_np
from ..base import BaseDesign, DesignSpec, Low6Core, RtlBinding
from .common import BooleanCoreMixin

CP_BASE = {
    'cp_lut01': "64'h6AC06AC0A0A0A0A0",
    'cp_lut23': "64'hE62A4C006A40EAC0",
    'cp_lut45': "64'hE62A4C80EA40EA40",
    'cp_lut67': "64'h88800000444C8000",
}

EDGE_REACH = tuple(list(range(16, 32)) + list(range(48, 64)))
CP_D3_EDGE = tuple(i for i in EDGE_REACH if (i & 3) == 3)
CP_D3_ALL = tuple(i for i in range(64) if (i & 3) == 3)
CP_MUTABLE = {
    'cp_lut01': CP_D3_EDGE,
    'cp_lut23': CP_D3_ALL,
    'cp_lut45': CP_D3_ALL,
    'cp_lut67': CP_D3_ALL,
}


class CPHybridCore(Low6Core, BooleanCoreMixin):
    def __init__(self, inits: Mapping[str, str], mutable_bits, approx_mask: int, init_conf: float, noise_std: float):
        super().__init__()
        self.approx_mask = int(approx_mask)
        self.tables = nn.ModuleDict({
            name: TrainableLUT6_2(inits[name], mutable_bits[name], init_conf, noise_std)
            for name in CP_BASE
        })

    def cp62(self, a: torch.Tensor, digit: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool):
        ab = int_bits_t(a, 6)
        db = int_bits_t(digit, 2)
        one = torch.ones_like(ab[0])
        p0, p1 = self.tables['cp_lut01'](db[0], db[1], ab[0], ab[1], one, one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p2, p3 = self.tables['cp_lut23'](db[0], db[1], ab[1], ab[2], ab[3], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p4, p5 = self.tables['cp_lut45'](db[0], db[1], ab[3], ab[4], ab[5], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p6, p7 = self.tables['cp_lut67'](db[0], db[1], ab[3], ab[4], ab[5], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return [p0, p1, p2, p3, p4, p5, p6, p7]

    def forward_bits(self, al: torch.Tensor, bl: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool):
        partials = []
        for position, shift in enumerate((0, 2, 4)):
            digit = (bl >> shift) & 3
            if (self.approx_mask >> position) & 1:
                partials.append(self.cp62(al, digit, c_init=c_init, c_out=c_out, hard_middle=hard_middle))
            else:
                partials.append(self.exact62(al, digit))
        return self.exact_compress(*partials, c_out=c_out, hard_middle=hard_middle)

    def hard_inits(self):
        return {name: self.tables[name].hard_hex() for name in CP_BASE}

    def bin_reg(self):
        return torch.stack([m.bin_reg() for m in self.tables.values()]).mean()


class CPHybridDesign(BaseDesign):
    def __init__(self, name: str, rtl_dir: str, approx_mask: int, resources: str, aliases=()):
        self.approx_mask = int(approx_mask)
        bindings = (
            RtlBinding('cp_lut01', 's8862_approx62_cp.v', 's8862_approx62_cp', 'low01_lut'),
            RtlBinding('cp_lut23', 's8862_approx62_cp.v', 's8862_approx62_cp', 'pair23_lut'),
            RtlBinding('cp_lut45', 's8862_approx62_cp.v', 's8862_approx62_cp', 'pair45_lut'),
            RtlBinding('cp_lut67', 's8862_approx62_cp.v', 's8862_approx62_cp', 'pair67_lut'),
        )
        self.spec = DesignSpec(
            name=name, rtl_dir=rtl_dir, resource_summary=resources,
            base_inits=dict(CP_BASE), mutable_bits=dict(CP_MUTABLE), search_bits=dict(CP_MUTABLE),
            rtl_bindings=bindings, aliases=tuple(aliases),
        )

    def build_core(self, inits, init_conf: float, noise_std: float):
        return CPHybridCore(inits, self.spec.mutable_bits, self.approx_mask, init_conf, noise_std)

    @staticmethod
    def _cp62_np(ints, a, digit):
        ab = [((a >> i) & 1).astype(np.uint64) for i in range(6)]
        d0 = (digit & 1).astype(np.uint64)
        d1 = ((digit >> 1) & 1).astype(np.uint64)
        addr01 = d0 + (d1 << 1) + (ab[0] << 2) + (ab[1] << 3) + np.uint64(16 + 32)
        addr23 = d0 + (d1 << 1) + (ab[1] << 2) + (ab[2] << 3) + (ab[3] << 4) + np.uint64(32)
        addr45 = d0 + (d1 << 1) + (ab[3] << 2) + (ab[4] << 3) + (ab[5] << 4) + np.uint64(32)
        p0, p1 = lut62_np(ints['cp_lut01'], addr01)
        p2, p3 = lut62_np(ints['cp_lut23'], addr23)
        p4, p5 = lut62_np(ints['cp_lut45'], addr45)
        p6, p7 = lut62_np(ints['cp_lut67'], addr45)
        out = np.zeros_like(a, dtype=np.int32)
        for i, p in enumerate((p0,p1,p2,p3,p4,p5,p6,p7)):
            out += p.astype(np.int32) << i
        return out

    def hard_low_numpy(self, inits):
        norm = self.normalize_inits(inits)
        ints = {k: hex_to_int(v) for k, v in norm.items()}
        a = np.repeat(np.arange(64, dtype=np.uint16), 64)
        b = np.tile(np.arange(64, dtype=np.uint16), 64)
        parts = []
        for pos, shift in enumerate((0,2,4)):
            digit = ((b >> shift) & 3).astype(np.uint16)
            if (self.approx_mask >> pos) & 1:
                parts.append(self._cp62_np(ints, a, digit))
            else:
                parts.append(a.astype(np.int32) * digit.astype(np.int32))
        return (parts[0] + (parts[1] << 2) + (parts[2] << 4)) & 0xFFF
