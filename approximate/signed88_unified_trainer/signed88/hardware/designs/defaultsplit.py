"""Default/Fast topology with the three 6x2 truth tables decoupled.

Stock ``default`` instantiates three physical ``s8862_approx62_cp`` children
(for ``b[1:0]``, ``b[3:2]``, ``b[5:4]``) but they share one INIT set, so the
x1 / x4 / x16 digits are forced to make the same approximation.  Splitting
them costs zero extra LUT6_2 (the three instances always existed) and triples
the trainable freedom from 56 to 168 bits.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from ...common import hex_to_int
from ...lut import TrainableLUT6_2, int_bits_t, lut62_np
from ..base import BaseDesign, DesignSpec, Low6Core, RtlBinding
from .common import BooleanCoreMixin
from .cphybrid import CP_BASE, CP_MUTABLE

DSPLIT_SEGMENTS = ('lo', 'mid', 'hi')
_CP_SUFFIXES = ('lut01', 'lut23', 'lut45', 'lut67')

DSPLIT_BASE = {
    f'{segment}_{suffix}': CP_BASE[f'cp_{suffix}']
    for segment in DSPLIT_SEGMENTS for suffix in _CP_SUFFIXES
}
DSPLIT_MUTABLE = {
    f'{segment}_{suffix}': CP_MUTABLE[f'cp_{suffix}']
    for segment in DSPLIT_SEGMENTS for suffix in _CP_SUFFIXES
}


class DefaultSplitCore(Low6Core, BooleanCoreMixin):
    def __init__(self, inits: Mapping[str, str], mutable_bits, init_conf: float, noise_std: float):
        super().__init__()
        self.tables = nn.ModuleDict({
            name: TrainableLUT6_2(inits[name], mutable_bits[name], init_conf, noise_std)
            for name in DSPLIT_BASE
        })

    def cp62(self, segment: str, a: torch.Tensor, digit: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool):
        ab = int_bits_t(a, 6)
        db = int_bits_t(digit, 2)
        one = torch.ones_like(ab[0])
        t = lambda suffix: self.tables[f'{segment}_{suffix}']
        p0, p1 = t('lut01')(db[0], db[1], ab[0], ab[1], one, one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p2, p3 = t('lut23')(db[0], db[1], ab[1], ab[2], ab[3], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p4, p5 = t('lut45')(db[0], db[1], ab[3], ab[4], ab[5], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        p6, p7 = t('lut67')(db[0], db[1], ab[3], ab[4], ab[5], one, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        return [p0, p1, p2, p3, p4, p5, p6, p7]

    def forward_bits(self, al: torch.Tensor, bl: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool):
        partials = []
        for segment, shift in (('lo', 0), ('mid', 2), ('hi', 4)):
            digit = (bl >> shift) & 3
            partials.append(self.cp62(segment, al, digit, c_init=c_init, c_out=c_out, hard_middle=hard_middle))
        return self.exact_compress(*partials, c_out=c_out, hard_middle=hard_middle)

    def hard_inits(self):
        return {name: self.tables[name].hard_hex() for name in DSPLIT_BASE}

    def bin_reg(self):
        return torch.stack([m.bin_reg() for m in self.tables.values()]).mean()


class DefaultSplitDesign(BaseDesign):
    def __init__(self):
        bindings = tuple(
            RtlBinding(
                f'{segment}_{suffix}',
                's8862_approx62_cp_dsplit.v',
                f's8862_approx62_cp_{segment}',
                f'{"low01" if suffix == "lut01" else "pair" + suffix[3:]}_lut',
            )
            for segment in DSPLIT_SEGMENTS for suffix in _CP_SUFFIXES
        )
        self.spec = DesignSpec(
            name='default_split',
            rtl_dir='DefaultSplit',
            resource_summary='37 LUT6_2 + 6 CARRY4 (lo/mid/hi INITs decoupled)',
            base_inits=dict(DSPLIT_BASE),
            mutable_bits=dict(DSPLIT_MUTABLE),
            search_bits=dict(DSPLIT_MUTABLE),
            rtl_bindings=bindings,
        )

    def build_core(self, inits, init_conf: float, noise_std: float):
        return DefaultSplitCore(inits, self.spec.mutable_bits, init_conf, noise_std)

    @staticmethod
    def _cp62_np(ints, segment, a, digit):
        ab = [((a >> i) & 1).astype(np.uint64) for i in range(6)]
        d0 = (digit & 1).astype(np.uint64)
        d1 = ((digit >> 1) & 1).astype(np.uint64)
        addr01 = d0 + (d1 << 1) + (ab[0] << 2) + (ab[1] << 3) + np.uint64(16 + 32)
        addr23 = d0 + (d1 << 1) + (ab[1] << 2) + (ab[2] << 3) + (ab[3] << 4) + np.uint64(32)
        addr45 = d0 + (d1 << 1) + (ab[3] << 2) + (ab[4] << 3) + (ab[5] << 4) + np.uint64(32)
        p0, p1 = lut62_np(ints[f'{segment}_lut01'], addr01)
        p2, p3 = lut62_np(ints[f'{segment}_lut23'], addr23)
        p4, p5 = lut62_np(ints[f'{segment}_lut45'], addr45)
        p6, p7 = lut62_np(ints[f'{segment}_lut67'], addr45)
        out = np.zeros_like(a, dtype=np.int32)
        for i, p in enumerate((p0, p1, p2, p3, p4, p5, p6, p7)):
            out += p.astype(np.int32) << i
        return out

    def hard_low_numpy(self, inits):
        norm = self.normalize_inits(inits)
        ints = {k: hex_to_int(v) for k, v in norm.items()}
        a = np.repeat(np.arange(64, dtype=np.uint16), 64)
        b = np.tile(np.arange(64, dtype=np.uint16), 64)
        parts = [
            self._cp62_np(ints, 'lo', a, (b & 3).astype(np.uint16)),
            self._cp62_np(ints, 'mid', a, ((b >> 2) & 3).astype(np.uint16)),
            self._cp62_np(ints, 'hi', a, ((b >> 4) & 3).astype(np.uint16)),
        ]
        return (parts[0] + (parts[1] << 2) + (parts[2] << 4)) & 0xFFF
