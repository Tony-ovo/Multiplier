"""Quad-INIT variants of the unified-trainer hardware cores.

Only the table parameterization changes (QuadLUT6_2 instead of the binary
TrainableLUT6_2); circuit arithmetic, frozen bits, RTL bindings, hard numpy
models, artifact formats and the whole verify/refine/export toolchain stay
identical, so quad runs are directly comparable with the wce_batch1 controls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

_UNIFIED = Path(__file__).resolve().parents[1] / 'signed88_unified_trainer'
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))

from signed88.hardware.base import Low6Core, SignedLow6Model  # noqa: E402
from signed88.hardware.designs.cphybrid import CPHybridCore, CP_BASE  # noqa: E402
from signed88.hardware.designs.cpsplit import CPSplitCore, SPLIT_BASE  # noqa: E402
from signed88.hardware.designs.defaultsplit import DefaultSplitCore, DSPLIT_BASE  # noqa: E402

from quad_lut import QuadLUT6_2  # noqa: E402


class QuadCPSplitCore(CPSplitCore):
    """balanced_split core with 4-level INIT entries (112 mutable entries)."""

    def __init__(self, inits: Mapping[str, str], mutable_bits, init_conf: float,
                 noise_std: float, tau0: float = 3.0):
        Low6Core.__init__(self)
        self.tables = nn.ModuleDict({
            name: QuadLUT6_2(inits[name], mutable_bits[name], init_conf, noise_std, tau0)
            for name in SPLIT_BASE
        })


class QuadDefaultSplitCore(DefaultSplitCore):
    """default_split core with 4-level INIT entries (168 mutable entries)."""

    def __init__(self, inits: Mapping[str, str], mutable_bits, init_conf: float,
                 noise_std: float, tau0: float = 3.0):
        Low6Core.__init__(self)
        self.tables = nn.ModuleDict({
            name: QuadLUT6_2(inits[name], mutable_bits[name], init_conf, noise_std, tau0)
            for name in DSPLIT_BASE
        })


class QuadCPHybridCore(CPHybridCore):
    """CP-shared-table core (default/fast/balanced/quality) with quad entries."""

    def __init__(self, inits: Mapping[str, str], mutable_bits, approx_mask: int,
                 init_conf: float, noise_std: float, tau0: float = 3.0):
        Low6Core.__init__(self)
        self.approx_mask = int(approx_mask)
        self.tables = nn.ModuleDict({
            name: QuadLUT6_2(inits[name], mutable_bits[name], init_conf, noise_std, tau0)
            for name in CP_BASE
        })


def build_quad_model(design, inits, init_conf: float, noise_std: float, tau0: float) -> SignedLow6Model:
    norm = design.normalize_inits(inits)
    name = design.spec.name
    if name == 'balanced_split':
        core = QuadCPSplitCore(norm, design.spec.mutable_bits, init_conf, noise_std, tau0)
    elif name == 'default_split':
        core = QuadDefaultSplitCore(norm, design.spec.mutable_bits, init_conf, noise_std, tau0)
    elif name in ('default', 'fast', 'balanced', 'quality'):
        core = QuadCPHybridCore(norm, design.spec.mutable_bits, design.approx_mask,
                                init_conf, noise_std, tau0)
    else:
        raise ValueError(f'quad parameterization not wired for design {name!r}')
    return SignedLow6Model(core)


def quad_tables(model: SignedLow6Model):
    return {name: module for name, module in model.core.tables.items()}


def collapse_regularizer(model: SignedLow6Model, c_init: float) -> torch.Tensor:
    regs = [t.collapse_reg(c_init) for t in model.core.tables.values()]
    return torch.stack(regs).mean()


def quad_usage_stats(model: SignedLow6Model) -> dict:
    """Fraction of mutable entries currently resting on each of the 4 levels."""
    counts = torch.zeros(4, dtype=torch.long)
    total = 0
    for table in model.core.tables.values():
        idx = table.quad_level_index()[table.mutable_mask]
        total += int(idx.numel())
        counts += torch.bincount(idx.cpu(), minlength=4)
    frac = (counts.to(torch.float64) / max(total, 1)).tolist()
    return {
        'mutable_entries': total,
        'level_counts': {name: int(c) for name, c in zip(('00', '01', '10', '11'), counts.tolist())},
        'level_fractions': {name: float(f) for name, f in zip(('00', '01', '10', '11'), frac)},
        'intermediate_fraction': float(frac[1] + frac[2]),
    }


@torch.no_grad()
def reset_quad_from_inits(model: SignedLow6Model, design, inits, init_conf: float,
                          noise_std: float) -> None:
    """Re-seed all quad tables from a binary INIT solution (population restart)."""
    norm = design.normalize_inits(inits)
    for name, table in model.core.tables.items():
        table.reset_from_hex(norm[name], init_conf, noise_std)


@torch.no_grad()
def randomize_quad_logits(model: SignedLow6Model, mean: float, std: float,
                          generator: torch.Generator | None = None) -> dict:
    table_count = 0
    logit_count = 0
    for table in model.core.tables.values():
        table_count += 1
        logit_count += table.randomize_mutable_logits(mean=mean, std=std, generator=generator)
    return {
        'distribution': 'normal', 'requested_mean': float(mean), 'requested_std': float(std),
        'quad_tables': table_count, 'randomized_logits': logit_count,
    }
