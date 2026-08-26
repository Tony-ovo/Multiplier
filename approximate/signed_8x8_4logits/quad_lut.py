"""Four-level (2-bit) INIT parameterization for LUT6_2 truth tables.

Each mutable INIT entry is modeled as a categorical variable over the four
ordered levels 00/01/10/11 with numeric embedding {0, 1/3, 2/3, 1}:

    v = sum_k softmax(tau * logits)_k * level_k

The gradient of v w.r.t. each level logit is

    dv/dlogit_k = tau * p_k * (level_k - v)

which is exactly the distance-weighted credit assignment requested by the
experiment design: when the entry currently sits near 00, the 01 logit
receives a small pull and the 11 logit a large one (proportional to how far
each level is from the current value), and symmetrically for 10 near 11.

Deployment collapse rule: levels {00,01} -> 0, {10,11} -> 1 (threshold 0.5).

In the hard-STE phase the entry value is the straight-through argmax level.
Because downstream signal binarization thresholds at 0.5 and 1/3 < 0.5 < 2/3,
the hard forward pass is numerically identical to the deployed binary
circuit, while the backward pass still spreads gradient over all four levels.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_UNIFIED = Path(__file__).resolve().parents[1] / 'signed88_unified_trainer'
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))

from signed88.common import hex_to_int, int_bits, int_to_hex  # noqa: E402
from signed88.lut import TrainableLUT6, sharp01, ste_binarize  # noqa: E402

QUAD_LEVELS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
LEVEL_NAMES = ('00', '01', '10', '11')


class QuadLUT6_2(nn.Module):
    """LUT6_2 whose 64 INIT entries are trained as 4-level categoricals.

    The forward signature matches signed88.lut.TrainableLUT6_2 so the module
    is a drop-in replacement inside existing hardware cores.  ``c_init``
    scales the softmax temperature (tau = tau0 * c_init).
    """

    def __init__(self, init_hex: str, mutable_bits: Sequence[int], init_conf: float,
                 noise_std: float, tau0: float = 3.0):
        super().__init__()
        self.tau0 = float(tau0)
        self.logits = nn.Parameter(torch.zeros(64, 4, dtype=torch.float32))
        mask = torch.zeros(64, dtype=torch.bool)
        if mutable_bits:
            mask[list(mutable_bits)] = True
        self.register_buffer('mutable_mask', mask)
        self.register_buffer('fixed_bits', torch.tensor(int_bits(hex_to_int(init_hex)), dtype=torch.float32))
        self.register_buffer('levels', torch.tensor(QUAD_LEVELS, dtype=torch.float32))
        self.reset_from_hex(init_hex, init_conf, noise_std)

    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_conf: float, noise_std: float) -> None:
        """Initialize each entry as a one-hot toward the extreme level of its bit.

        bit=0 -> level 00, bit=1 -> level 11 with softmax confidence
        ``init_conf`` at tau=tau0 (i.e. baseline behaviour is reproduced).
        """
        bits = torch.tensor(int_bits(hex_to_int(init_hex)), dtype=torch.float32, device=self.logits.device)
        conf = min(max(float(init_conf), 1e-4), 1.0 - 1e-4)
        # softmax(s * onehot) mass on the target: e^s / (e^s + 3); solve for s.
        s = math.log(conf * 3.0 / (1.0 - conf)) / self.tau0
        logits = torch.zeros(64, 4, device=self.logits.device)
        logits[bits > 0.5, 3] = s
        logits[bits <= 0.5, 0] = s
        if noise_std > 0:
            noise = torch.randn_like(logits) * float(noise_std)
            logits += noise * self.mutable_mask.to(logits.dtype).unsqueeze(1)
        self.logits.copy_(logits)

    @torch.no_grad()
    def randomize_mutable_logits(self, mean: float = 0.0, std: float = 1.0,
                                 generator: torch.Generator | None = None) -> int:
        if not math.isfinite(float(mean)):
            raise ValueError('random-logit mean must be finite')
        if not math.isfinite(float(std)) or float(std) <= 0.0:
            raise ValueError('random-logit std must be finite and > 0')
        count = int(self.mutable_mask.sum().item())
        if count:
            samples = torch.empty(count, 4, dtype=self.logits.dtype, device=self.logits.device)
            samples.normal_(mean=float(mean), std=float(std), generator=generator)
            self.logits[self.mutable_mask] = samples
        return count * 4

    def quad_probs(self, c_init: float) -> torch.Tensor:
        """(64,4) softmax level probabilities at the current temperature."""
        tau = self.tau0 * float(c_init)
        return F.softmax(self.logits * tau, dim=-1)

    def table(self, c_init: float) -> torch.Tensor:
        """Soft expected entry values in [0,1]; frozen addresses keep RTL bits."""
        v = torch.mv(self.quad_probs(c_init), self.levels)
        return torch.where(self.mutable_mask, v, self.fixed_bits)

    def quad_ste_table(self, c_init: float) -> torch.Tensor:
        """Hard argmax level per entry with straight-through soft gradients."""
        soft = torch.mv(self.quad_probs(c_init), self.levels)
        hard = self.levels[self.logits.argmax(dim=-1)]
        v = hard.detach() - soft.detach() + soft
        return torch.where(self.mutable_mask, v, self.fixed_bits)

    def quad_level_index(self) -> torch.Tensor:
        """Current discrete level per entry (frozen entries map to 00/11)."""
        idx = self.logits.argmax(dim=-1)
        frozen = torch.where(self.fixed_bits > 0.5, torch.full_like(idx, 3), torch.zeros_like(idx))
        return torch.where(self.mutable_mask, idx, frozen)

    def hard_bits(self) -> torch.Tensor:
        """Deployment collapse {00,01}->0, {10,11}->1."""
        bit = (self.logits.argmax(dim=-1) >= 2).to(torch.float32)
        return torch.where(self.mutable_mask, bit, self.fixed_bits)

    def hard_hex(self) -> str:
        bits = self.hard_bits().detach().cpu().to(torch.int32).tolist()
        return int_to_hex(sum((int(b) & 1) << i for i, b in enumerate(bits)))

    def bin_reg(self) -> torch.Tensor:
        """Normalized categorical entropy of mutable entries (0=committed)."""
        if not bool(torch.any(self.mutable_mask)):
            return torch.zeros((), device=self.logits.device)
        p = self.quad_probs(1.0)[self.mutable_mask]
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
        return entropy.mean() / math.log(4.0)

    def collapse_reg(self, c_init: float) -> torch.Tensor:
        """Penalty for entries whose value sits away from binary {0,1}.

        Pushes 01 toward 00 and 10 toward 11 late in training so the final
        binary collapse is a small perturbation instead of a cliff.
        """
        if not bool(torch.any(self.mutable_mask)):
            return torch.zeros((), device=self.logits.device)
        v = torch.mv(self.quad_probs(c_init), self.levels)[self.mutable_mask]
        target = (v >= 0.5).to(v.dtype)
        return torch.mean(torch.square(v - target.detach()))

    def forward(self, *inputs: torch.Tensor, c_init: float, c_out: float, hard_middle: bool):
        xs = [ste_binarize(x) if hard_middle else x for x in inputs]
        table = self.quad_ste_table(c_init) if hard_middle else self.table(c_init)
        o5 = TrainableLUT6.soft_lut(xs[:5], table[:32])
        o6 = TrainableLUT6.soft_lut(xs[:6], table)
        o5 = sharp01(o5, c_out)
        o6 = sharp01(o6, c_out)
        if hard_middle:
            o5 = ste_binarize(o5)
            o6 = ste_binarize(o6)
        return o5, o6
