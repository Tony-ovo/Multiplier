from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .common import hex_to_int, int_bits, int_to_hex


def sharp01(x: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)
    if abs(float(c) - 1.0) < 1e-12:
        return x
    xc = torch.pow(x, float(c))
    yc = torch.pow(1.0 - x, float(c))
    return xc / (xc + yc + eps)


def ste_binarize(x: torch.Tensor) -> torch.Tensor:
    hard = (x >= 0.5).to(x.dtype)
    return hard.detach() - x.detach() + x


def int_bits_t(x: torch.Tensor, width: int) -> list[torch.Tensor]:
    return [((x >> i) & 1).to(torch.float32) for i in range(width)]


class TrainableLUT6(nn.Module):
    def __init__(self, init_hex: str, mutable_bits: Sequence[int], init_conf: float, noise_std: float):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        mask = torch.zeros(64, dtype=torch.bool)
        if mutable_bits:
            mask[list(mutable_bits)] = True
        self.register_buffer('mutable_mask', mask)
        self.register_buffer('fixed_bits', torch.tensor(int_bits(hex_to_int(init_hex)), dtype=torch.float32))
        self.reset_from_hex(init_hex, init_conf, noise_std)

    @torch.no_grad()
    def reset_from_hex(self, init_hex: str, init_conf: float, noise_std: float) -> None:
        bits = torch.tensor(int_bits(hex_to_int(init_hex)), dtype=torch.float32, device=self.logits.device)
        conf = min(max(float(init_conf), 1e-4), 1 - 1e-4)
        p = torch.where(bits > 0.5, torch.full_like(bits, conf), torch.full_like(bits, 1-conf))
        logits = torch.log(p / (1-p))
        if noise_std > 0:
            logits += torch.randn_like(logits) * float(noise_std) * self.mutable_mask.to(logits.dtype)
        self.logits.copy_(logits)

    def table(self, c_init: float) -> torch.Tensor:
        trained = sharp01(torch.sigmoid(self.logits), c_init)
        return torch.where(self.mutable_mask, trained, self.fixed_bits)

    def hard_bits(self) -> torch.Tensor:
        trained = (torch.sigmoid(self.logits).detach() >= 0.5).to(torch.float32)
        return torch.where(self.mutable_mask, trained, self.fixed_bits)

    def hard_hex(self) -> str:
        bits = self.hard_bits().detach().cpu().to(torch.int32).tolist()
        value = sum((int(bit) & 1) << i for i, bit in enumerate(bits))
        return int_to_hex(value)

    def bin_reg(self) -> torch.Tensor:
        if not bool(torch.any(self.mutable_mask)):
            return torch.zeros((), device=self.logits.device)
        p = torch.sigmoid(self.logits[self.mutable_mask])
        return torch.mean(p * (1-p))

    @staticmethod
    def soft_lut(inputs: list[torch.Tensor], table: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(inputs[0])
        for addr in range(1 << len(inputs)):
            w = torch.ones_like(inputs[0])
            for i, x in enumerate(inputs):
                w = w * (x if ((addr >> i) & 1) else (1-x))
            out = out + w * table[addr]
        return out

    def forward(self, *inputs: torch.Tensor, c_init: float, c_out: float, hard_middle: bool) -> torch.Tensor:
        xs = [ste_binarize(x) if hard_middle else x for x in inputs]
        y = self.soft_lut(xs, self.table(c_init))
        y = sharp01(y, c_out)
        return ste_binarize(y) if hard_middle else y


class TrainableLUT6_2(TrainableLUT6):
    def forward(self, *inputs: torch.Tensor, c_init: float, c_out: float, hard_middle: bool):
        xs = [ste_binarize(x) if hard_middle else x for x in inputs]
        table = self.table(c_init)
        o5 = self.soft_lut(xs[:5], table[:32])
        o6 = self.soft_lut(xs[:6], table)
        o5 = sharp01(o5, c_out)
        o6 = sharp01(o6, c_out)
        if hard_middle:
            o5 = ste_binarize(o5)
            o6 = ste_binarize(o6)
        return o5, o6


def lut_bit_np(init: int, addr):
    import numpy as np
    return ((np.uint64(init) >> addr.astype(np.uint64)) & np.uint64(1)).astype(np.uint16)


def lut62_np(init: int, addr6):
    import numpy as np
    return lut_bit_np(init, addr6 & np.uint64(31)), lut_bit_np(init, addr6)
