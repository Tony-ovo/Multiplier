from __future__ import annotations

import torch

from ...lut import TrainableLUT6, ste_binarize, sharp01, int_bits_t


class BooleanCoreMixin:
    @staticmethod
    def xor2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y - 2.0 * x * y

    @staticmethod
    def fixed_node(value: torch.Tensor, *, c_out: float, hard_middle: bool) -> torch.Tensor:
        if hard_middle:
            return ste_binarize(value)
        return sharp01(torch.clamp(value, 0.0, 1.0), c_out)

    def add_bits(self, x: list[torch.Tensor], y: list[torch.Tensor], *, c_out: float, hard_middle: bool) -> list[torch.Tensor]:
        carry = torch.zeros_like(x[0])
        result: list[torch.Tensor] = []
        for xi, yi in zip(x, y):
            s = self.xor2(self.xor2(xi, yi), carry)
            c = xi * yi + xi * carry + yi * carry - 2.0 * xi * yi * carry
            s = self.fixed_node(s, c_out=c_out, hard_middle=hard_middle)
            c = self.fixed_node(c, c_out=c_out, hard_middle=hard_middle)
            result.append(s)
            carry = c
        return result

    def exact_compress(self, plow: list[torch.Tensor], pmid: list[torch.Tensor], phigh: list[torch.Tensor], *, c_out: float, hard_middle: bool) -> list[torch.Tensor]:
        z = torch.zeros_like(plow[0])
        low = plow + [z] * 4
        mid = [z, z] + pmid + [z, z]
        high = [z] * 4 + phigh
        return self.add_bits(self.add_bits(low, mid, c_out=c_out, hard_middle=hard_middle), high, c_out=c_out, hard_middle=hard_middle)

    @staticmethod
    def exact62(a: torch.Tensor, digit: torch.Tensor) -> list[torch.Tensor]:
        return int_bits_t(a * digit, 8)

    def fixed_lut62(self, table: torch.Tensor, inputs: list[torch.Tensor], *, c_out: float, hard_middle: bool):
        o5 = TrainableLUT6.soft_lut(inputs[:5], table[:32])
        o6 = TrainableLUT6.soft_lut(inputs[:6], table)
        o5 = self.fixed_node(o5, c_out=c_out, hard_middle=hard_middle)
        o6 = self.fixed_node(o6, c_out=c_out, hard_middle=hard_middle)
        return o5, o6
