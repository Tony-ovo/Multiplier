from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..common import FORMAT_VERSION, hex_to_int, int_to_hex, sha256_file, write_json


@dataclass(frozen=True)
class RtlBinding:
    table: str
    file: str
    module: str
    instance: str


@dataclass(frozen=True)
class DesignSpec:
    name: str
    rtl_dir: str
    resource_summary: str
    base_inits: Mapping[str, str]
    mutable_bits: Mapping[str, tuple[int, ...]]
    search_bits: Mapping[str, tuple[int, ...]]
    rtl_bindings: tuple[RtlBinding, ...]
    aliases: tuple[str, ...] = ()

    @property
    def train_names(self) -> tuple[str, ...]:
        return tuple(self.base_inits.keys())

    def metadata(self) -> dict:
        payload = {
            'format_version': FORMAT_VERSION,
            'design': self.name,
            'rtl_dir': self.rtl_dir,
            'resources': self.resource_summary,
            'train_names': list(self.train_names),
            'mutable_bits': {k: list(v) for k, v in self.mutable_bits.items()},
            'search_bits': {k: list(v) for k, v in self.search_bits.items()},
            'rtl_bindings': [b.__dict__ for b in self.rtl_bindings],
        }
        import hashlib
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        payload['schema_sha256'] = hashlib.sha256(raw).hexdigest()
        return payload


class Low6Core(nn.Module):
    def forward_bits(self, al: torch.Tensor, bl: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool) -> list[torch.Tensor]:
        raise NotImplementedError

    def hard_inits(self) -> Dict[str, str]:
        raise NotImplementedError

    def bin_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, al: torch.Tensor, bl: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool):
        bits = self.forward_bits(al, bl, c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        value = torch.zeros_like(al, dtype=torch.float32)
        for i, bit in enumerate(bits):
            value = value + bit * float(1 << i)
        return value, bits


class SignedLow6Model(nn.Module):
    """Full signed8x8 training model.

    The user-facing inputs are signed int8 rows and the loss is on the final
    signed16 product.  All supplied RTLs keep signed upper terms exact, hence
      approx_signed = exact_signed + approx_LL - AL*BL.
    The 4096 LL states are evaluated once per pass and gathered by row; this is
    only a cache optimization, not a change of the signed8 training objective.
    """
    def __init__(self, core: Low6Core):
        super().__init__()
        self.core = core
        al = torch.arange(64, dtype=torch.int64).repeat_interleave(64)
        bl = torch.arange(64, dtype=torch.int64).repeat(64)
        self.register_buffer('grid_al', al)
        self.register_buffer('grid_bl', bl)
        self.register_buffer('grid_exact_ll', (al * bl).to(torch.float32))

    def forward_low_grid(self, *, c_init: float, c_out: float, hard_middle: bool):
        return self.core(self.grid_al, self.grid_bl, c_init=c_init, c_out=c_out, hard_middle=hard_middle)

    def forward_signed_rows(self, a_signed: torch.Tensor, b_signed: torch.Tensor, state_index: torch.Tensor, *, c_init: float, c_out: float, hard_middle: bool):
        low_value, low_bits = self.forward_low_grid(c_init=c_init, c_out=c_out, hard_middle=hard_middle)
        exact_signed = (a_signed * b_signed).to(torch.float32)
        exact_ll = self.grid_exact_ll[state_index]
        approx_signed = exact_signed + low_value[state_index] - exact_ll
        return approx_signed, low_value, low_bits

    def hard_inits(self) -> Dict[str, str]:
        return self.core.hard_inits()

    def bin_reg(self) -> torch.Tensor:
        return self.core.bin_reg()


class BaseDesign:
    spec: DesignSpec

    def normalize_inits(self, inits: Mapping[str, str | int]) -> Dict[str, str]:
        if 'inits' in inits:
            inits = inits['inits']  # type: ignore[index]
        if set(inits) != set(self.spec.train_names):
            raise ValueError(f'INIT keys mismatch for {self.spec.name}: got={sorted(inits)} expected={sorted(self.spec.train_names)}')
        out: Dict[str, str] = {}
        for name in self.spec.train_names:
            value = hex_to_int(inits[name])
            base = hex_to_int(self.spec.base_inits[name])
            mutable = set(self.spec.mutable_bits[name])
            for bit in range(64):
                if bit not in mutable and (((value ^ base) >> bit) & 1):
                    raise ValueError(f'{name}[{bit}] is frozen')
            out[name] = int_to_hex(value)
        return out

    def random_inits(self, p_one: float, rng) -> Dict[str, str]:
        out = {}
        for name in self.spec.train_names:
            bits = [(hex_to_int(self.spec.base_inits[name]) >> i) & 1 for i in range(64)]
            for i in self.spec.mutable_bits[name]:
                bits[i] = 1 if rng.random() < p_one else 0
            value = sum(bit << i for i, bit in enumerate(bits))
            out[name] = int_to_hex(value)
        return self.normalize_inits(out)

    def perturb_inits(self, base: Mapping[str, str], p_flip: float, rng, force_change: bool = False) -> Dict[str, str]:
        out = {k: hex_to_int(v) for k, v in self.normalize_inits(base).items()}
        changed = 0
        for name in self.spec.train_names:
            for bit in self.spec.search_bits[name]:
                if rng.random() < p_flip:
                    out[name] ^= 1 << bit
                    changed += 1
        if force_change and changed == 0:
            name = rng.choice(self.spec.train_names)
            bit = rng.choice(self.spec.search_bits[name])
            out[name] ^= 1 << bit
        return self.normalize_inits({k: int_to_hex(v) for k, v in out.items()})

    def build_model(self, inits: Mapping[str, str], init_conf: float, noise_std: float) -> SignedLow6Model:
        return SignedLow6Model(self.build_core(self.normalize_inits(inits), init_conf, noise_std))

    def build_core(self, inits: Mapping[str, str], init_conf: float, noise_std: float) -> Low6Core:
        raise NotImplementedError

    def hard_low_numpy(self, inits: Mapping[str, str]) -> np.ndarray:
        raise NotImplementedError

    def artifact(self, inits: Mapping[str, str], *, metrics: dict | None = None, extra: dict | None = None) -> dict:
        obj = {
            'format_version': FORMAT_VERSION,
            'design': self.spec.name,
            'design_spec': self.spec.metadata(),
            'inits': self.normalize_inits(inits),
        }
        if metrics is not None:
            obj['metrics'] = metrics
        if extra:
            obj.update(extra)
        return obj

    def export_rtl(self, template_root: Path, out_dir: Path, inits: Mapping[str, str], metadata: dict | None = None) -> Path:
        inits = self.normalize_inits(inits)
        src = Path(template_root) / self.spec.rtl_dir
        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(src, out_dir)
        by_file: dict[str, list[RtlBinding]] = {}
        for binding in self.spec.rtl_bindings:
            by_file.setdefault(binding.file, []).append(binding)
        for rel, bindings in by_file.items():
            path = out_dir / rel
            text = path.read_text(encoding='utf-8')
            for binding in bindings:
                text = self._patch_binding(text, binding, inits[binding.table])
            path.write_text(text, encoding='utf-8')
        meta = {'design': self.spec.name, 'spec': self.spec.metadata(), 'inits': inits}
        if metadata:
            meta.update(metadata)
        write_json(out_dir / 'trained_artifact.json', meta)
        return out_dir

    @staticmethod
    def _patch_binding(text: str, binding: RtlBinding, new_init: str) -> str:
        module_re = re.compile(rf'(module\s+{re.escape(binding.module)}\b.*?endmodule)', re.S)
        m = module_re.search(text)
        if not m:
            raise ValueError(f'module {binding.module} not found in {binding.file}')
        block = m.group(1)
        inst_re = re.compile(
            rf'((?:LUT6_2|LUT6)\s*#\s*\(\s*\.INIT\()\s*64\'h[0-9a-fA-F_]+\s*(\)\s*\)\s*{re.escape(binding.instance)}\s*\()',
            re.S,
        )
        hits = list(inst_re.finditer(block))
        if len(hits) != 1:
            raise ValueError(f'expected one binding for {binding.module}.{binding.instance}, found {len(hits)}')
        block2 = inst_re.sub(lambda mm: mm.group(1) + new_init + mm.group(2), block, count=1)
        return text[:m.start(1)] + block2 + text[m.end(1):]
