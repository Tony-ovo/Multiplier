from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

FORMAT_VERSION = 1
SIGNED_TOTAL = 256 * 256
SIGNED_NONZERO = 255 * 255
MAX_ABS_PRODUCT = 128 * 128
LL_STATES = 64 * 64


def int_to_hex(value: int) -> str:
    return f"64'h{int(value) & ((1 << 64) - 1):016X}"


def hex_to_int(value: str | int) -> int:
    if isinstance(value, int):
        if not 0 <= value < (1 << 64):
            raise ValueError(value)
        return value
    text = str(value).strip().replace('_', '')
    if text.lower().startswith("64'h"):
        text = text[4:]
    elif text.lower().startswith('0x'):
        text = text[2:]
    if not text or len(text) > 16 or any(c not in '0123456789abcdefABCDEF' for c in text):
        raise ValueError(f'invalid LUT INIT: {value!r}')
    return int(text, 16)


def int_bits(value: int, width: int = 64) -> list[int]:
    return [(int(value) >> i) & 1 for i in range(width)]


def bits_int(bits: Iterable[int]) -> int:
    out = 0
    for i, bit in enumerate(bits):
        if int(bit):
            out |= 1 << i
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass(frozen=True)
class ObjectiveWeights:
    workload_mred: float = 1.0
    workload_er: float = 0.25
    workload_ned: float = 0.10
    workload_bias: float = 0.05
    uniform_mred: float = 0.05


@dataclass
class Metrics:
    total_cases: int
    error_cases: int
    ER: float
    MED: float
    NED: float
    MRED: float
    MRED_total: float
    WCE: int
    RMSE: float
    bias: float
    zero_violations: int
    symmetry_violations: int
    objective_score: float
    workload_ER: float
    workload_MED: float
    workload_NED: float
    workload_MRED: float
    workload_MRED_total: float
    workload_WCE: int
    workload_RMSE: float
    workload_bias: float
    workload_zero_violation_probability: float
    workload_zero_violation_rate: float

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        return (
            f"score={self.objective_score:.10f} "
            f"wMRED={self.workload_MRED:.10f} wER={self.workload_ER:.6f} "
            f"wMAE={self.workload_MED:.4f} wBias={self.workload_bias:.4f} "
            f"uMRED={self.MRED:.10f} uER={self.ER:.6f} uMAE={self.MED:.4f} "
            f"WCE={self.WCE} bias={self.bias:.4f}"
        )


def hamming(a: Mapping[str, str], b: Mapping[str, str], bits: Mapping[str, Sequence[int]] | None = None) -> int:
    names = sorted(a)
    if set(names) != set(b):
        raise ValueError('INIT key mismatch')
    total = 0
    for name in names:
        diff = hex_to_int(a[name]) ^ hex_to_int(b[name])
        if bits is None:
            total += diff.bit_count()
        else:
            total += sum((diff >> int(i)) & 1 for i in bits[name])
    return total
