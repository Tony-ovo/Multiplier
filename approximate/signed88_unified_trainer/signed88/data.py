from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .common import LL_STATES, SIGNED_TOTAL, sha256_file


@dataclass(frozen=True)
class CalibrationProfile:
    source: str
    sha256: str
    weight_column: str
    row_count: int
    raw_weight_sum: float
    a: np.ndarray
    b: np.ndarray
    exact: np.ndarray
    probability: np.ndarray
    state_index: np.ndarray
    state_probability: np.ndarray
    red_weight_by_state: np.ndarray
    zero_probability_by_state: np.ndarray
    nonzero_probability: float
    zero_probability: float

    def metadata(self) -> dict:
        return {
            'source': self.source,
            'sha256': self.sha256,
            'weight_column': self.weight_column,
            'row_count': self.row_count,
            'raw_weight_sum': self.raw_weight_sum,
            'nonzero_probability': self.nonzero_probability,
            'zero_probability': self.zero_probability,
            'low_state_coverage': int(np.count_nonzero(self.state_probability)),
            'training_semantics': (
                'loss is evaluated on signed-int8 (a,b)->signed-int16 product rows; '
                'the 4096 low-state core is cached once per epoch only as an exact acceleration'
            ),
        }


@dataclass
class TorchCalibration:
    a: torch.Tensor
    b: torch.Tensor
    exact: torch.Tensor
    probability: torch.Tensor
    state_index: torch.Tensor
    state_probability: torch.Tensor
    red_weight_by_state: torch.Tensor
    zero_probability_by_state: torch.Tensor
    nonzero_probability: float
    zero_probability: float


def _select_weight(fields: set[str], requested: str) -> str:
    if requested != 'auto':
        if requested not in fields:
            raise ValueError(f'CSV has no column {requested!r}')
        return requested
    for name in ('count', 'p_calib', 'weight', 'probability'):
        if name in fields:
            return name
    raise ValueError('CSV needs count, p_calib, weight, or probability')


def load_calibration_csv(path: Path, weight_column: str = 'auto') -> CalibrationProfile:
    path = Path(path).resolve()
    rows: list[tuple[int, int, float]] = []
    seen = set()
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV has no header')
        mapping = {x.strip().lower(): x for x in reader.fieldnames}
        if 'a' not in mapping or 'b' not in mapping:
            raise ValueError('CSV must contain signed-int8 columns a,b')
        selected = _select_weight(set(mapping), weight_column.lower())
        for line_no, row in enumerate(reader, 2):
            a = int(row[mapping['a']])
            b = int(row[mapping['b']])
            if not (-128 <= a <= 127 and -128 <= b <= 127):
                raise ValueError(f'line {line_no}: operands are not signed int8')
            if (a, b) in seen:
                raise ValueError(f'line {line_no}: duplicate pair {(a,b)}')
            seen.add((a, b))
            w = float(row[mapping[selected]])
            if not math.isfinite(w) or w <= 0:
                raise ValueError(f'line {line_no}: weight must be finite and positive')
            rows.append((a, b, w))
    if not rows:
        raise ValueError('empty calibration CSV')
    raw_total = math.fsum(w for _, _, w in rows)
    a = np.asarray([x[0] for x in rows], dtype=np.int16)
    b = np.asarray([x[1] for x in rows], dtype=np.int16)
    probability = np.asarray([x[2] / raw_total for x in rows], dtype=np.float64)
    exact = (a.astype(np.int32) * b.astype(np.int32)).astype(np.int32)
    state_index = ((a.astype(np.int32) & 63) * 64 + (b.astype(np.int32) & 63)).astype(np.int64)
    state_probability = np.zeros(LL_STATES, dtype=np.float64)
    red_weight = np.zeros(LL_STATES, dtype=np.float64)
    zero_probability = np.zeros(LL_STATES, dtype=np.float64)
    np.add.at(state_probability, state_index, probability)
    nonzero = exact != 0
    np.add.at(red_weight, state_index[nonzero], probability[nonzero] / np.abs(exact[nonzero]))
    np.add.at(zero_probability, state_index[~nonzero], probability[~nonzero])
    nonzero_probability = float(probability[nonzero].sum())
    zero_probability_total = float(probability[~nonzero].sum())
    return CalibrationProfile(
        source=str(path), sha256=sha256_file(path), weight_column=selected,
        row_count=len(rows), raw_weight_sum=raw_total, a=a, b=b, exact=exact,
        probability=probability, state_index=state_index,
        state_probability=state_probability, red_weight_by_state=red_weight,
        zero_probability_by_state=zero_probability,
        nonzero_probability=nonzero_probability, zero_probability=zero_probability_total,
    )


def to_torch(profile: CalibrationProfile, device: torch.device) -> TorchCalibration:
    return TorchCalibration(
        a=torch.as_tensor(profile.a.astype(np.int64), device=device),
        b=torch.as_tensor(profile.b.astype(np.int64), device=device),
        exact=torch.as_tensor(profile.exact.astype(np.float32), device=device),
        probability=torch.as_tensor(profile.probability.astype(np.float32), device=device),
        state_index=torch.as_tensor(profile.state_index, dtype=torch.long, device=device),
        state_probability=torch.as_tensor(profile.state_probability.astype(np.float32), device=device),
        red_weight_by_state=torch.as_tensor(profile.red_weight_by_state.astype(np.float32), device=device),
        zero_probability_by_state=torch.as_tensor(profile.zero_probability_by_state.astype(np.float32), device=device),
        nonzero_probability=profile.nonzero_probability,
        zero_probability=profile.zero_probability,
    )


def uniform_signed_state_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_probability = np.full(LL_STATES, 1.0 / LL_STATES, dtype=np.float64)
    red_weight = np.zeros(LL_STATES, dtype=np.float64)
    zero_probability = np.zeros(LL_STATES, dtype=np.float64)
    for a_raw in range(256):
        a = a_raw if a_raw < 128 else a_raw - 256
        al = a_raw & 63
        for b_raw in range(256):
            b = b_raw if b_raw < 128 else b_raw - 256
            bl = b_raw & 63
            idx = al * 64 + bl
            exact = a * b
            if exact == 0:
                zero_probability[idx] += 1.0 / SIGNED_TOTAL
            else:
                red_weight[idx] += (1.0 / SIGNED_TOTAL) / abs(exact)
    return state_probability, red_weight, zero_probability


UNIFORM_STATE_PROB, UNIFORM_RED_WEIGHT, UNIFORM_ZERO_PROB = uniform_signed_state_weights()
UNIFORM_NONZERO_PROB = 1.0 - float(UNIFORM_ZERO_PROB.sum())
