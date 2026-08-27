"""Sparse dot-product traces for application-aware signed8x8 training.

For the current RTL families the final product error depends only on
``state = (a & 63) * 64 + (b & 63)``.  A complete dot product can therefore
be represented by 4096 sparse state counts.  This module keeps that trace
representation independent from the legacy joint-histogram loader in
``data.py`` and provides both a hard NumPy evaluator and a differentiable
PyTorch objective.

The proxy score is not a literal perplexity prediction.  It is a
sensitivity-weighted output-perturbation objective intended to preserve the
error accumulation, scale and layer/channel identity that a flat (a,b)
histogram loses.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import LL_STATES, sha256_file
from .data import CalibrationProfile, load_calibration_csv


DOT_TRACE_FORMAT = 'signed88-dot-trace-v1'
STANDARD_SPLITS = frozenset(('train', 'validation', 'test'))


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DotTraceProfile:
    """Compact CSR-like storage for a collection of dot-product groups."""

    source: str
    sha256: str
    ids: tuple[str, ...]
    layers: tuple[str, ...]
    channels: tuple[str, ...]
    splits: tuple[str, ...]
    scale: np.ndarray
    sensitivity: np.ndarray
    normalizer: np.ndarray
    group_index: np.ndarray
    state_index: np.ndarray
    count: np.ndarray

    @property
    def group_count(self) -> int:
        return len(self.ids)

    @property
    def nnz(self) -> int:
        return int(self.state_index.size)

    @property
    def mac_count(self) -> int:
        return int(self.count.sum(dtype=np.int64))

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.splits)))

    def metadata(self) -> dict:
        return {
            'format': DOT_TRACE_FORMAT,
            'source': self.source,
            'sha256': self.sha256,
            'group_count': self.group_count,
            'nnz': self.nnz,
            'mac_count': self.mac_count,
            'splits': list(self.split_names),
            'layer_count': len(set(self.layers)),
            'channel_count': len(set(zip(self.layers, self.channels))),
            'low_state_coverage': int(np.unique(self.state_index).size),
            'error_convention': 'approx_minus_exact',
        }

    def select_split(self, split: str | Sequence[str]) -> 'DotTraceProfile':
        """Return a compact profile containing only the requested split(s)."""

        wanted = {split} if isinstance(split, str) else set(split)
        if not wanted:
            raise ValueError('at least one split must be selected')
        unknown = wanted.difference(self.split_names)
        if unknown:
            raise ValueError(f'trace has no split(s): {sorted(unknown)}')
        keep = np.asarray([value in wanted for value in self.splits], dtype=bool)
        old_groups = np.flatnonzero(keep)
        remap = np.full(self.group_count, -1, dtype=np.int64)
        remap[old_groups] = np.arange(old_groups.size, dtype=np.int64)
        edge_keep = keep[self.group_index]
        return DotTraceProfile(
            source=f'{self.source}#{",".join(sorted(wanted))}',
            sha256=self.sha256,
            ids=tuple(self.ids[i] for i in old_groups),
            layers=tuple(self.layers[i] for i in old_groups),
            channels=tuple(self.channels[i] for i in old_groups),
            splits=tuple(self.splits[i] for i in old_groups),
            scale=_readonly(self.scale[keep].copy()),
            sensitivity=_readonly(self.sensitivity[keep].copy()),
            normalizer=_readonly(self.normalizer[keep].copy()),
            group_index=_readonly(remap[self.group_index[edge_keep]]),
            state_index=_readonly(self.state_index[edge_keep].copy()),
            count=_readonly(self.count[edge_keep].copy()),
        )


@dataclass
class TorchDotTrace:
    """Device-resident sparse trace used by the differentiable objective."""

    ids: tuple[str, ...]
    layers: tuple[str, ...]
    channels: tuple[str, ...]
    splits: tuple[str, ...]
    scale: torch.Tensor
    sensitivity: torch.Tensor
    normalizer: torch.Tensor
    group_index: torch.Tensor
    state_index: torch.Tensor
    count: torch.Tensor
    layer_index: torch.Tensor
    channel_index: torch.Tensor
    layer_count: int
    channel_count: int

    @property
    def group_count(self) -> int:
        return len(self.ids)

    def delta_y(self, error_table: torch.Tensor) -> torch.Tensor:
        """Compute ``scale[g] * sum_s count[g,s] * error_table[s]``."""

        flat_error = error_table.reshape(-1)
        if flat_error.numel() != LL_STATES:
            raise ValueError(f'error_table must contain {LL_STATES} states')
        if not torch.is_floating_point(flat_error):
            flat_error = flat_error.to(self.count.dtype)
        edge_error = flat_error[self.state_index] * self.count.to(flat_error.dtype)
        summed = torch.zeros(
            self.group_count, dtype=flat_error.dtype, device=flat_error.device
        )
        summed.scatter_add_(0, self.group_index, edge_error)
        return summed * self.scale.to(flat_error.dtype)


@dataclass(frozen=True)
class DotProxyLossConfig:
    """Weights for the sensitivity-weighted output perturbation proxy."""

    huber_delta: float = 1.0
    output_weight: float = 1.0
    channel_bias_weight: float = 0.25
    layer_bias_weight: float = 0.10
    tail_weight: float = 0.10
    tail_fraction: float = 0.05
    mse_weight: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError('huber_delta must be finite and positive')
        if not math.isfinite(self.tail_fraction) or not 0 < self.tail_fraction <= 1:
            raise ValueError('tail_fraction must be in (0,1]')
        for name in (
            'output_weight', 'channel_bias_weight', 'layer_bias_weight',
            'tail_weight', 'mse_weight',
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class DotTraceMetrics:
    group_count: int
    mac_count: int
    proxy_score: float
    weighted_huber: float
    weighted_mae: float
    weighted_rmse: float
    weighted_bias: float
    max_abs_error: float
    p95_abs_error: float
    p99_abs_error: float
    layer_bias_rms: float
    channel_bias_rms: float
    tail_mean_abs_error: float

    @property
    def score(self) -> float:
        """Alias used by generic hard-candidate selectors."""

        return self.proxy_score

    def to_dict(self) -> dict:
        return asdict(self)

    def to_task_dict(self) -> dict:
        """Return metrics with the generic selector's required ``score`` key."""

        result = self.to_dict()
        result['score'] = self.proxy_score
        return result


def _require_number(record: Mapping, key: str, line_no: int) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'line {line_no}: {key} must be a JSON number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'line {line_no}: {key} must be finite')
    return value


def _require_text(record: Mapping, key: str, line_no: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'line {line_no}: {key} must be a nonempty string')
    return value.strip()


def load_dot_trace_jsonl(path: Path) -> DotTraceProfile:
    """Load and strictly validate ``signed88-dot-trace-v1`` JSONL."""

    path = Path(path).resolve()
    ids: list[str] = []
    layers: list[str] = []
    channels: list[str] = []
    splits: list[str] = []
    scale: list[float] = []
    sensitivity: list[float] = []
    normalizer: list[float] = []
    group_index: list[int] = []
    state_index: list[int] = []
    count: list[int] = []
    seen_ids: set[str] = set()
    metadata_seen = False

    with path.open('r', encoding='utf-8') as stream:
        for line_no, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'line {line_no}: invalid JSON: {exc.msg}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'line {line_no}: each JSONL row must be an object')
            record_type = record.get('type')
            if not metadata_seen:
                if record_type != 'metadata':
                    raise ValueError('first nonempty row must be metadata')
                if record.get('format') != DOT_TRACE_FORMAT:
                    raise ValueError(
                        f'line {line_no}: format must be {DOT_TRACE_FORMAT!r}'
                    )
                if record.get('state_count') != LL_STATES:
                    raise ValueError(f'line {line_no}: state_count must be {LL_STATES}')
                if record.get('error_convention') != 'approx_minus_exact':
                    raise ValueError(
                        f'line {line_no}: error_convention must be approx_minus_exact'
                    )
                metadata_seen = True
                continue
            if record_type == 'metadata':
                raise ValueError(f'line {line_no}: duplicate metadata row')
            if record_type != 'group':
                raise ValueError(f'line {line_no}: type must be group')

            group_id = _require_text(record, 'id', line_no)
            if group_id in seen_ids:
                raise ValueError(f'line {line_no}: duplicate group id {group_id!r}')
            seen_ids.add(group_id)
            layer = _require_text(record, 'layer', line_no)
            channel = _require_text(record, 'channel', line_no)
            split = _require_text(record, 'split', line_no)
            if split not in STANDARD_SPLITS:
                raise ValueError(
                    f'line {line_no}: split must be train, validation, or test'
                )
            group_scale = _require_number(record, 'scale', line_no)
            group_sensitivity = _require_number(record, 'sensitivity', line_no)
            group_normalizer = (
                _require_number(record, 'normalizer', line_no)
                if 'normalizer' in record else 1.0
            )
            if group_scale == 0:
                raise ValueError(f'line {line_no}: scale must be nonzero')
            if group_sensitivity <= 0:
                raise ValueError(f'line {line_no}: sensitivity must be positive')
            if group_normalizer <= 0:
                raise ValueError(f'line {line_no}: normalizer must be positive')

            counts = record.get('counts')
            if not isinstance(counts, list) or not counts:
                raise ValueError(f'line {line_no}: counts must be a nonempty list')
            local_states: set[int] = set()
            parsed_counts: list[tuple[int, int]] = []
            for pair_no, pair in enumerate(counts, 1):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(
                        f'line {line_no}: counts item {pair_no} must be [state,count]'
                    )
                state, repetitions = pair
                if (
                    isinstance(state, bool) or not isinstance(state, int)
                    or not 0 <= state < LL_STATES
                ):
                    raise ValueError(
                        f'line {line_no}: state in counts item {pair_no} '
                        f'must be an integer in [0,{LL_STATES - 1}]'
                    )
                if (
                    isinstance(repetitions, bool) or not isinstance(repetitions, int)
                    or repetitions <= 0
                ):
                    raise ValueError(
                        f'line {line_no}: count in counts item {pair_no} '
                        'must be a positive integer'
                    )
                if repetitions > np.iinfo(np.int64).max:
                    raise ValueError(f'line {line_no}: count exceeds int64')
                if state in local_states:
                    raise ValueError(f'line {line_no}: duplicate state {state}')
                local_states.add(state)
                parsed_counts.append((state, repetitions))

            group = len(ids)
            ids.append(group_id)
            layers.append(layer)
            channels.append(channel)
            splits.append(split)
            scale.append(group_scale)
            sensitivity.append(group_sensitivity)
            normalizer.append(group_normalizer)
            for state, repetitions in sorted(parsed_counts):
                group_index.append(group)
                state_index.append(state)
                count.append(repetitions)

    if not metadata_seen:
        raise ValueError('empty trace or missing metadata row')
    if not ids:
        raise ValueError('dot trace contains no groups')

    return DotTraceProfile(
        source=str(path),
        sha256=sha256_file(path),
        ids=tuple(ids),
        layers=tuple(layers),
        channels=tuple(channels),
        splits=tuple(splits),
        scale=_readonly(np.asarray(scale, dtype=np.float64)),
        sensitivity=_readonly(np.asarray(sensitivity, dtype=np.float64)),
        normalizer=_readonly(np.asarray(normalizer, dtype=np.float64)),
        group_index=_readonly(np.asarray(group_index, dtype=np.int64)),
        state_index=_readonly(np.asarray(state_index, dtype=np.int64)),
        count=_readonly(np.asarray(count, dtype=np.int64)),
    )


def load_objective_profile(
    path: Path, *, weight_column: str = 'auto'
) -> CalibrationProfile | DotTraceProfile:
    """Auto-load a legacy calibration CSV or a grouped JSONL trace."""

    path = Path(path)
    if path.suffix.lower() == '.csv':
        return load_calibration_csv(path, weight_column=weight_column)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        return load_dot_trace_jsonl(path)
    raise ValueError('objective data must be .csv, .jsonl, or .ndjson')


def build_dot_group_record(
    *,
    group_id: str,
    a: Sequence[int] | np.ndarray,
    b: Sequence[int] | np.ndarray,
    scale: float,
    sensitivity: float,
    layer: str,
    channel: str,
    split: str,
    normalizer: float = 1.0,
) -> dict:
    """Build one sparse JSONL group record from signed-int8 MAC operands."""

    a_array = np.asarray(a)
    b_array = np.asarray(b)
    if a_array.shape != b_array.shape or a_array.ndim != 1 or a_array.size == 0:
        raise ValueError('a and b must be nonempty one-dimensional arrays of equal shape')
    if not np.issubdtype(a_array.dtype, np.integer) or not np.issubdtype(
        b_array.dtype, np.integer
    ):
        raise ValueError('a and b must contain integers')
    if np.any(a_array < -128) or np.any(a_array > 127):
        raise ValueError('a contains a value outside signed int8')
    if np.any(b_array < -128) or np.any(b_array > 127):
        raise ValueError('b contains a value outside signed int8')
    states = ((a_array.astype(np.int64) & 63) * 64 + (
        b_array.astype(np.int64) & 63
    ))
    unique, repetitions = np.unique(states, return_counts=True)
    return {
        'type': 'group',
        'id': str(group_id),
        'layer': str(layer),
        'channel': str(channel),
        'split': str(split),
        'scale': float(scale),
        'sensitivity': float(sensitivity),
        'normalizer': float(normalizer),
        'counts': [[int(s), int(c)] for s, c in zip(unique, repetitions)],
    }


def write_dot_trace_jsonl(path: Path, records: Iterable[Mapping]) -> None:
    """Write group records with a canonical metadata header."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        'type': 'metadata',
        'format': DOT_TRACE_FORMAT,
        'state_count': LL_STATES,
        'error_convention': 'approx_minus_exact',
    }
    with path.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(header, ensure_ascii=False, separators=(',', ':')) + '\n')
        for record in records:
            stream.write(
                json.dumps(dict(record), ensure_ascii=False, separators=(',', ':')) + '\n'
            )


def _encode_keys(keys: Sequence[object]) -> tuple[np.ndarray, int]:
    mapping: dict[object, int] = {}
    encoded = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded[i] = mapping[key]
    return encoded, len(mapping)


def to_torch_dot_trace(
    profile: DotTraceProfile,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> TorchDotTrace:
    layer_index, layer_count = _encode_keys(profile.layers)
    channel_keys = tuple(zip(profile.layers, profile.channels))
    channel_index, channel_count = _encode_keys(channel_keys)
    return TorchDotTrace(
        ids=profile.ids,
        layers=profile.layers,
        channels=profile.channels,
        splits=profile.splits,
        # torch.tensor intentionally copies: profile arrays are read-only so that
        # an evaluator cannot accidentally mutate a loaded trace in place.
        scale=torch.tensor(profile.scale, dtype=dtype, device=device),
        sensitivity=torch.tensor(profile.sensitivity, dtype=dtype, device=device),
        normalizer=torch.tensor(profile.normalizer, dtype=dtype, device=device),
        group_index=torch.tensor(profile.group_index, dtype=torch.long, device=device),
        state_index=torch.tensor(profile.state_index, dtype=torch.long, device=device),
        count=torch.tensor(profile.count, dtype=dtype, device=device),
        layer_index=torch.tensor(layer_index, dtype=torch.long, device=device),
        channel_index=torch.tensor(channel_index, dtype=torch.long, device=device),
        layer_count=layer_count,
        channel_count=channel_count,
    )


def compute_delta_y_numpy(
    error_table: Sequence[float] | np.ndarray, profile: DotTraceProfile
) -> np.ndarray:
    """Hard evaluator for a 4096-entry ``approx - exact`` error table."""

    error = np.asarray(error_table, dtype=np.float64).reshape(-1)
    if error.size != LL_STATES:
        raise ValueError(f'error_table must contain {LL_STATES} states')
    if not np.all(np.isfinite(error)):
        raise ValueError('error_table contains a non-finite value')
    edge_error = error[profile.state_index] * profile.count.astype(np.float64)
    sums = np.bincount(
        profile.group_index, weights=edge_error, minlength=profile.group_count
    )
    return profile.scale * sums


def _torch_weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.clamp(torch.sum(weight), min=1e-12)


def _torch_segment_bias(
    value: torch.Tensor,
    weight: torch.Tensor,
    segment_index: torch.Tensor,
    segment_count: int,
) -> torch.Tensor:
    weighted_sum = torch.zeros(
        segment_count, dtype=value.dtype, device=value.device
    ).scatter_add(0, segment_index, value * weight)
    segment_weight = torch.zeros(
        segment_count, dtype=value.dtype, device=value.device
    ).scatter_add(0, segment_index, weight)
    mean = weighted_sum / torch.clamp(segment_weight, min=1e-12)
    return _torch_weighted_mean(mean.square(), segment_weight)


def compute_dot_proxy_loss(
    error_table: torch.Tensor,
    batch: TorchDotTrace,
    cfg: DotProxyLossConfig = DotProxyLossConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a differentiable, grouped output-perturbation proxy.

    ``error_table`` should be the model's 4096-entry low-product error table.
    Counts preserve accumulation inside each dot product.  ``normalizer`` makes
    output perturbations comparable across layers, while ``sensitivity``
    weights groups by their estimated impact on the downstream objective.
    """

    cfg.validate()
    delta_y = batch.delta_y(error_table)
    normalizer = batch.normalizer.to(delta_y.dtype)
    weight = batch.sensitivity.to(delta_y.dtype)
    normalized = delta_y / normalizer
    point_huber = F.huber_loss(
        normalized,
        torch.zeros_like(normalized),
        reduction='none',
        delta=float(cfg.huber_delta),
    )
    output_huber = _torch_weighted_mean(point_huber, weight)
    normalized_mse = _torch_weighted_mean(normalized.square(), weight)
    channel_bias = _torch_segment_bias(
        normalized, weight, batch.channel_index, batch.channel_count
    )
    layer_bias = _torch_segment_bias(
        normalized, weight, batch.layer_index, batch.layer_count
    )
    tail_count = max(1, int(math.ceil(batch.group_count * cfg.tail_fraction)))
    tail_index = torch.topk(torch.abs(normalized), k=tail_count, sorted=False).indices
    tail = _torch_weighted_mean(point_huber[tail_index], weight[tail_index])
    total = (
        cfg.output_weight * output_huber
        + cfg.channel_bias_weight * channel_bias
        + cfg.layer_bias_weight * layer_bias
        + cfg.tail_weight * tail
        + cfg.mse_weight * normalized_mse
    )
    terms = {
        'dot_output_huber': output_huber,
        'dot_channel_bias': channel_bias,
        'dot_layer_bias': layer_bias,
        'dot_tail_cvar': tail,
        'dot_normalized_mse': normalized_mse,
        'dot_weighted_mae': _torch_weighted_mean(torch.abs(delta_y), weight),
        'dot_weighted_bias': _torch_weighted_mean(delta_y, weight),
        'dot_max_abs': torch.max(torch.abs(delta_y)),
    }
    return total, terms


def _numpy_huber(value: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(
        absolute <= delta,
        0.5 * np.square(value),
        delta * (absolute - 0.5 * delta),
    )


def _numpy_weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(value * weight) / max(float(np.sum(weight)), 1e-15))


def _numpy_segment_bias(
    value: np.ndarray, weight: np.ndarray, keys: Sequence[object]
) -> float:
    index, count = _encode_keys(keys)
    segment_weight = np.bincount(index, weights=weight, minlength=count)
    segment_sum = np.bincount(index, weights=value * weight, minlength=count)
    segment_mean = segment_sum / np.maximum(segment_weight, 1e-15)
    return _numpy_weighted_mean(np.square(segment_mean), segment_weight)


def _weighted_quantile(value: np.ndarray, weight: np.ndarray, q: float) -> float:
    order = np.argsort(value, kind='stable')
    sorted_value = value[order]
    sorted_weight = weight[order]
    threshold = q * float(sorted_weight.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weight), threshold, side='left'))
    return float(sorted_value[min(index, sorted_value.size - 1)])


def evaluate_dot_trace(
    error_table: Sequence[float] | np.ndarray,
    profile: DotTraceProfile,
    *,
    split: str | Sequence[str] | None = None,
    cfg: DotProxyLossConfig = DotProxyLossConfig(),
) -> DotTraceMetrics:
    """Evaluate a hard INIT error table on validation/test dot traces."""

    cfg.validate()
    selected = profile.select_split(split) if split is not None else profile
    delta_y = compute_delta_y_numpy(error_table, selected)
    normalized = delta_y / selected.normalizer
    weight = selected.sensitivity
    point_huber = _numpy_huber(normalized, cfg.huber_delta)
    output_huber = _numpy_weighted_mean(point_huber, weight)
    normalized_mse = _numpy_weighted_mean(np.square(normalized), weight)
    channel_keys = tuple(zip(selected.layers, selected.channels))
    channel_bias = _numpy_segment_bias(normalized, weight, channel_keys)
    layer_bias = _numpy_segment_bias(normalized, weight, selected.layers)
    tail_count = max(1, int(math.ceil(selected.group_count * cfg.tail_fraction)))
    # Stable sort gives deterministic hard-candidate ranking even at a tie.
    tail_index = np.argsort(np.abs(normalized), kind='stable')[-tail_count:]
    tail_huber = _numpy_weighted_mean(point_huber[tail_index], weight[tail_index])
    score = (
        cfg.output_weight * output_huber
        + cfg.channel_bias_weight * channel_bias
        + cfg.layer_bias_weight * layer_bias
        + cfg.tail_weight * tail_huber
        + cfg.mse_weight * normalized_mse
    )

    def group_bias_rms(keys: Sequence[object]) -> float:
        index, count = _encode_keys(keys)
        segment_weight = np.bincount(index, weights=weight, minlength=count)
        segment_sum = np.bincount(index, weights=delta_y * weight, minlength=count)
        means = segment_sum / np.maximum(segment_weight, 1e-15)
        return math.sqrt(_numpy_weighted_mean(np.square(means), segment_weight))

    absolute = np.abs(delta_y)
    return DotTraceMetrics(
        group_count=selected.group_count,
        mac_count=selected.mac_count,
        proxy_score=float(score),
        weighted_huber=output_huber,
        weighted_mae=_numpy_weighted_mean(absolute, weight),
        weighted_rmse=math.sqrt(_numpy_weighted_mean(np.square(delta_y), weight)),
        weighted_bias=_numpy_weighted_mean(delta_y, weight),
        max_abs_error=float(absolute.max()),
        p95_abs_error=_weighted_quantile(absolute, weight, 0.95),
        p99_abs_error=_weighted_quantile(absolute, weight, 0.99),
        layer_bias_rms=group_bias_rms(selected.layers),
        channel_bias_rms=group_bias_rms(channel_keys),
        tail_mean_abs_error=_numpy_weighted_mean(absolute[tail_index], weight[tail_index]),
    )


def make_dot_trace_task_evaluator(
    profile: DotTraceProfile,
    *,
    split: str | Sequence[str] | None = 'validation',
    cfg: DotProxyLossConfig = DotProxyLossConfig(),
) -> Callable[[np.ndarray], dict]:
    """Build a callback compatible with ``selection_v2.HardwareSelector``."""

    selected = profile.select_split(split) if split is not None else profile

    def task_evaluator(error_table: np.ndarray) -> dict:
        return evaluate_dot_trace(error_table, selected, cfg=cfg).to_task_dict()

    return task_evaluator
