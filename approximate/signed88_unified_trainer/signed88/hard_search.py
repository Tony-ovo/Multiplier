"""Gradient-guided, exact hard-INIT neighbourhood search.

The gradient is used only to rank candidate LUT bits.  Every accepted move is
evaluated again through the caller supplied ``evaluate(inits)`` callback, so
this module can be used with the legacy multiplier metrics, dot-product
validation, hidden-state error, or any other application-level objective.

The public search API deliberately depends only on the common ``BaseDesign``
surface (``spec.train_names``, ``spec.search_bits`` and ``normalize_inits``).
It therefore works for every registered signed88 topology without encoding
topology-specific module paths.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .common import hex_to_int, int_to_hex


BitRef = Tuple[str, int]
Evaluation = Any
Evaluate = Callable[[Mapping[str, str]], Evaluation]
Better = Callable[[Evaluation, Evaluation], bool]
Feasible = Callable[[Evaluation, Evaluation], bool]


@dataclass(frozen=True)
class RankedBit:
    """One hard INIT flip ranked by the local differentiable gradient.

    ``bit_gradient`` estimates d(loss)/d(discrete LUT bit).  A flip changes a
    hard bit by ``flip_direction`` (+1 for 0->1 and -1 for 1->0), hence
    ``predicted_delta`` estimates the loss change and ``predicted_gain`` is its
    negation.  Larger ``predicted_gain`` values are searched first.
    """

    table: str
    bit: int
    hard_value: int
    logit_gradient: float
    bit_gradient: float
    flip_direction: int
    predicted_delta: float
    predicted_gain: float

    @property
    def ref(self) -> BitRef:
        return self.table, self.bit


@dataclass(frozen=True)
class HardCandidate:
    """An exactly evaluated hard single- or pair-flip candidate."""

    flips: Tuple[BitRef, ...]
    inits: Dict[str, str]
    evaluation: Evaluation
    predicted_gain: float
    feasible: bool
    improves_reference: bool

    @property
    def kind(self) -> str:
        return "single" if len(self.flips) == 1 else "pair"


@dataclass(frozen=True)
class HardSearchResult:
    """Result of one gradient-ranked exact neighbourhood search step."""

    base_inits: Dict[str, str]
    base_evaluation: Evaluation
    rankings: Tuple[RankedBit, ...]
    candidates: Tuple[HardCandidate, ...]
    accepted: Optional[HardCandidate]
    pareto_front: Tuple[HardCandidate, ...]
    evaluations: int

    @property
    def best_inits(self) -> Dict[str, str]:
        return dict(self.accepted.inits if self.accepted is not None else self.base_inits)

    @property
    def best_evaluation(self) -> Evaluation:
        return self.accepted.evaluation if self.accepted is not None else self.base_evaluation


@dataclass(frozen=True)
class MetricConstraint:
    """An absolute and/or reference-relative bound on one metric.

    Metric paths may be nested, for example ``"validation.ppl"``.  A mapping
    key is preferred over an attribute at each path component.

    ``max_increase`` and ``max_decrease`` are absolute changes relative to the
    search-step reference evaluation.  This makes safety rules such as
    ``ER <= baseline_ER`` expressible as ``max_increase=0``.
    """

    metric: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    max_increase: Optional[float] = None
    max_decrease: Optional[float] = None
    atol: float = 1e-12


@dataclass(frozen=True)
class ParetoMetric:
    """One component of a Pareto comparison."""

    metric: str
    minimize: bool = True
    atol: float = 1e-12


def metric_value(evaluation: Evaluation, path: str) -> float:
    """Read a finite scalar from a mapping/object using a dotted path."""

    value = evaluation
    for component in str(path).split("."):
        if isinstance(value, Mapping):
            if component not in value:
                raise KeyError("metric path {!r} has no component {!r}".format(path, component))
            value = value[component]
        else:
            if not hasattr(value, component):
                raise AttributeError("metric path {!r} has no component {!r}".format(path, component))
            value = getattr(value, component)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("metric {!r} is not scalar".format(path))
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric {!r} is not finite: {!r}".format(path, result))
    return result


def make_constraint_filter(constraints: Sequence[MetricConstraint]) -> Feasible:
    """Build ``feasible(new, reference)`` from multiple metric constraints."""

    rules = tuple(constraints)

    def feasible(new: Evaluation, reference: Evaluation) -> bool:
        for rule in rules:
            value = metric_value(new, rule.metric)
            ref = metric_value(reference, rule.metric)
            tol = abs(float(rule.atol))
            if rule.minimum is not None and value < float(rule.minimum) - tol:
                return False
            if rule.maximum is not None and value > float(rule.maximum) + tol:
                return False
            if rule.max_increase is not None and value - ref > float(rule.max_increase) + tol:
                return False
            if rule.max_decrease is not None and ref - value > float(rule.max_decrease) + tol:
                return False
        return True

    return feasible


def make_pareto_better(metrics: Sequence[ParetoMetric]) -> Better:
    """Build a strict Pareto-dominance comparator.

    The returned comparator is a partial order: incomparable trade-offs return
    ``False`` in both directions.  At least one metric must improve outside its
    tolerance and no metric may degrade outside its tolerance.
    """

    goals = tuple(metrics)
    if not goals:
        raise ValueError("Pareto comparison needs at least one metric")

    def dominates(new: Evaluation, old: Evaluation) -> bool:
        strictly_better = False
        for goal in goals:
            nv = metric_value(new, goal.metric)
            ov = metric_value(old, goal.metric)
            tol = abs(float(goal.atol))
            if goal.minimize:
                if nv > ov + tol:
                    return False
                strictly_better = strictly_better or nv < ov - tol
            else:
                if nv < ov - tol:
                    return False
                strictly_better = strictly_better or nv > ov + tol
        return strictly_better

    return dominates


def _validate_gradient_tables(design, gradients: Mapping[str, Sequence[float]]) -> None:
    expected = set(design.spec.train_names)
    got = set(gradients)
    if got != expected:
        raise ValueError("gradient table mismatch: got={} expected={}".format(sorted(got), sorted(expected)))
    for name in design.spec.train_names:
        if len(gradients[name]) != 64:
            raise ValueError("{} gradient has length {}, expected 64".format(name, len(gradients[name])))


def rank_bit_gradients(
    design,
    inits: Mapping[str, str],
    bit_gradients: Mapping[str, Sequence[float]],
    *,
    logit_gradients: Optional[Mapping[str, Sequence[float]]] = None,
) -> Tuple[RankedBit, ...]:
    """Rank search bits from d(loss)/d(discrete-LUT-bit) values.

    This lower-level entry point is useful for preconditioned, accumulated, or
    conflict-projected gradients.  ``logit_gradients`` is optional provenance;
    when omitted it is reported as the same value as ``bit_gradients``.
    """

    normalized = design.normalize_inits(inits)
    _validate_gradient_tables(design, bit_gradients)
    if logit_gradients is not None:
        _validate_gradient_tables(design, logit_gradients)

    table_order = {name: index for index, name in enumerate(design.spec.train_names)}
    ranked = []
    for name in design.spec.train_names:
        value = hex_to_int(normalized[name])
        for raw_bit in design.spec.search_bits[name]:
            bit = int(raw_bit)
            if bit < 0 or bit >= 64:
                raise ValueError("invalid search bit {}[{}]".format(name, bit))
            grad = float(bit_gradients[name][bit])
            raw_grad = grad if logit_gradients is None else float(logit_gradients[name][bit])
            if not math.isfinite(grad) or not math.isfinite(raw_grad):
                raise ValueError("non-finite gradient for {}[{}]".format(name, bit))
            hard = (value >> bit) & 1
            direction = 1 if hard == 0 else -1
            predicted_delta = grad * direction
            ranked.append(
                RankedBit(
                    table=name,
                    bit=bit,
                    hard_value=hard,
                    logit_gradient=raw_grad,
                    bit_gradient=grad,
                    flip_direction=direction,
                    predicted_delta=predicted_delta,
                    predicted_gain=-predicted_delta,
                )
            )
    ranked.sort(key=lambda x: (-x.predicted_gain, -abs(x.bit_gradient), table_order[x.table], x.bit))
    return tuple(ranked)


def _terminal_logit_modules(model, train_names: Sequence[str]) -> Dict[str, torch.nn.Module]:
    """Resolve LUT modules by the final component of ``named_modules()``."""

    expected = set(train_names)
    found: Dict[str, torch.nn.Module] = {}
    paths: Dict[str, str] = {}
    for path, module in model.named_modules():
        terminal = path.rsplit(".", 1)[-1]
        if terminal not in expected or not hasattr(module, "logits"):
            continue
        if terminal in found:
            raise ValueError(
                "multiple logit modules end in {!r}: {!r} and {!r}".format(terminal, paths[terminal], path)
            )
        found[terminal] = module
        paths[terminal] = path
    missing = expected - set(found)
    if missing:
        raise ValueError("model has no terminal-name logit modules for {}".format(sorted(missing)))
    return found


def rank_model_gradients(
    model,
    design,
    inits: Optional[Mapping[str, str]] = None,
    *,
    normalization: str = "table",
    c_init: float = 1.0,
    sensitivity_floor: float = 1e-12,
    require_model_match: bool = True,
) -> Tuple[RankedBit, ...]:
    """Extract and rank LUT flip directions from ``model`` logit gradients.

    Call this after ``loss.backward()`` and before the optimizer clears the
    gradients.  Modules are resolved by matching the final component of their
    ``named_modules`` path to every ``design.spec.train_names`` entry.  This
    handles ``core.tables.*``, ``core.lut6.*`` and ``core.lut62.*`` uniformly.

    ``normalization='table'`` (default) divides the logit gradient by the exact
    local derivative of ``module.table(c_init)``.  The result estimates
    d(loss)/d(LUT-bit) and avoids ranking changes caused merely by unequal
    sigmoid confidence.  ``normalization='logit'`` uses raw logit gradients.
    """

    if normalization not in ("table", "logit"):
        raise ValueError("normalization must be 'table' or 'logit'")
    modules = _terminal_logit_modules(model, design.spec.train_names)
    model_inits = design.normalize_inits(model.hard_inits())
    normalized = model_inits if inits is None else design.normalize_inits(inits)
    if require_model_match and normalized != model_inits:
        raise ValueError("provided hard INITs do not match model.hard_inits()")

    raw_by_table: Dict[str, Sequence[float]] = {}
    bit_by_table: Dict[str, Sequence[float]] = {}
    floor = abs(float(sensitivity_floor))
    if floor == 0.0:
        raise ValueError("sensitivity_floor must be positive")

    for name in design.spec.train_names:
        module = modules[name]
        logits = module.logits
        if not isinstance(logits, torch.Tensor) or logits.numel() != 64:
            raise ValueError("{}.logits must contain 64 values".format(name))
        if logits.grad is None:
            raise ValueError("{}.logits has no gradient; call loss.backward() first".format(name))
        raw = logits.grad.detach().reshape(-1)
        if raw.numel() != 64:
            raise ValueError("{}.logits.grad must contain 64 values".format(name))
        if not bool(torch.all(torch.isfinite(raw))):
            raise ValueError("{}.logits.grad contains non-finite values".format(name))
        raw_cpu = raw.to(dtype=torch.float64, device="cpu")
        raw_by_table[name] = raw_cpu.tolist()

        if normalization == "logit":
            bit_by_table[name] = raw_cpu.tolist()
            continue
        if not hasattr(module, "table"):
            raise ValueError("{} has no table(c_init) method for table normalization".format(name))
        # Build a fresh, tiny graph.  autograd.grad does not alter .grad and
        # remains valid after the caller has already freed the loss graph.
        with torch.enable_grad():
            table = module.table(float(c_init))
            sensitivity = torch.autograd.grad(table.sum(), logits, retain_graph=False, create_graph=False)[0]
        sensitivity = sensitivity.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        normalized_grad = torch.zeros_like(raw_cpu)
        stable = torch.abs(sensitivity) >= floor
        normalized_grad[stable] = raw_cpu[stable] / sensitivity[stable]
        # A saturated bit normally has zero numerator and denominator.  If the
        # numerator is non-zero, clamp only the divisor to retain its direction.
        unstable_nonzero = (~stable) & (torch.abs(raw_cpu) >= floor)
        if bool(torch.any(unstable_nonzero)):
            sign = torch.where(sensitivity >= 0, torch.ones_like(sensitivity), -torch.ones_like(sensitivity))
            normalized_grad[unstable_nonzero] = raw_cpu[unstable_nonzero] / (sign[unstable_nonzero] * floor)
        bit_by_table[name] = normalized_grad.tolist()

    return rank_bit_gradients(
        design,
        normalized,
        bit_by_table,
        logit_gradients=raw_by_table,
    )


def flip_hard_bits(design, inits: Mapping[str, str], flips: Iterable[BitRef]) -> Dict[str, str]:
    """Return normalized INITs with the requested searchable bits toggled."""

    normalized = design.normalize_inits(inits)
    allowed = {name: set(int(bit) for bit in design.spec.search_bits[name]) for name in design.spec.train_names}
    ints = {name: hex_to_int(value) for name, value in normalized.items()}
    seen = set()
    for raw_name, raw_bit in flips:
        name, bit = str(raw_name), int(raw_bit)
        ref = (name, bit)
        if ref in seen:
            raise ValueError("duplicate flip {}[{}]".format(name, bit))
        seen.add(ref)
        if name not in allowed or bit not in allowed[name]:
            raise ValueError("{}[{}] is not a searchable bit".format(name, bit))
        ints[name] ^= 1 << bit
    return design.normalize_inits({name: int_to_hex(value) for name, value in ints.items()})


def _inits_key(design, inits: Mapping[str, str]) -> Tuple[int, ...]:
    return tuple(hex_to_int(inits[name]) for name in design.spec.train_names)


def _pareto_front(
    candidates: Sequence[HardCandidate],
    dominates: Better,
    reference: Evaluation,
) -> Tuple[HardCandidate, ...]:
    front = []
    for i, candidate in enumerate(candidates):
        if not candidate.feasible:
            continue
        # The current hard INIT participates in dominance even though the
        # result type intentionally contains only actual flip candidates.
        if dominates(reference, candidate.evaluation):
            continue
        if any(
            j != i and other.feasible and dominates(other.evaluation, candidate.evaluation)
            for j, other in enumerate(candidates)
        ):
            continue
        front.append(candidate)
    return tuple(front)


def gradient_ranked_hard_search(
    design,
    base_inits: Mapping[str, str],
    rankings: Sequence[RankedBit],
    *,
    evaluate: Evaluate,
    better: Better,
    feasible: Optional[Feasible] = None,
    top_k: int = 32,
    pair_top_k: int = 0,
    max_pairs: Optional[int] = None,
    pareto_dominates: Optional[Better] = None,
) -> HardSearchResult:
    """Evaluate top-ranked hard single flips and optional bit pairs exactly.

    ``evaluate`` may return any object.  ``better(new, old)`` defines genuine
    improvement and is used both to gate moves against the reference and to
    select among improving candidates.  ``feasible(new, reference)`` can add
    multiple safety constraints.  No candidate is accepted based on its
    gradient prediction alone.

    Pair candidates are combinations of the first ``pair_top_k`` ranked bits,
    ordered by summed predicted gain before the optional ``max_pairs`` cap.
    This permits exact recovery of pair synergies even when neither individual
    flip improves the hard objective.
    """

    if top_k < 0 or pair_top_k < 0:
        raise ValueError("top_k and pair_top_k must be non-negative")
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("max_pairs must be non-negative or None")
    normalized = design.normalize_inits(base_inits)
    allowed = {(name, int(bit)) for name in design.spec.train_names for bit in design.spec.search_bits[name]}

    ordered = []
    seen = set()
    for item in rankings:
        if item.ref not in allowed:
            raise ValueError("ranking contains non-searchable bit {}[{}]".format(item.table, item.bit))
        if item.ref in seen:
            raise ValueError("ranking contains duplicate bit {}[{}]".format(item.table, item.bit))
        if not math.isfinite(float(item.predicted_gain)):
            raise ValueError("ranking contains a non-finite predicted gain")
        seen.add(item.ref)
        ordered.append(item)
    ordered.sort(key=lambda x: (-x.predicted_gain, -abs(x.bit_gradient), x.table, x.bit))

    cache: Dict[Tuple[int, ...], Evaluation] = {}

    def evaluate_cached(inits: Mapping[str, str]) -> Evaluation:
        key = _inits_key(design, inits)
        if key not in cache:
            cache[key] = evaluate(dict(inits))
        return cache[key]

    base_evaluation = evaluate_cached(normalized)
    check_feasible = feasible if feasible is not None else (lambda new, reference: True)
    candidates = []

    def add_candidate(items: Sequence[RankedBit]) -> None:
        flips = tuple(item.ref for item in items)
        inits = flip_hard_bits(design, normalized, flips)
        evaluation = evaluate_cached(inits)
        is_feasible = bool(check_feasible(evaluation, base_evaluation))
        improves = is_feasible and bool(better(evaluation, base_evaluation))
        candidates.append(
            HardCandidate(
                flips=flips,
                inits=inits,
                evaluation=evaluation,
                predicted_gain=sum(float(item.predicted_gain) for item in items),
                feasible=is_feasible,
                improves_reference=improves,
            )
        )

    for item in ordered[:top_k]:
        add_candidate((item,))

    pair_items = ordered[:pair_top_k]
    pairs = list(itertools.combinations(pair_items, 2))
    pairs.sort(
        key=lambda pair: (
            -(pair[0].predicted_gain + pair[1].predicted_gain),
            pair[0].table,
            pair[0].bit,
            pair[1].table,
            pair[1].bit,
        )
    )
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    for pair in pairs:
        add_candidate(pair)

    improving = [candidate for candidate in candidates if candidate.improves_reference]
    accepted = None
    for candidate in improving:
        if accepted is None or better(candidate.evaluation, accepted.evaluation):
            accepted = candidate

    dominance = pareto_dominates if pareto_dominates is not None else better
    # The base participates in dominance, but is not represented as a fake
    # zero-flip candidate in the returned front.
    front = _pareto_front(candidates, dominance, base_evaluation)
    return HardSearchResult(
        base_inits=dict(normalized),
        base_evaluation=base_evaluation,
        rankings=tuple(ordered),
        candidates=tuple(candidates),
        accepted=accepted,
        pareto_front=front,
        evaluations=len(cache),
    )


__all__ = [
    "BitRef",
    "HardCandidate",
    "HardSearchResult",
    "MetricConstraint",
    "ParetoMetric",
    "RankedBit",
    "flip_hard_bits",
    "gradient_ranked_hard_search",
    "make_constraint_filter",
    "make_pareto_better",
    "metric_value",
    "rank_bit_gradients",
    "rank_model_gradients",
]
