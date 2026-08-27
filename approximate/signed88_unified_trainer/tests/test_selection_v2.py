import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if not (Path(__file__).resolve().parents[1] / 'signed88' / 'selection_v2.py').exists():
    raise unittest.SkipTest('V2 was intentionally removed; testing the V1 improvement branch')

from signed88.common import hex_to_int, int_to_hex
from signed88.data import load_calibration_csv
from signed88.hard_search import gradient_ranked_hard_search, rank_bit_gradients
from signed88.hardware import get_design
from signed88.selection_v2 import (
    CandidateEvaluation,
    HardwareProxyWeights,
    HardwareSafety,
    HardwareSelector,
    SelectionConfig,
)


STATE_ONE = 1 * 64 + 1
STATE_TWO = 1 * 64 + 2
STATE_ZERO = 0
ROOT = Path(__file__).resolve().parents[1]


class SyntheticDesign:
    """Tiny INIT-controlled facade exercising the real exact metric path."""

    def __init__(self):
        self.spec = SimpleNamespace(
            name="selection_v2_synthetic",
            train_names=("lut",),
            search_bits={"lut": (0, 1, 2)},
        )
        self.base_inits = {"lut": int_to_hex(0)}
        al = np.repeat(np.arange(64, dtype=np.int32), 64)
        bl = np.tile(np.arange(64, dtype=np.int32), 64)
        self.exact_low = al * bl

    def normalize_inits(self, inits):
        if set(inits) != {"lut"}:
            raise ValueError("expected exactly one LUT")
        return {"lut": int_to_hex(hex_to_int(inits["lut"]))}

    def hard_low_numpy(self, inits):
        mode = hex_to_int(self.normalize_inits(inits)["lut"])
        error = np.zeros(64 * 64, dtype=np.int32)
        if mode == 0:
            # Canonical reference: ER=.4, WCE=1, workload bias=.4, zeroP=0.
            error[STATE_ONE] = 1
        elif mode == 1:
            # Better task score is possible, but workload ER doubles to .8.
            error[STATE_ONE] = 1
            error[STATE_TWO] = 1
        elif mode == 2:
            # Same absolute hardware metrics as reference, opposite error sign.
            error[STATE_ONE] = -1
        elif mode == 4:
            # Bit 2: preserves WCE and lowers ER, but corrupts a zero product.
            error[STATE_ZERO] = 1
        else:
            # Mode 3 (bits 0+1) and other combinations are not search targets.
            error[STATE_ONE] = 2
        return self.exact_low + error


def synthetic_profile():
    return SimpleNamespace(
        state_index=np.asarray([STATE_ONE, STATE_TWO, STATE_ZERO], dtype=np.int64),
        exact=np.asarray([1, 2, 0], dtype=np.int32),
        probability=np.asarray([0.4, 0.4, 0.2], dtype=np.float64),
        nonzero_probability=0.8,
        zero_probability=0.2,
    )


def eval_record(*, score, feasible, violation, er=0.0, wce=0.0):
    return CandidateEvaluation(
        score=float(score),
        feasible=bool(feasible),
        constraint_violation=float(violation),
        metrics={"workload_ER": float(er), "WCE": float(wce)},
        proxy={},
        task={},
    )


class HardwareSelectorSafetyTest(unittest.TestCase):
    def setUp(self):
        self.design = SyntheticDesign()
        self.profile = synthetic_profile()
        self.base = self.design.base_inits

    def test_reference_baseline_is_feasible(self):
        selector = HardwareSelector(self.design, self.profile, self.base)
        reference = selector.reference
        self.assertTrue(reference.feasible)
        self.assertEqual(reference.constraint_violation, 0.0)
        self.assertAlmostEqual(reference.metrics["workload_ER"], 0.4)
        self.assertEqual(reference.metrics["WCE"], 1)
        self.assertAlmostEqual(reference.metrics["workload_bias"], 0.4)
        self.assertEqual(reference.metrics["workload_zero_violation_probability"], 0.0)
        self.assertTrue(math.isfinite(reference.score))

    def test_er_constraint_is_relative_to_reference(self):
        selector = HardwareSelector(self.design, self.profile, self.base)
        candidate = selector.evaluate({"lut": int_to_hex(1)})
        self.assertFalse(candidate.feasible)
        self.assertAlmostEqual(candidate.proxy["er_limit"], 0.4)
        self.assertGreater(candidate.proxy["violation_workload_er"], 0.0)
        self.assertEqual(candidate.proxy["violation_wce"], 0.0)
        self.assertEqual(candidate.proxy["violation_zero_probability"], 0.0)

    def test_wce_constraint_uses_exact_uniform_hard_wce(self):
        selector = HardwareSelector(self.design, self.profile, self.base)
        # lut=3 selects the synthetic +2 WCE mode.
        candidate = selector.evaluate({"lut": int_to_hex(3)})
        self.assertFalse(candidate.feasible)
        self.assertEqual(candidate.metrics["WCE"], 2)
        self.assertEqual(candidate.proxy["wce_limit"], 1.0)
        self.assertGreater(candidate.proxy["violation_wce"], 0.0)
        self.assertEqual(candidate.proxy["violation_workload_er"], 0.0)

    def test_zero_product_constraint_is_strict_by_default(self):
        selector = HardwareSelector(self.design, self.profile, self.base)
        candidate = selector.evaluate({"lut": int_to_hex(4)})
        self.assertFalse(candidate.feasible)
        self.assertAlmostEqual(
            candidate.metrics["workload_zero_violation_probability"], 0.2
        )
        self.assertEqual(candidate.proxy["zero_probability_limit"], 0.0)
        self.assertGreater(candidate.proxy["violation_zero_probability"], 0.0)
        self.assertEqual(candidate.proxy["violation_workload_er"], 0.0)
        self.assertEqual(candidate.proxy["violation_wce"], 0.0)

    def test_explicit_absolute_bias_constraint(self):
        config = SelectionConfig(
            safety=HardwareSafety(
                max_er_ratio=2.0,
                max_wce_ratio=2.0,
                max_abs_bias=0.5,
                max_zero_violation_probability=0.0,
            )
        )
        selector = HardwareSelector(self.design, self.profile, self.base, config=config)
        candidate = selector.evaluate({"lut": int_to_hex(1)})
        self.assertTrue(selector.reference.feasible)
        self.assertFalse(candidate.feasible)
        self.assertAlmostEqual(candidate.metrics["workload_bias"], 0.8)
        self.assertEqual(candidate.proxy["bias_limit"], 0.5)
        self.assertGreater(candidate.proxy["violation_workload_abs_bias"], 0.0)
        self.assertEqual(candidate.proxy["violation_workload_er"], 0.0)
        self.assertEqual(candidate.proxy["violation_wce"], 0.0)

    def test_reference_evaluation_is_preseeded_in_cache(self):
        calls = []

        def task(error_table):
            calls.append(error_table.copy())
            # Deliberately stateful: a second reference call would alter score.
            return {"score": float(len(calls))}

        selector = HardwareSelector(
            self.design, self.profile, self.base, task_evaluator=task
        )
        self.assertEqual(len(calls), 1)
        evaluated = selector.evaluate(self.base)
        self.assertEqual(len(calls), 1)
        self.assertIs(evaluated, selector.reference)

    def test_nonfinite_or_negative_configuration_is_rejected(self):
        invalid = (
            SelectionConfig(weights=HardwareProxyWeights(workload_er=float("nan"))),
            SelectionConfig(safety=HardwareSafety(max_er_ratio=-1.0)),
            SelectionConfig(safety=HardwareSafety(max_wce_ratio=float("inf"))),
            SelectionConfig(safety=HardwareSafety(bias_floor=0.0)),
            SelectionConfig(mred_denominator_floor=float("nan")),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                HardwareSelector(self.design, self.profile, self.base, config=config)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            HardwareSelector(
                self.design,
                self.profile,
                self.base,
                task_evaluator=lambda error: {"score": -1.0},
            )


class RegisteredReferenceSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_calibration_csv(
            ROOT / "data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"
        )

    def test_every_registered_canonical_reference_is_feasible(self):
        for name in ("aggressive", "fast", "default", "balanced", "quality", "area"):
            with self.subTest(design=name):
                design = get_design(name)
                selector = HardwareSelector(
                    design, self.profile, design.spec.base_inits
                )
                self.assertTrue(selector.reference.feasible)
                self.assertEqual(selector.reference.constraint_violation, 0.0)
                self.assertIs(selector.evaluate(design.spec.base_inits), selector.reference)


class HardwareSelectorComparatorTest(unittest.TestCase):
    def test_better_is_feasibility_first(self):
        feasible_bad_score = eval_record(
            score=100.0, feasible=True, violation=0.0, er=0.4, wce=2
        )
        infeasible_good_score = eval_record(
            score=1.0, feasible=False, violation=0.1, er=0.2, wce=1
        )
        self.assertTrue(HardwareSelector.better(feasible_bad_score, infeasible_good_score))
        self.assertFalse(HardwareSelector.better(infeasible_good_score, feasible_bad_score))

        lower_violation = eval_record(
            score=100.0, feasible=False, violation=0.1, er=0.5, wce=2
        )
        higher_violation = eval_record(
            score=1.0, feasible=False, violation=0.2, er=0.1, wce=1
        )
        self.assertTrue(HardwareSelector.better(lower_violation, higher_violation))
        self.assertFalse(HardwareSelector.better(higher_violation, lower_violation))

    def test_task_score_changes_ranking_with_identical_hardware_magnitudes(self):
        design = SyntheticDesign()
        profile = synthetic_profile()

        def task(error_table):
            return {"score": 1.0 if error_table[STATE_ONE] < 0 else 10.0}

        selector = HardwareSelector(
            design, profile, design.base_inits, task_evaluator=task
        )
        reference = selector.reference
        candidate = selector.evaluate({"lut": int_to_hex(2)})
        for metric in (
            "workload_mae",
            "workload_rmse",
            "clipped_mred",
            "workload_er",
            "workload_abs_bias",
            "tail_cvar",
        ):
            self.assertAlmostEqual(candidate.proxy[metric], reference.proxy[metric])
        self.assertTrue(candidate.feasible)
        self.assertLess(candidate.score, reference.score)
        self.assertTrue(selector.better(candidate, reference))

    def test_comparator_plugs_directly_into_hard_search(self):
        design = SyntheticDesign()
        profile = synthetic_profile()

        def task(error_table):
            if error_table[STATE_TWO] != 0:
                return {"score": 0.01}  # bit 0: attractive but ER-infeasible
            if error_table[STATE_ONE] < 0:
                return {"score": 1.0}  # bit 1: feasible improvement
            return {"score": 10.0}

        selector = HardwareSelector(
            design, profile, design.base_inits, task_evaluator=task
        )
        gradients = {"lut": [0.0] * 64}
        gradients["lut"][0] = -2.0
        gradients["lut"][1] = -1.0
        ranking = rank_bit_gradients(design, design.base_inits, gradients)
        result = gradient_ranked_hard_search(
            design,
            design.base_inits,
            ranking,
            evaluate=selector.evaluate,
            better=selector.better,
            feasible=lambda new, reference: bool(new.feasible),
            top_k=2,
        )

        by_flip = {candidate.flips: candidate for candidate in result.candidates}
        self.assertLess(by_flip[(("lut", 0),)].evaluation.score, selector.reference.score)
        self.assertFalse(by_flip[(("lut", 0),)].feasible)
        self.assertEqual(result.accepted.flips, (("lut", 1),))
        self.assertTrue(result.accepted.evaluation.feasible)


if __name__ == "__main__":
    unittest.main()
