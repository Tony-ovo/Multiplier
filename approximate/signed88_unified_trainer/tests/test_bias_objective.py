import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from signed88.common import GEMM_OBJECTIVE_SCHEMA, LL_STATES, ObjectiveWeights
from signed88.data import CalibrationProfile, TorchCalibration, load_calibration_csv, to_torch
from signed88.hardware import get_design
from signed88.losses import LossConfig, compute_loss
from signed88.metrics import evaluate_design


ROOT = Path(__file__).resolve().parents[1]


def _state_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int64) & 63) * 64 + (b.astype(np.int64) & 63)).astype(np.int64)


def _profile(a_values, b_values, probabilities=None) -> CalibrationProfile:
    a = np.asarray(a_values, dtype=np.int16)
    b = np.asarray(b_values, dtype=np.int16)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be equally sized one-dimensional arrays")
    if probabilities is None:
        probability = np.full(a.size, 1.0 / a.size, dtype=np.float64)
    else:
        probability = np.asarray(probabilities, dtype=np.float64)
        probability = probability / probability.sum()

    exact = (a.astype(np.int32) * b.astype(np.int32)).astype(np.int32)
    state_index = _state_index(a, b)
    state_probability = np.zeros(LL_STATES, dtype=np.float64)
    red_weight = np.zeros(LL_STATES, dtype=np.float64)
    zero_probability = np.zeros(LL_STATES, dtype=np.float64)
    np.add.at(state_probability, state_index, probability)
    nonzero = exact != 0
    np.add.at(
        red_weight,
        state_index[nonzero],
        probability[nonzero] / np.abs(exact[nonzero]),
    )
    np.add.at(zero_probability, state_index[~nonzero], probability[~nonzero])
    return CalibrationProfile(
        source="synthetic",
        sha256="synthetic",
        weight_column="probability",
        row_count=int(a.size),
        raw_weight_sum=1.0,
        a=a,
        b=b,
        exact=exact,
        probability=probability,
        state_index=state_index,
        state_probability=state_probability,
        red_weight_by_state=red_weight,
        zero_probability_by_state=zero_probability,
        nonzero_probability=float(probability[nonzero].sum()),
        zero_probability=float(probability[~nonzero].sum()),
    )


def _torch_batch(profile: CalibrationProfile) -> TorchCalibration:
    return TorchCalibration(
        a=torch.as_tensor(profile.a.astype(np.int64)),
        b=torch.as_tensor(profile.b.astype(np.int64)),
        exact=torch.as_tensor(profile.exact.astype(np.float32)),
        probability=torch.as_tensor(profile.probability.astype(np.float32)),
        state_index=torch.as_tensor(profile.state_index, dtype=torch.long),
        state_probability=torch.as_tensor(profile.state_probability.astype(np.float32)),
        red_weight_by_state=torch.as_tensor(profile.red_weight_by_state.astype(np.float32)),
        zero_probability_by_state=torch.as_tensor(profile.zero_probability_by_state.astype(np.float32)),
        nonzero_probability=profile.nonzero_probability,
        zero_probability=profile.zero_probability,
    )


class _ErrorTableModel(nn.Module):
    """Minimal differentiable model with the same signed-error semantics as the RTL."""

    def __init__(self, profile: CalibrationProfile, row_errors) -> None:
        super().__init__()
        al = torch.arange(64, dtype=torch.int64).repeat_interleave(64)
        bl = torch.arange(64, dtype=torch.int64).repeat(64)
        self.register_buffer("grid_al", al)
        self.register_buffer("grid_bl", bl)
        self.register_buffer("grid_exact_ll", (al * bl).to(torch.float32))
        self.error_table = nn.Parameter(torch.zeros(LL_STATES, dtype=torch.float32))

        assigned = {}
        for state, error in zip(profile.state_index, row_errors):
            state = int(state)
            error = float(error)
            if state in assigned and assigned[state] != error:
                raise ValueError("one low state cannot have two different hardware errors")
            assigned[state] = error
        with torch.no_grad():
            for state, error in assigned.items():
                self.error_table[state] = error

    def forward_signed_rows(
        self,
        a,
        b,
        state_index,
        *,
        c_init,
        c_out,
        hard_middle,
    ):
        del c_init, c_out, hard_middle
        low_value = self.grid_exact_ll + self.error_table
        exact_signed = (a * b).to(torch.float32)
        approx_signed = exact_signed + self.error_table[state_index]
        exact_ll_i = self.grid_exact_ll.to(torch.int64)
        low_bits = [((exact_ll_i >> bit) & 1).to(torch.float32) for bit in range(12)]
        return approx_signed, low_value, low_bits

    def bin_reg(self):
        return self.error_table.sum() * 0.0


class _HardTableDesign:
    spec = SimpleNamespace(name="synthetic-bias-design")

    def __init__(self, tables):
        al = np.repeat(np.arange(64, dtype=np.int32), 64)
        bl = np.tile(np.arange(64, dtype=np.int32), 64)
        exact_ll = al * bl
        self.tables = {
            name: exact_ll + np.asarray(error, dtype=np.int32)
            for name, error in tables.items()
        }

    def hard_low_numpy(self, inits):
        return self.tables[inits["candidate"]]


def _loss_terms(profile, row_errors, *, effective_k, conditional_weight):
    model = _ErrorTableModel(profile, row_errors)
    cfg = replace(
        LossConfig(),
        calibration_mix=1.0,
        er_weight=0.0,
        bias_weight=1.0,
        conditional_bias_weight=conditional_weight,
        bias_effective_k=effective_k,
        zero_weight=0.0,
        symmetry_weight=0.0,
        bin_weight=0.0,
    )
    loss, terms = compute_loss(
        model,
        _torch_batch(profile),
        c_init=1.0,
        c_out=1.0,
        hard_middle=True,
        bit_weight=0.0,
        mae_weight=0.0,
        mred_weight=0.0,
        er_temperature=1.0,
        cfg=cfg,
    )
    return model, loss, terms


def _objective(*, global_bias, conditional_bias, effective_k):
    return replace(
        ObjectiveWeights(),
        workload_mred=0.0,
        workload_er=0.0,
        workload_ned=0.0,
        workload_bias=global_bias,
        workload_conditional_bias=conditional_bias,
        uniform_mred=0.0,
        bias_effective_k=effective_k,
    )


class BiasLossFormulaTest(unittest.TestCase):
    def test_global_squared_bias_and_gemm_nmse_formula(self):
        profile = _profile([1, 2], [1, 1])
        _, loss, terms = _loss_terms(
            profile,
            [2.0, 2.0],
            effective_k=5.0,
            conditional_weight=0.0,
        )

        # D=E[y^2]=(1^2+2^2)/2=2.5, E[e^2]=4, mu=2.
        denominator = 2.5
        expected_nmse = 4.0 / denominator
        expected_global_bias = 2.0**2 / denominator
        expected_gemm_nmse = expected_nmse + (5.0 - 1.0) * expected_global_bias
        self.assertAlmostEqual(float(terms["workload_nmse"]), expected_nmse, places=6)
        self.assertAlmostEqual(
            float(terms["workload_bias_squared"]), expected_global_bias, places=6
        )
        self.assertAlmostEqual(
            float(terms["workload_gemm_nmse"]), expected_gemm_nmse, places=6
        )
        self.assertAlmostEqual(float(loss), expected_gemm_nmse, places=6)

    def test_effective_k_one_removes_all_accumulation_penalties(self):
        profile = _profile([1, 2], [1, 1])
        _, loss, terms = _loss_terms(
            profile,
            [2.0, 2.0],
            effective_k=1.0,
            conditional_weight=7.0,
        )
        self.assertAlmostEqual(
            float(terms["workload_gemm_nmse"]),
            float(terms["workload_nmse"]),
            places=7,
        )
        self.assertAlmostEqual(float(loss), float(terms["workload_nmse"]), places=7)

    def test_global_zero_does_not_hide_operand_conditional_bias(self):
        # Ordering: (a1,b1), (a2,b1), (a1,b2), (a2,b2).
        profile = _profile([1, 2, 1, 2], [1, 1, 2, 2])
        _, directional_loss, directional_terms = _loss_terms(
            profile,
            [2.0, 2.0, -2.0, -2.0],
            effective_k=5.0,
            conditional_weight=1.0,
        )
        _, checkerboard_loss, checkerboard_terms = _loss_terms(
            profile,
            [2.0, -2.0, -2.0, 2.0],
            effective_k=5.0,
            conditional_weight=1.0,
        )

        # Both candidates have mu=0 and identical MSE.  For the first one,
        # Ba=0, Bb=4, so CE=(0.5*(Ba+Bb)-mu^2)/D=2/6.25=0.32.
        self.assertAlmostEqual(
            float(directional_terms["workload_bias_squared"]), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(directional_terms["workload_conditional_bias_excess"]),
            2.0 / 6.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(checkerboard_terms["workload_conditional_bias_excess"]),
            0.0,
            places=7,
        )
        self.assertGreater(
            float(directional_terms["workload_gemm_nmse"]),
            float(checkerboard_terms["workload_gemm_nmse"]),
        )
        self.assertAlmostEqual(
            float(directional_loss),
            float(directional_terms["workload_gemm_nmse"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(checkerboard_loss),
            float(checkerboard_terms["workload_gemm_nmse"]),
            places=6,
        )

    def test_zero_signal_uses_unit_energy_floor_and_stays_finite(self):
        profile = _profile([0], [5])
        _, loss, terms = _loss_terms(
            profile,
            [2.0],
            effective_k=5.0,
            conditional_weight=1.0,
        )
        # D=max(E[exact^2],1)=1.  Conditional excess is zero because there is
        # only one populated a group and one populated b group.
        self.assertAlmostEqual(float(terms["workload_nmse"]), 4.0, places=7)
        self.assertAlmostEqual(float(terms["workload_bias_squared"]), 4.0, places=7)
        self.assertAlmostEqual(
            float(terms["workload_conditional_bias_excess"]), 0.0, places=7
        )
        self.assertAlmostEqual(float(terms["workload_gemm_nmse"]), 20.0, places=7)
        self.assertTrue(torch.isfinite(loss))

    def test_gemm_nmse_gradient_pushes_signed_mean_toward_zero(self):
        profile = _profile([1, 2], [1, 1], [0.25, 0.75])
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                model, loss, terms = _loss_terms(
                    profile,
                    [sign * 2.0, sign * 4.0],
                    effective_k=3.0,
                    conditional_weight=0.0,
                )
                self.assertAlmostEqual(
                    float(loss), float(terms["workload_gemm_nmse"]), places=6
                )
                loss.backward()

                # L=(E[e^2]+(K-1)E[e]^2)/D.  Therefore
                # dL/de_i=2*p_i*(e_i+(K-1)*mu)/D.
                probability = np.asarray([0.25, 0.75], dtype=np.float64)
                error = sign * np.asarray([2.0, 4.0], dtype=np.float64)
                exact = np.asarray([1.0, 2.0], dtype=np.float64)
                mean_error = float(np.sum(probability * error))
                denominator = float(np.sum(probability * np.square(exact)))
                expected = (
                    2.0
                    * probability
                    * (error + (3.0 - 1.0) * mean_error)
                    / denominator
                )
                selected_gradient = model.error_table.grad[
                    torch.as_tensor(profile.state_index, dtype=torch.long)
                ].detach().numpy()
                np.testing.assert_allclose(
                    selected_gradient, expected, rtol=1e-6, atol=1e-6
                )
                self.assertTrue(np.all(sign * selected_gradient > 0.0))


class HardSoftBiasEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_calibration_csv(
            ROOT / "data" / "w8a8_calibration_hist_smoke_pcalib_nonzero.csv"
        )
        cls.batch = to_torch(cls.profile, torch.device("cpu"))

    def test_actual_hard_models_match_numpy_bias_metrics(self):
        conditional_weight = 0.03
        effective_k = 7.0
        cfg = replace(
            LossConfig(),
            calibration_mix=1.0,
            conditional_bias_weight=conditional_weight,
            bias_effective_k=effective_k,
        )
        objective = replace(
            ObjectiveWeights(),
            workload_conditional_bias=conditional_weight,
            bias_effective_k=effective_k,
        )
        for design_name in ("quality", "balanced", "fast", "area", "aggressive"):
            with self.subTest(design=design_name):
                design = get_design(design_name)
                model = design.build_model(design.spec.base_inits, 0.999, 0.0).cpu()
                _, terms = compute_loss(
                    model,
                    self.batch,
                    c_init=1.0,
                    c_out=1.0,
                    hard_middle=True,
                    bit_weight=0.0,
                    mae_weight=0.0,
                    mred_weight=0.0,
                    er_temperature=1.0,
                    cfg=cfg,
                )
                hard = evaluate_design(
                    design, design.spec.base_inits, self.profile, objective
                )
                pairs = (
                    ("workload_mse", hard.workload_MSE),
                    ("workload_nmse", hard.workload_NMSE),
                    ("workload_signed_bias", hard.workload_bias),
                    (
                        "workload_bias_squared",
                        hard.workload_bias_squared_normalized,
                    ),
                    (
                        "workload_conditional_bias_a_rms",
                        hard.workload_conditional_bias_a_rms,
                    ),
                    (
                        "workload_conditional_bias_b_rms",
                        hard.workload_conditional_bias_b_rms,
                    ),
                    (
                        "workload_conditional_bias_excess",
                        hard.workload_conditional_bias_excess_normalized,
                    ),
                    ("workload_gemm_nmse", hard.workload_gemm_NMSE),
                )
                for term_name, expected in pairs:
                    self.assertTrue(
                        np.isclose(
                            float(terms[term_name]),
                            float(expected),
                            rtol=2e-5,
                            atol=2e-7,
                        ),
                        f"{design_name} {term_name}: {float(terms[term_name])} != {expected}",
                    )


class HardBiasObjectiveTest(unittest.TestCase):
    @staticmethod
    def _error_table(profile, row_errors):
        error = np.zeros(LL_STATES, dtype=np.int32)
        for state, value in zip(profile.state_index, row_errors):
            error[int(state)] = int(value)
        return error

    def test_global_bias_changes_hard_candidate_ranking(self):
        profile = _profile([1, 2], [1, 1])
        design = _HardTableDesign(
            {
                "one_direction": self._error_table(profile, [2, 2]),
                "balanced": self._error_table(profile, [2, -2]),
            }
        )
        objective = _objective(global_bias=1.0, conditional_bias=0.0, effective_k=5.0)
        directional = evaluate_design(design, {"candidate": "one_direction"}, profile, objective)
        balanced = evaluate_design(design, {"candidate": "balanced"}, profile, objective)

        self.assertAlmostEqual(directional.workload_bias, 2.0, places=7)
        self.assertAlmostEqual(balanced.workload_bias, 0.0, places=7)
        self.assertGreater(directional.objective_score, balanced.objective_score)
        self.assertAlmostEqual(
            directional.objective_score - balanced.objective_score,
            (5.0 - 1.0) * (2.0**2 / 2.5),
            places=7,
        )

    def test_conditional_bias_changes_hard_candidate_ranking(self):
        profile = _profile([1, 2, 1, 2], [1, 1, 2, 2])
        design = _HardTableDesign(
            {
                "conditional": self._error_table(profile, [2, 2, -2, -2]),
                "checkerboard": self._error_table(profile, [2, -2, -2, 2]),
            }
        )
        objective = _objective(global_bias=0.0, conditional_bias=1.0, effective_k=5.0)
        conditional = evaluate_design(design, {"candidate": "conditional"}, profile, objective)
        checkerboard = evaluate_design(design, {"candidate": "checkerboard"}, profile, objective)

        self.assertAlmostEqual(conditional.workload_bias, 0.0, places=7)
        self.assertAlmostEqual(checkerboard.workload_bias, 0.0, places=7)
        self.assertGreater(conditional.objective_score, checkerboard.objective_score)
        self.assertAlmostEqual(
            conditional.objective_score - checkerboard.objective_score,
            (5.0 - 1.0) * (2.0 / 6.25),
            places=7,
        )


class BiasCliArtifactIntegrationTest(unittest.TestCase):
    def test_train_verify_and_refine_preserve_bias_objective(self):
        with tempfile.TemporaryDirectory(prefix="signed88_bias_cli_") as temp_dir:
            temp = Path(temp_dir)
            train_out = temp / "train"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train.py"),
                    "--design",
                    "quality",
                    "--device",
                    "cpu",
                    "--init-mode",
                    "baseline",
                    "--out-dir",
                    str(train_out),
                    "--stage1-epochs",
                    "0",
                    "--stage2-epochs",
                    "0",
                    "--stage3-epochs",
                    "0",
                    "--population-size",
                    "0",
                    "--mse-weight",
                    "0.75",
                    "--bias-weight",
                    "1.25",
                    "--conditional-bias-weight",
                    "0.2",
                    "--bias-effective-k",
                    "7",
                    "--score-mse-weight",
                    "2.5",
                    "--score-bias-weight",
                    "3.5",
                    "--score-conditional-bias-weight",
                    "0.125",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            summary = json.loads((train_out / "summary.json").read_text(encoding="utf-8"))
            objective = summary["objective_weights"]
            self.assertEqual(summary["objective_schema"], GEMM_OBJECTIVE_SCHEMA)
            self.assertEqual(objective["workload_nmse"], 2.5)
            self.assertEqual(objective["workload_bias"], 3.5)
            self.assertEqual(objective["workload_conditional_bias"], 0.125)
            self.assertEqual(objective["bias_effective_k"], 7.0)
            self.assertEqual(summary["train_args"]["mse_weight"], 0.75)
            self.assertEqual(summary["train_args"]["bias_weight"], 1.25)
            self.assertEqual(summary["train_args"]["conditional_bias_weight"], 0.2)

            best_json = train_out / "best_signed88_inits.json"
            best = json.loads(best_json.read_text(encoding="utf-8"))
            rtl_artifact = json.loads(
                (train_out / "best_rtl" / "trained_artifact.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(best["objective_weights"], objective)
            self.assertEqual(best["objective_schema"], GEMM_OBJECTIVE_SCHEMA)
            self.assertEqual(rtl_artifact["objective_weights"], objective)
            self.assertEqual(rtl_artifact["objective_schema"], GEMM_OBJECTIVE_SCHEMA)
            for name in (
                "workload_MSE",
                "workload_NMSE",
                "workload_bias_squared_normalized",
                "workload_conditional_bias_excess_normalized",
                "workload_gemm_NMSE",
            ):
                self.assertIn(name, best["metrics"])

            # No score override: verify must inherit all new objective fields from
            # the artifact, otherwise the saved GEMM metrics no longer match.
            verified = subprocess.run(
                [sys.executable, str(ROOT / "verify.py"), "--inits-json", str(best_json)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertIn("[artifact] PASS", verified.stdout)

            refine_out = temp / "refine"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "refine.py"),
                    "--base-inits-json",
                    str(best_json),
                    "--out-dir",
                    str(refine_out),
                    "--bit-rounds",
                    "0",
                    "--pair-rounds",
                    "0",
                    "--basin-iters",
                    "0",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            refined = json.loads(
                (refine_out / "best_signed88_inits.json").read_text(encoding="utf-8")
            )
            self.assertEqual(refined["objective_weights"], objective)

            # A V1 artifact used workload_bias for |E[e]|/16384.  It must not be
            # silently reinterpreted as the new (K-1)*E[e]^2/D coefficient.
            legacy_json = temp / "legacy.json"
            legacy = dict(best)
            legacy.pop("objective_schema", None)
            legacy["objective_weights"] = {
                "workload_mred": 1.0,
                "workload_er": 0.25,
                "workload_ned": 0.10,
                "workload_bias": 0.05,
                "uniform_mred": 0.05,
            }
            # Real legacy artifacts do not contain any of the new GEMM metrics.
            legacy.pop("metrics", None)
            legacy_json.write_text(json.dumps(legacy), encoding="utf-8")
            legacy_out = temp / "legacy_refine"
            migrated = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "refine.py"),
                    "--base-inits-json",
                    str(legacy_json),
                    "--out-dir",
                    str(legacy_out),
                    "--bit-rounds",
                    "0",
                    "--pair-rounds",
                    "0",
                    "--basin-iters",
                    "0",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertIn("legacy artifact detected", migrated.stdout)
            migrated_artifact = json.loads(
                (legacy_out / "best_signed88_inits.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                migrated_artifact["objective_weights"], ObjectiveWeights().__dict__
            )
            self.assertEqual(
                migrated_artifact["objective_schema"], GEMM_OBJECTIVE_SCHEMA
            )


if __name__ == "__main__":
    unittest.main()
