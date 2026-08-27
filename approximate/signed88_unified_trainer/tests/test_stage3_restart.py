from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from signed88.common import ObjectiveWeights, hex_to_int, int_to_hex, read_json
from signed88.data import load_calibration_csv
from signed88.hard_search import rank_bit_gradients
from signed88.hardware import get_design

import train


ROOT = Path(__file__).resolve().parents[1]

# All 16 direct-random-logit runs converged to this hard Default INIT.  It is a
# deliberately important Stage-3 regression fixture: its exact GEMM score is
# poor because of coherent bias, while one searchable hard bit has a large and
# unambiguous exact improvement.
COMMON_RANDOM_LOGIT_INITS = {
    "cp_lut01": "64'h6AC06AC0A0A0A0A0",
    "cp_lut23": "64'hE622CC00EA40EAC0",
    "cp_lut45": "64'hE62A4C80EA40EA40",
    "cp_lut67": "64'h88800000444C8000",
}

COMMON_SCORE = 0.05012012142198031
BEST_SINGLE_SCORE = 0.004290861273213888
BEST_SINGLE = ("cp_lut01", 55)
BEST_PAIR_SCORE = 0.003678489478700591
BEST_PAIR = (("cp_lut01", 59), ("cp_lut01", 63))


def flipped(inits, *refs):
    result = dict(inits)
    for name, bit in refs:
        result[name] = int_to_hex(hex_to_int(result[name]) ^ (1 << bit))
    return result


def zero_gradient_ranking(design, inits):
    """Return every searchable bit without giving the answer via the ranking."""

    gradients = {
        name: [0.0] * 64
        for name in design.spec.train_names
    }
    return rank_bit_gradients(design, inits, gradients)


class ExactSingleStepRegressionTest(unittest.TestCase):
    def test_full_scan_selects_true_best_not_merely_first_ranked_bit(self):
        design = get_design("default")
        profile = load_calibration_csv(
            ROOT / "data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"
        )
        objective = ObjectiveWeights(bias_effective_k=4096.0)
        rankings = zero_gradient_ranking(design, COMMON_RANDOM_LOGIT_INITS)

        result = train.exact_single_step(
            design,
            COMMON_RANDOM_LOGIT_INITS,
            rankings,
            profile,
            objective,
            top_k=0,
        )

        self.assertIsNotNone(result.accepted)
        self.assertEqual(result.accepted.flips, (BEST_SINGLE,))
        self.assertAlmostEqual(
            result.base_evaluation.objective_score, COMMON_SCORE, places=12
        )
        self.assertAlmostEqual(
            result.accepted.evaluation.objective_score,
            BEST_SINGLE_SCORE,
            places=12,
        )
        expected = sum(
            len(design.spec.search_bits[name])
            for name in design.spec.train_names
        )
        self.assertEqual(expected, 56)
        self.assertEqual(len(result.candidates), expected)
        self.assertEqual(result.evaluations, expected + 1)

    def test_full_pair_scan_recovers_synergy_after_best_single(self):
        design = get_design("default")
        profile = load_calibration_csv(
            ROOT / "data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"
        )
        objective = ObjectiveWeights(bias_effective_k=4096.0)
        after_single = flipped(COMMON_RANDOM_LOGIT_INITS, BEST_SINGLE)
        rankings = zero_gradient_ranking(design, after_single)

        result = train.exact_hard_step(
            design,
            after_single,
            rankings,
            profile,
            objective,
            0,
            56,
            1540,
        )

        self.assertIsNotNone(result.accepted)
        self.assertEqual(result.accepted.kind, "pair")
        self.assertEqual(result.accepted.flips, BEST_PAIR)
        self.assertAlmostEqual(
            result.base_evaluation.objective_score, BEST_SINGLE_SCORE, places=12
        )
        self.assertAlmostEqual(
            result.accepted.evaluation.objective_score,
            BEST_PAIR_SCORE,
            places=12,
        )

        singles = [candidate for candidate in result.candidates if candidate.kind == "single"]
        pairs = [candidate for candidate in result.candidates if candidate.kind == "pair"]
        self.assertEqual(len(singles), 56)
        self.assertEqual(len(pairs), 1540)
        self.assertEqual(result.evaluations, 1 + 56 + 1540)
        self.assertFalse(any(candidate.improves_reference for candidate in singles))

        # Neither constituent is useful by itself; the accepted improvement is
        # genuinely a two-bit interaction that single-only Stage3 cannot see.
        for ref in BEST_PAIR:
            candidate = next(row for row in singles if row.flips == (ref,))
            self.assertGreater(
                candidate.evaluation.objective_score,
                result.base_evaluation.objective_score,
            )


class Stage3RestartCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="signed88_stage3_restart_")
        self.temp_root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _run(
        self,
        label: str,
        *,
        stage3_epochs: int,
        disable_exact: bool = False,
        pair_top_k: int = 0,
        pair_max_pairs: int = 1540,
        no_progress_rounds: int = 1,
    ):
        out_dir = self.temp_root / label
        source = self.temp_root / f"{label}_source.json"
        source.write_text(
            json.dumps({"design": "default", "inits": COMMON_RANDOM_LOGIT_INITS}),
            encoding="utf-8",
        )

        design = get_design("default")
        original_build_model = design.build_model
        build_calls = []

        def capture_build(inits, init_conf, noise_std):
            model = original_build_model(inits, init_conf, noise_std)
            build_calls.append(
                {
                    "inits": design.normalize_inits(inits),
                    "hard_inits": design.normalize_inits(model.hard_inits()),
                    "init_conf": float(init_conf),
                    "noise_std": float(noise_std),
                    "model_id": id(model),
                }
            )
            return model

        argv = [
            "train.py",
            "--design", "default",
            "--device", "cpu",
            "--seed", "17",
            "--init-mode", "json",
            "--base-inits-json", str(source),
            "--calibration-csv",
            str(ROOT / "data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"),
            "--out-dir", str(out_dir),
            "--rtl-template-root", str(ROOT / "rtl_sources"),
            "--stage1-epochs", "0",
            "--stage2-epochs", "0",
            "--stage3-epochs", str(stage3_epochs),
            "--stage3-restart-conf", "0.51",
            "--stage3-lr", "1e-9",
            "--stage3-block-epochs", "1",
            "--stage3-no-progress-rounds", str(no_progress_rounds),
            "--stage3-single-top-k", "0",
            "--stage3-pair-top-k", str(pair_top_k),
            "--stage3-pair-max-pairs", str(pair_max_pairs),
            "--eval-every", "1",
            "--print-every", "1",
            "--population-size", "0",
            "--population-epochs", "0",
            "--bias-effective-k", "4096",
        ]
        if disable_exact:
            argv.append("--disable-stage3-exact-single")

        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            design, "build_model", side_effect=capture_build
        ), mock.patch.object(sys, "stdout", io.StringIO()), mock.patch.object(
            sys, "stderr", io.StringIO()
        ):
            rc = train.main()
        self.assertEqual(rc, 0)

        history_path = out_dir / "history.jsonl"
        history = []
        if history_path.exists():
            history = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return {
            "out": out_dir,
            "build_calls": build_calls,
            "history": history,
            "initial": read_json(out_dir / "initial_signed88_inits.json"),
            "best": read_json(out_dir / "best_signed88_inits.json"),
            "rtl": read_json(out_dir / "best_rtl" / "trained_artifact.json"),
            "summary": read_json(out_dir / "summary.json"),
        }

    def test_stage3_restarts_from_exact_hard_best_and_accepts_best_single(self):
        # The no-progress event happens exactly on the final budgeted epoch.
        # It must still be reported as convergence, not overwritten with the
        # less informative ``budget_exhausted`` termination.
        run = self._run("exact", stage3_epochs=2)
        calls = run["build_calls"]

        # The initial JSON model and Stage-3 near-threshold model must be
        # distinct.  The exact-gradient ranking uses another fresh model, and
        # the next block restarts once more after the accepted hard move.
        self.assertGreaterEqual(len(calls), 3)
        restart = calls[1]
        self.assertEqual(restart["inits"], COMMON_RANDOM_LOGIT_INITS)
        self.assertEqual(restart["hard_inits"], COMMON_RANDOM_LOGIT_INITS)
        self.assertEqual(restart["init_conf"], 0.51)
        self.assertEqual(restart["noise_std"], 0.0)
        self.assertNotEqual(calls[0]["model_id"], restart["model_id"])

        best = run["best"]
        self.assertAlmostEqual(
            best["metrics"]["objective_score"], BEST_SINGLE_SCORE, places=12
        )
        changed = hex_to_int(best["inits"][BEST_SINGLE[0]]) ^ hex_to_int(
            COMMON_RANDOM_LOGIT_INITS[BEST_SINGLE[0]]
        )
        self.assertEqual(changed, 1 << BEST_SINGLE[1])
        for name in COMMON_RANDOM_LOGIT_INITS:
            if name != BEST_SINGLE[0]:
                self.assertEqual(best["inits"][name], COMMON_RANDOM_LOGIT_INITS[name])

        # With one epoch per block, call 2 is the round-0 ranking model and
        # call 3 is the next block.  It must restart from the accepted exact
        # hard state rather than resume stale round-0 logits or Adam moments.
        self.assertGreaterEqual(len(calls), 4)
        self.assertEqual(calls[3]["inits"], best["inits"])
        self.assertEqual(calls[3]["hard_inits"], best["inits"])
        self.assertEqual(calls[3]["init_conf"], 0.51)
        self.assertNotEqual(calls[1]["model_id"], calls[3]["model_id"])

        exact_rows = [
            row for row in run["history"]
            if row["stage"].startswith("stage3_exact_single_r")
        ]
        self.assertGreaterEqual(len(exact_rows), 1)
        accepted = [row for row in exact_rows if row.get("improved")]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            accepted[0]["extra"]["accepted_flips"], [["cp_lut01", 55]]
        )
        self.assertAlmostEqual(
            accepted[0]["metrics"]["objective_score"],
            BEST_SINGLE_SCORE,
            places=12,
        )

        # The deployable JSON, exported RTL metadata and summary must describe
        # the same exact hard state selected in history.
        self.assertEqual(run["rtl"]["inits"], best["inits"])
        self.assertEqual(run["summary"]["best_metrics"], best["metrics"])
        self.assertEqual(run["summary"]["best_stage"], accepted[0]["stage"])
        self.assertEqual(run["summary"]["train_args"]["stage3_restart_conf"], 0.51)
        self.assertEqual(run["summary"]["train_args"]["stage3_lr"], 1e-9)
        self.assertEqual(run["summary"]["stage3"]["requested_epochs"], 2)
        self.assertEqual(run["summary"]["stage3"]["completed_epochs"], 2)
        self.assertEqual(run["summary"]["stage3"]["rounds"], 2)
        self.assertEqual(run["summary"]["stage3"]["exact_single_accepts"], 1)
        self.assertEqual(run["summary"]["stage3"]["termination"], "no_hard_progress")

    def test_full_pair_stage_reaches_refined_default_and_records_provenance(self):
        # Round 0 takes the best single cp_lut01[55].  At that new state no
        # single improves, so round 1 must discover the synergistic [59]+[63]
        # pair.  The tiny STE learning rate keeps this test focused on exact
        # hard acceptance rather than optimizer threshold crossings.
        run = self._run(
            "pair",
            stage3_epochs=2,
            pair_top_k=56,
            pair_max_pairs=1540,
        )
        best = run["best"]
        expected = flipped(COMMON_RANDOM_LOGIT_INITS, BEST_SINGLE, *BEST_PAIR)
        self.assertEqual(best["inits"], expected)
        self.assertAlmostEqual(
            best["metrics"]["objective_score"], BEST_PAIR_SCORE, places=12
        )

        exact_rows = [
            row for row in run["history"]
            if row.get("event") == "exact_hard_search"
        ]
        self.assertEqual(len(exact_rows), 2)
        self.assertEqual(exact_rows[0]["stage"], "stage3_exact_single_r00")
        self.assertEqual(
            exact_rows[0]["extra"]["accepted_flips"], [["cp_lut01", 55]]
        )
        self.assertEqual(exact_rows[1]["stage"], "stage3_exact_pair_r01")
        self.assertEqual(
            {tuple(ref) for ref in exact_rows[1]["extra"]["accepted_flips"]},
            set(BEST_PAIR),
        )
        self.assertEqual(exact_rows[1]["extra"]["single_candidates"], 56)
        self.assertEqual(exact_rows[1]["extra"]["pair_candidates"], 1540)
        self.assertAlmostEqual(
            exact_rows[1]["metrics"]["objective_score"],
            BEST_PAIR_SCORE,
            places=12,
        )

        info = run["summary"]["stage3"]
        self.assertEqual(info["rounds"], 2)
        self.assertEqual(info["exact_single_accepts"], 1)
        self.assertEqual(info["exact_pair_accepts"], 1)
        self.assertEqual(info["exact_single_candidates"], 112)
        self.assertEqual(info["exact_pair_candidates"], 3080)
        self.assertEqual(info["exact_hard_evaluations"], 3194)
        self.assertEqual(run["summary"]["best_stage"], "stage3_exact_pair_r01")
        self.assertEqual(run["rtl"]["inits"], expected)

    def test_zero_stage3_preserves_legacy_zero_epoch_behavior(self):
        run = self._run("zero", stage3_epochs=0)
        self.assertEqual(len(run["build_calls"]), 1)
        self.assertEqual(run["history"], [])
        self.assertEqual(run["initial"]["inits"], COMMON_RANDOM_LOGIT_INITS)
        self.assertEqual(run["best"]["inits"], COMMON_RANDOM_LOGIT_INITS)
        self.assertEqual(run["rtl"]["inits"], COMMON_RANDOM_LOGIT_INITS)
        self.assertEqual(run["summary"]["best_stage"], "initial")
        self.assertEqual(
            run["summary"]["initial_metrics"], run["summary"]["best_metrics"]
        )

    def test_exact_scan_can_be_disabled_for_control_experiments(self):
        run = self._run("disabled", stage3_epochs=1, disable_exact=True)
        self.assertFalse(
            any(
                row["stage"].startswith("stage3_exact_single_r")
                for row in run["history"]
            )
        )
        self.assertEqual(run["best"]["inits"], COMMON_RANDOM_LOGIT_INITS)
        self.assertAlmostEqual(
            run["best"]["metrics"]["objective_score"], COMMON_SCORE, places=12
        )

    def test_no_progress_patience_continues_same_logits_and_optimizer(self):
        run = self._run(
            "patience",
            stage3_epochs=2,
            disable_exact=True,
            no_progress_rounds=2,
        )

        # One initial model plus one Stage-3 restart.  The second hard block
        # must continue that same model; rebuilding here would deterministically
        # repeat block zero and make the patience option meaningless.
        self.assertEqual(len(run["build_calls"]), 2)
        hard_rows = [
            row for row in run["history"]
            if row["stage"].startswith("stage3_hard_ste_r")
        ]
        self.assertEqual(len(hard_rows), 2)
        self.assertFalse(hard_rows[0]["extra"]["continued_model"])
        self.assertTrue(hard_rows[1]["extra"]["continued_model"])
        self.assertEqual(run["summary"]["stage3"]["model_restarts"], 1)
        self.assertEqual(
            run["summary"]["stage3"]["termination"], "no_hard_progress"
        )


if __name__ == "__main__":
    unittest.main()
