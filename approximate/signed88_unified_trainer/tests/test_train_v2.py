import json
import tempfile
import unittest
from pathlib import Path

if not (Path(__file__).resolve().parents[1] / 'train_v2.py').exists():
    raise unittest.SkipTest('V2 was intentionally removed; testing the V1 improvement branch')

from signed88.dot_trace import write_dot_trace_jsonl
from signed88.hardware import get_design
from train_v2 import main


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv"


class TrainV2IntegrationTest(unittest.TestCase):
    def _common(self, out):
        return [
            "--design", "balanced",
            "--device", "cpu",
            "--calibration-csv", str(CALIBRATION),
            "--out-dir", str(out),
            "--search-rounds", "1",
            "--top-k", "2",
            "--pair-top-k", "0",
            "--max-pairs", "0",
        ]

    def _assert_artifacts(self, out):
        summary = json.loads((out / "summary_v2.json").read_text(encoding="utf-8"))
        artifact = json.loads(
            (out / "best_signed88_inits.json").read_text(encoding="utf-8")
        )
        rtl = json.loads(
            (out / "best_rtl" / "trained_artifact.json").read_text(encoding="utf-8")
        )
        self.assertEqual(artifact["inits"], rtl["inits"])
        self.assertEqual(summary["artifacts"]["best_json"], "best_signed88_inits.json")
        self.assertTrue(summary["best_evaluation"]["feasible"])
        history = (out / "history_v2.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(history), 1)
        self.assertIn("effective_weights", json.loads(history[0]))
        return summary

    def test_histogram_pcgrad_smoke_and_rtl_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            self.assertEqual(main(self._common(out)), 0)
            summary = self._assert_artifacts(out)
            self.assertIsNone(summary["dot_trace"])
            self.assertEqual(summary["gradient_mode"], "pcgrad")

    def test_grouped_dot_trace_enters_train_and_hard_validation(self):
        design = get_design("balanced")
        error = design.hard_low_numpy(design.spec.base_inits)
        exact = [a * b for a in range(64) for b in range(64)]
        state = next(i for i, (approx, target) in enumerate(zip(error, exact)) if approx != target)
        records = [
            {
                "type": "group", "id": "train-0", "layer": "ffn", "channel": "0",
                "split": "train", "scale": 0.01, "sensitivity": 1.0,
                "normalizer": 1.0, "counts": [[state, 4]],
            },
            {
                "type": "group", "id": "validation-0", "layer": "ffn", "channel": "0",
                "split": "validation", "scale": 0.01, "sensitivity": 1.0,
                "normalizer": 1.0, "counts": [[state, 4]],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            out = root / "run"
            write_dot_trace_jsonl(trace, records)
            args = self._common(out) + ["--dot-trace", str(trace)]
            self.assertEqual(main(args), 0)
            summary = self._assert_artifacts(out)
            self.assertEqual(summary["dot_trace"]["train"]["group_count"], 1)
            self.assertIn("score", summary["best_evaluation"]["task"])
            row = json.loads(
                (out / "history_v2.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("dot", row["component_values"])
            self.assertIn("dot_output_huber", row["dot_terms"])

    def test_infeasible_random_start_is_not_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            args = [
                "--design", "fast",
                "--device", "cpu",
                "--calibration-csv", str(CALIBRATION),
                "--out-dir", str(out),
                "--init-mode", "random",
                "--seed", "811",
                "--search-rounds", "0",
            ]
            with self.assertRaisesRegex(ValueError, "safety envelope"):
                main(args)
            self.assertFalse((out / "best_signed88_inits.json").exists())
            self.assertFalse((out / "best_rtl").exists())

    def test_non_bit_exact_hard_forward_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            with self.assertRaisesRegex(ValueError, "hard-forward"):
                main(self._common(out) + ["--c-out", "50"])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
