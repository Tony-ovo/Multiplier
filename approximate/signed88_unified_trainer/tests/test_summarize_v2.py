import csv
import json
import tempfile
import unittest
from pathlib import Path

if not (Path(__file__).resolve().parents[1] / 'summarize_v2.py').exists():
    raise unittest.SkipTest('V2 was intentionally removed; testing the V1 improvement branch')

import summarize_v2


def write_run(
    root,
    name,
    *,
    feasible,
    violation,
    score,
    artifact_rel="artifacts/best.json",
    rtl_rel="rtl/best",
):
    run = Path(root) / name
    artifact = run / artifact_rel
    rtl = run / rtl_rel
    artifact.parent.mkdir(parents=True, exist_ok=True)
    rtl.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({"marker": name, "score": score}), encoding="utf-8"
    )
    summary = {
        "format": "signed88-train-v2-summary-v1",
        "design": "balanced",
        "seed": 123,
        # These deliberately stale legacy-style fields must never be used.
        "best_json": "/old/moved/run/best_signed88_inits.json",
        "best_rtl": "/old/moved/run/best_rtl",
        "best_evaluation": {
            "score": score,
            "feasible": feasible,
            "constraint_violation": violation,
            "metrics": {
                "workload_MRED": score / 10.0,
                "workload_ER": 0.1,
                "workload_MED": 2.0,
                "workload_RMSE": 3.0,
                "workload_bias": 0.25,
                "workload_zero_violation_probability": 0.0,
                "MRED": 0.2,
                "ER": 0.3,
                "MED": 4.0,
                "RMSE": 5.0,
                "WCE": 16,
                "bias": 0.5,
            },
            "proxy": {},
            "task": {"score": score / 2.0},
        },
        "accepted_moves": 2,
        "hamming_from_reference": 3,
        "termination": "no_hard_improvement",
        "artifacts": {
            "best_json": artifact_rel,
            "best_rtl": rtl_rel,
        },
    }
    (run / "summary_v2.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return artifact, rtl


class SummarizeV2Test(unittest.TestCase):
    def test_relative_artifacts_are_resolved_from_each_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact, _ = write_run(
                root,
                "run_00_seed_123",
                feasible=True,
                violation=0.0,
                score=1.25,
                artifact_rel="nested/best_signed88_inits.json",
                rtl_rel="exports/best_rtl",
            )
            aggregate = summarize_v2.summarize(root)

            row = aggregate["best"]
            self.assertEqual(
                row["best_json"],
                "run_00_seed_123/nested/best_signed88_inits.json",
            )
            self.assertEqual(row["best_rtl"], "run_00_seed_123/exports/best_rtl")
            self.assertEqual(
                row["source_summary"], "run_00_seed_123/summary_v2.json"
            )
            copied = root / "overall_best_signed88_inits_v2.json"
            self.assertEqual(copied.read_bytes(), artifact.read_bytes())
            on_disk = json.loads((root / "summary_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk, aggregate)

    def test_sort_is_feasible_then_violation_then_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Score alone would choose the infeasible run; it must be last.
            write_run(root, "run_00_infeasible", feasible=False, violation=0.0, score=0.01)
            # Violation precedes score among equally feasible rows.
            write_run(root, "run_01_violation", feasible=True, violation=0.1, score=0.1)
            write_run(root, "run_02_score_high", feasible=True, violation=0.0, score=5.0)
            best_artifact, _ = write_run(
                root, "run_03_score_low", feasible=True, violation=0.0, score=2.0
            )

            aggregate = summarize_v2.summarize(root)
            self.assertEqual(
                [row["run"] for row in aggregate["rows"]],
                [
                    "run_03_score_low",
                    "run_02_score_high",
                    "run_01_violation",
                    "run_00_infeasible",
                ],
            )
            self.assertEqual(aggregate["best"]["run"], "run_03_score_low")
            self.assertEqual(
                (root / "overall_best_signed88_inits_v2.json").read_bytes(),
                best_artifact.read_bytes(),
            )

    def test_empty_directory_writes_empty_outputs_and_clears_stale_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "overall_best_signed88_inits_v2.json"
            stale.write_text("stale", encoding="utf-8")
            aggregate = summarize_v2.summarize(root)

            self.assertEqual(aggregate["rows"], [])
            self.assertIsNone(aggregate["best"])
            self.assertFalse(stale.exists())
            with (root / "summary_v2.csv").open(
                "r", newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            persisted = json.loads(
                (root / "summary_v2.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["rows"], [])
            self.assertIsNone(persisted["best"])


if __name__ == "__main__":
    unittest.main()
