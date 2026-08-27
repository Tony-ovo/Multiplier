import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from signed88.common import ObjectiveWeights, MAX_ABS_PRODUCT, read_json
from signed88.data import load_calibration_csv
from signed88.hardware import get_design
from signed88.metrics import evaluate_design

ROOT = Path(__file__).resolve().parents[1]
SEGS = ('lo', 'mid')
SUFS = ('lut01', 'lut23', 'lut45', 'lut67')


def tie(balanced_inits):
    return {f'{seg}_{suf}': balanced_inits[f'cp_{suf}'] for seg in SEGS for suf in SUFS}


class BalancedSplitTest(unittest.TestCase):
    def test_matches_balanced_when_tied(self):
        bal = get_design('balanced')
        spl = get_design('balanced_split')
        self.assertTrue(np.array_equal(
            bal.hard_low_numpy(bal.spec.base_inits),
            spl.hard_low_numpy(spl.spec.base_inits)))
        rng = random.Random(11)
        for _ in range(3):
            rb = bal.random_inits(0.5, rng)
            self.assertTrue(np.array_equal(
                bal.hard_low_numpy(rb), spl.hard_low_numpy(tie(rb))))

    def test_decoupling_changes_mid_segment_only(self):
        bal = get_design('balanced')
        spl = get_design('balanced_split')
        rng = random.Random(12)
        shared = tie(bal.random_inits(0.5, rng))
        other = bal.random_inits(0.5, rng)
        decoupled = dict(shared)
        for suf in SUFS:
            decoupled[f'mid_{suf}'] = other[f'cp_{suf}']
        low_shared = spl.hard_low_numpy(shared).astype(np.int64)
        low_dec = spl.hard_low_numpy(decoupled).astype(np.int64)
        self.assertFalse(np.array_equal(low_shared, low_dec))
        # Changing mid tables must never touch states where b[3:2]==0 digit
        # contribution is exact... the lo segment output for bl in {0,1,2,3}
        # with mid digit 0 must be identical.
        bl = np.tile(np.arange(64), 64)
        mid_zero = ((bl >> 2) & 3) == 0
        self.assertTrue(np.array_equal(low_shared[mid_zero], low_dec[mid_zero]))

    def test_hard_forward_matches_numpy(self):
        spl = get_design('balanced_split')
        model = spl.build_model(spl.spec.base_inits, 0.999, 0.0).cpu()
        with torch.no_grad():
            value, _ = model.forward_low_grid(c_init=1.0, c_out=1.0, hard_middle=True)
        got = value.numpy().round().astype(np.int32)
        self.assertTrue(np.array_equal(got, spl.hard_low_numpy(spl.spec.base_inits)))

    def test_rtl_export_and_distinct_patch(self):
        spl = get_design('balanced_split')
        rng = random.Random(13)
        inits = spl.random_inits(0.5, rng)
        with tempfile.TemporaryDirectory() as td:
            out = spl.export_rtl(ROOT / 'rtl_sources', Path(td) / 'split', inits)
            obj = read_json(out / 'trained_artifact.json')
            self.assertEqual(obj['design'], 'balanced_split')
            self.assertEqual(spl.normalize_inits(obj['inits']), spl.normalize_inits(inits))
            text = (out / 's8862_approx62_cp_split.v').read_text()
            for seg in SEGS:
                for suf in SUFS:
                    init_hex = spl.normalize_inits(inits)[f'{seg}_{suf}'][4:]
                    self.assertIn(init_hex.upper(), text.upper())

    def test_wce_objective_term(self):
        spl = get_design('balanced_split')
        profile = load_calibration_csv(
            ROOT / 'data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv', 'auto')
        base = evaluate_design(spl, spl.spec.base_inits, profile, ObjectiveWeights())
        weighted = evaluate_design(
            spl, spl.spec.base_inits, profile, ObjectiveWeights(uniform_wce=2.0))
        self.assertEqual(base.WCE, weighted.WCE)
        expect = base.objective_score + 2.0 * base.WCE / MAX_ABS_PRODUCT
        self.assertAlmostEqual(weighted.objective_score, expect, places=12)


if __name__ == '__main__':
    unittest.main()
