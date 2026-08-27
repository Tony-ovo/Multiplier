import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from signed88.common import read_json
from signed88.hardware import get_design

ROOT = Path(__file__).resolve().parents[1]
SEGS = ('lo', 'mid', 'hi')
SUFS = ('lut01', 'lut23', 'lut45', 'lut67')


def tie(default_inits):
    return {f'{seg}_{suf}': default_inits[f'cp_{suf}'] for seg in SEGS for suf in SUFS}


class DefaultSplitTest(unittest.TestCase):
    def test_matches_default_when_tied(self):
        dft = get_design('default')
        spl = get_design('default_split')
        self.assertEqual(sum(len(v) for v in spl.spec.mutable_bits.values()), 168)
        self.assertTrue(np.array_equal(
            dft.hard_low_numpy(dft.spec.base_inits),
            spl.hard_low_numpy(spl.spec.base_inits)))
        rng = random.Random(11)
        for _ in range(3):
            rd = dft.random_inits(0.5, rng)
            self.assertTrue(np.array_equal(
                dft.hard_low_numpy(rd), spl.hard_low_numpy(tie(rd))))

    def test_decoupling_hi_changes_only_when_hi_digit_nonzero(self):
        dft = get_design('default')
        spl = get_design('default_split')
        rng = random.Random(12)
        shared = tie(dft.random_inits(0.5, rng))
        other = dft.random_inits(0.5, rng)
        decoupled = dict(shared)
        for suf in SUFS:
            decoupled[f'hi_{suf}'] = other[f'cp_{suf}']
        low_shared = spl.hard_low_numpy(shared).astype(np.int64)
        low_dec = spl.hard_low_numpy(decoupled).astype(np.int64)
        self.assertFalse(np.array_equal(low_shared, low_dec))
        bl = np.tile(np.arange(64), 64)
        hi_zero = ((bl >> 4) & 3) == 0
        self.assertTrue(np.array_equal(low_shared[hi_zero], low_dec[hi_zero]))

    def test_hard_forward_matches_numpy(self):
        spl = get_design('default_split')
        model = spl.build_model(spl.spec.base_inits, 0.999, 0.0).cpu()
        with torch.no_grad():
            value, _ = model.forward_low_grid(c_init=1.0, c_out=1.0, hard_middle=True)
        got = value.numpy().round().astype(np.int32)
        self.assertTrue(np.array_equal(got, spl.hard_low_numpy(spl.spec.base_inits)))

    def test_rtl_export_and_distinct_patch(self):
        spl = get_design('default_split')
        rng = random.Random(13)
        inits = spl.random_inits(0.5, rng)
        with tempfile.TemporaryDirectory() as td:
            out = spl.export_rtl(ROOT / 'rtl_sources', Path(td) / 'dsplit', inits)
            obj = read_json(out / 'trained_artifact.json')
            self.assertEqual(obj['design'], 'default_split')
            self.assertEqual(spl.normalize_inits(obj['inits']), spl.normalize_inits(inits))
            text = (out / 's8862_approx62_cp_dsplit.v').read_text()
            for seg in SEGS:
                for suf in SUFS:
                    init_hex = spl.normalize_inits(inits)[f'{seg}_{suf}'][4:]
                    self.assertIn(init_hex.upper(), text.upper())


if __name__ == '__main__':
    unittest.main()
