import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_UNIFIED = _HERE.parent / 'signed88_unified_trainer'
for p in (str(_HERE), str(_UNIFIED)):
    if p not in sys.path:
        sys.path.insert(0, p)

from signed88.hardware import get_design  # noqa: E402

from quad_lut import QUAD_LEVELS, QuadLUT6_2  # noqa: E402
from quad_model import build_quad_model, collapse_regularizer, quad_usage_stats  # noqa: E402


class QuadLutTest(unittest.TestCase):
    def test_baseline_init_reproduces_binary_table(self):
        init = "64'hE62A4C006A40EAC0"
        lut = QuadLUT6_2(init, list(range(64)), 0.999, 0.0)
        self.assertEqual(lut.hard_hex().upper(), init.upper())
        # Soft expected values must sit within init_conf of the extremes.
        v = lut.table(1.0)
        bits = lut.fixed_bits
        self.assertTrue(torch.all(torch.abs(v - bits) < 0.01))

    def test_distance_weighted_gradients(self):
        # Entry committed near 00: the pull on the 11 logit must exceed the
        # pull on the 01 logit by roughly the level-distance ratio (3x).
        lut = QuadLUT6_2("64'h0000000000000000", [0], 0.9, 0.0)
        v = lut.table(1.0)[0]
        v.backward()
        g = lut.logits.grad[0]
        # dv/dlogit_k = tau * p_k * (level_k - v); with v near 0 all pulls are
        # positive toward higher levels and scale with distance.
        self.assertGreater(float(g[3]), float(g[1]) * 2.5)
        self.assertGreater(float(g[2]), float(g[1]) * 1.5)
        self.assertLess(float(g[0]), 0.0)  # mass conservation pulls 00 down

    def test_quad_ste_matches_binary_collapse_forward(self):
        # Hard-STE circuit output must equal the deployed binary hard model.
        design = get_design('balanced_split')
        torch.manual_seed(3)
        model = build_quad_model(design, design.spec.base_inits, 0.9, 0.0, 3.0)
        for table in model.core.tables.values():
            table.randomize_mutable_logits(std=1.5)
        with torch.no_grad():
            value, _ = model.forward_low_grid(c_init=2.0, c_out=1.0, hard_middle=True)
        got = value.cpu().numpy().round().astype(np.int32)
        expected = design.hard_low_numpy(design.normalize_inits(model.hard_inits()))
        self.assertTrue(np.array_equal(got, expected))

    def test_intermediate_levels_are_reachable_and_counted(self):
        design = get_design('balanced_split')
        torch.manual_seed(5)
        model = build_quad_model(design, design.spec.base_inits, 0.9, 0.0, 3.0)
        for table in model.core.tables.values():
            table.randomize_mutable_logits(std=2.0)
        usage = quad_usage_stats(model)
        self.assertEqual(usage['mutable_entries'], 112)
        # Random 4-way logits should place a nontrivial share on 01/10.
        self.assertGreater(usage['intermediate_fraction'], 0.2)

    def test_default_split_quad_ste_and_entry_count(self):
        design = get_design('default_split')
        torch.manual_seed(3)
        model = build_quad_model(design, design.spec.base_inits, 0.9, 0.0, 3.0)
        usage = quad_usage_stats(model)
        self.assertEqual(usage['mutable_entries'], 168)
        for table in model.core.tables.values():
            table.randomize_mutable_logits(std=1.5)
        with torch.no_grad():
            value, _ = model.forward_low_grid(c_init=2.0, c_out=1.0, hard_middle=True)
        got = value.cpu().numpy().round().astype(np.int32)
        expected = design.hard_low_numpy(design.normalize_inits(model.hard_inits()))
        self.assertTrue(np.array_equal(got, expected))

    def test_collapse_regularizer_zero_when_committed(self):
        design = get_design('balanced_split')
        model = build_quad_model(design, design.spec.base_inits, 0.9999, 0.0, 3.0)
        reg = collapse_regularizer(model, 2.0)
        self.assertLess(float(reg), 1e-4)

    def test_frozen_bits_survive_randomization(self):
        init = "64'hE62A4C006A40EAC0"
        mutable = [i for i in range(64) if (i & 3) == 3]
        lut = QuadLUT6_2(init, mutable, 0.999, 0.0)
        lut.randomize_mutable_logits(std=3.0)
        hard = int(lut.hard_hex()[4:], 16)
        base = int(init[4:], 16)
        frozen_mask = 0
        for i in range(64):
            if i not in mutable:
                frozen_mask |= 1 << i
        self.assertEqual(hard & frozen_mask, base & frozen_mask)

    def test_levels_embedding(self):
        self.assertEqual(QUAD_LEVELS, (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))


if __name__ == '__main__':
    unittest.main()
