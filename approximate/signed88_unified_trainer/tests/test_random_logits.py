from __future__ import annotations

import math
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import train
from signed88.common import hex_to_int, read_json, set_seed
from signed88.hardware import get_design
from signed88.lut import TrainableLUT6


ROOT = Path(__file__).resolve().parents[1]


def _lut_modules(model):
    return [module for module in model.modules() if isinstance(module, TrainableLUT6)]


def _mutable_logits(model) -> torch.Tensor:
    values = [
        module.logits.detach().cpu()[module.mutable_mask.detach().cpu()]
        for module in _lut_modules(model)
    ]
    return torch.cat(values)


class RandomizeMutableLogitsUnitTest(unittest.TestCase):
    def _new_lut(self) -> TrainableLUT6:
        return TrainableLUT6(
            "64'hA55A0123456789EF",
            mutable_bits=tuple(range(64)),
            init_conf=0.80,
            noise_std=0.0,
        )

    def test_same_seed_is_exactly_reproducible_and_different_seed_differs(self):
        first = self._new_lut()
        second = self._new_lut()
        third = self._new_lut()

        set_seed(12345)
        first.randomize_mutable_logits(mean=0.25, std=0.75)
        set_seed(12345)
        second.randomize_mutable_logits(mean=0.25, std=0.75)
        set_seed(12346)
        third.randomize_mutable_logits(mean=0.25, std=0.75)

        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertFalse(torch.equal(first.logits, third.logits))

    def test_mutable_values_are_a_real_distribution_not_fixed_confidence(self):
        lut = self._new_lut()
        set_seed(20260819)
        lut.randomize_mutable_logits(mean=0.25, std=0.75)
        values = lut.logits.detach().cpu().numpy()

        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(np.unique(values).size, 48)
        self.assertGreater(float(values.max()), 0.25)
        self.assertLess(float(values.min()), 0.25)
        # Loose statistical bounds avoid testing a particular RNG algorithm,
        # while still catching constant, Bernoulli, or wrongly scaled logits.
        self.assertLess(abs(float(values.mean()) - 0.25), 0.35)
        self.assertGreater(float(values.std()), 0.40)
        self.assertLess(float(values.std()), 1.10)

    def test_only_mutable_logits_change_and_frozen_hard_bits_stay_exact(self):
        init_hex = "64'hA55A0123456789EF"
        mutable = (0, 3, 7, 19, 42, 63)
        lut = TrainableLUT6(init_hex, mutable, init_conf=0.80, noise_std=0.0)
        before_logits = lut.logits.detach().clone()
        before_hard = lut.hard_bits().detach().clone()

        set_seed(9)
        lut.randomize_mutable_logits(mean=0.0, std=1.0)

        frozen = ~lut.mutable_mask
        self.assertTrue(torch.equal(lut.logits.detach()[frozen], before_logits[frozen]))
        self.assertTrue(torch.equal(lut.hard_bits().detach()[frozen], before_hard[frozen]))
        self.assertFalse(
            torch.equal(lut.logits.detach()[lut.mutable_mask], before_logits[lut.mutable_mask])
        )

    def test_invalid_distribution_parameters_are_rejected(self):
        invalid = (
            (math.nan, 1.0),
            (math.inf, 1.0),
            (-math.inf, 1.0),
            (0.0, 0.0),
            (0.0, -1.0),
            (0.0, math.nan),
            (0.0, math.inf),
        )
        for mean, std in invalid:
            with self.subTest(mean=mean, std=std):
                with self.assertRaises(ValueError):
                    self._new_lut().randomize_mutable_logits(mean=mean, std=std)


class RandomLogitsZeroEpochCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls._temporary.name)
        cls.same_a = cls._run_zero_epoch('same_a', seed=314159)
        cls.same_b = cls._run_zero_epoch('same_b', seed=314159)
        cls.different = cls._run_zero_epoch('different', seed=314160)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @classmethod
    def _run_zero_epoch(cls, label: str, seed: int):
        out_dir = cls.temp_root / label
        design = get_design('default')
        original_build_model = design.build_model
        captured = []

        def capture_model(*args, **kwargs):
            model = original_build_model(*args, **kwargs)
            captured.append(model)
            return model

        argv = [
            'train.py',
            '--design', 'default',
            '--device', 'cpu',
            '--seed', str(seed),
            '--init-mode', 'random_logits',
            '--random-logit-mean', '0.0',
            '--random-logit-std', '1.0',
            '--stage1-epochs', '0',
            '--stage2-epochs', '0',
            '--stage3-epochs', '0',
            '--population-size', '0',
            '--population-epochs', '0',
            '--out-dir', str(out_dir),
            '--rtl-template-root', str(ROOT / 'rtl_sources'),
        ]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(
            design, 'build_model', side_effect=capture_model
        ), mock.patch.object(sys, 'stdout', io.StringIO()), mock.patch.object(
            sys, 'stderr', io.StringIO()
        ):
            rc = train.main()

        if rc != 0:
            raise AssertionError(f'train.main returned {rc}')
        if len(captured) != 1:
            raise AssertionError(f'expected one model, captured {len(captured)}')
        return {
            'model': captured[0],
            'initial': read_json(out_dir / 'initial_signed88_inits.json'),
            'best': read_json(out_dir / 'best_signed88_inits.json'),
            'rtl': read_json(out_dir / 'best_rtl' / 'trained_artifact.json'),
            'summary': read_json(out_dir / 'summary.json'),
        }

    def test_default_has_all_56_trainable_logits_randomized(self):
        model = self.same_a['model']
        values = _mutable_logits(model)
        self.assertEqual(values.numel(), 56)
        self.assertGreater(torch.unique(values).numel(), 48)
        self.assertTrue(bool(torch.isfinite(values).all()))
        self.assertTrue(bool(torch.any(values < 0.0)))
        self.assertTrue(bool(torch.any(values > 0.0)))

    def test_cli_same_seed_reproduces_logits_and_hard_init(self):
        self.assertTrue(
            torch.equal(
                _mutable_logits(self.same_a['model']),
                _mutable_logits(self.same_b['model']),
            )
        )
        self.assertEqual(self.same_a['initial']['inits'], self.same_b['initial']['inits'])
        self.assertEqual(self.same_a['initial']['metrics'], self.same_b['initial']['metrics'])

    def test_cli_different_seed_changes_logits_and_hard_init(self):
        self.assertFalse(
            torch.equal(
                _mutable_logits(self.same_a['model']),
                _mutable_logits(self.different['model']),
            )
        )
        self.assertNotEqual(self.same_a['initial']['inits'], self.different['initial']['inits'])

    def test_frozen_bits_remain_the_default_baseline(self):
        design = get_design('default')
        hard = self.same_a['model'].hard_inits()
        for name in design.spec.train_names:
            got = hex_to_int(hard[name])
            base = hex_to_int(design.spec.base_inits[name])
            mutable = set(design.spec.mutable_bits[name])
            for bit in range(64):
                if bit not in mutable:
                    self.assertEqual(
                        (got >> bit) & 1,
                        (base >> bit) & 1,
                        f'{name}[{bit}] changed although it is frozen',
                    )

    def test_zero_epoch_initial_best_rtl_and_summary_are_consistent(self):
        run = self.same_a
        model_hard = run['model'].hard_inits()
        self.assertEqual(run['initial']['inits'], model_hard)
        self.assertEqual(run['best']['inits'], model_hard)
        self.assertEqual(run['rtl']['inits'], model_hard)
        self.assertEqual(run['summary']['initial_metrics'], run['initial']['metrics'])
        self.assertEqual(run['summary']['best_metrics'], run['best']['metrics'])
        self.assertEqual(run['summary']['best_stage'], 'initial')
        self.assertEqual(run['summary']['train_args']['init_mode'], 'random_logits')
        self.assertEqual(run['summary']['train_args']['random_logit_mean'], 0.0)
        self.assertEqual(run['summary']['train_args']['random_logit_std'], 1.0)

    def test_hard_model_behavior_matches_exported_init_numpy_model(self):
        design = get_design('default')
        model = self.same_a['model'].cpu()
        with torch.no_grad():
            value, _ = model.forward_low_grid(
                c_init=1.0,
                c_out=1.0,
                hard_middle=True,
            )
        got = value.cpu().numpy().round().astype(np.int32)
        expected = design.hard_low_numpy(self.same_a['best']['inits'])
        self.assertTrue(np.array_equal(got, expected))


if __name__ == '__main__':
    unittest.main()
