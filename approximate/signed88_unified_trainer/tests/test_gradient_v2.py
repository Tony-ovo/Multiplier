import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

if not (Path(__file__).resolve().parents[1] / 'signed88' / 'gradient_v2.py').exists():
    raise unittest.SkipTest('V2 was intentionally removed; testing the V1 improvement branch')

from signed88.common import hex_to_int, int_to_hex
from signed88.data import load_calibration_csv, to_torch
from signed88.gradient_v2 import (
    GradientBalanceConfig,
    GradientNormBalancer,
    pcgrad_backward,
)
from signed88.hardware import choices, get_design
from signed88.losses_v2 import (
    ERSurrogateConfig,
    HistogramLossConfig,
    compute_histogram_components,
    er_surrogate,
)
from signed88.lut import TrainableLUT6
from signed88.stage3_v2 import (
    GradientGuidedStage3Config,
    HardBestTracker,
    rank_hard_bit_flips,
    run_gradient_guided_stage3,
)


ROOT = Path(__file__).resolve().parents[1]


class LossV2Test(unittest.TestCase):
    def test_default_er_has_gradient_where_old_tau_point_one_saturates(self):
        old_error = torch.tensor(4.0, requires_grad=True)
        old = 1.0 - torch.exp(-torch.abs(old_error) / 0.1)
        old.backward()
        self.assertLess(abs(float(old_error.grad)), 1e-15)

        new_error = torch.tensor(4.0, requires_grad=True)
        new = er_surrogate(new_error, temperature=0.1)
        new.backward()
        self.assertGreater(float(new_error.grad), 1e-2)
        self.assertGreater(float(new), 0.0)
        self.assertLess(float(new), 1.0)

    def test_histogram_components_are_independent_and_differentiable(self):
        profile = load_calibration_csv(ROOT / "data/w8a8_calibration_hist_smoke_pcalib_nonzero.csv")
        batch = to_torch(profile, torch.device("cpu"))
        design = get_design("balanced")
        model = design.build_model(design.spec.base_inits, 0.60, 0.0)
        components = compute_histogram_components(
            model,
            batch,
            hard_middle=True,
            er_temperature=0.1,
            cfg=HistogramLossConfig(),
        )
        primary = (
            "nmae",
            "nrmse",
            "clipped_mred",
            "er",
            "squared_bias",
            "bit_exact",
        )
        for name in primary:
            self.assertIn(name, components)
            self.assertTrue(bool(torch.isfinite(components[name])))
            gradients = torch.autograd.grad(
                components[name], tuple(model.parameters()), retain_graph=True, allow_unused=True
            )
            norm = sum(float(torch.sum(g * g)) for g in gradients if g is not None) ** 0.5
            self.assertGreater(norm, 0.0, name)


class GradientCompositionTest(unittest.TestCase):
    def test_inverse_norm_balance_equalizes_contributions(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        components = {"small": parameter * 1e-6, "large": parameter * 1e3}
        balancer = GradientNormBalancer(
            GradientBalanceConfig(
                ema_decay=0.0,
                min_multiplier=1e-12,
                max_multiplier=1e12,
                relative_gradient_floor=0.0,
            )
        )
        result = balancer.combine(components, {"small": 1.0, "large": 1.0}, [parameter])
        a = result.estimated_contribution_norms["small"]
        b = result.estimated_contribution_norms["large"]
        self.assertAlmostEqual(a / b, 1.0, places=5)
        result.loss.backward()
        self.assertTrue(bool(torch.isfinite(parameter.grad)))

        restored = GradientNormBalancer()
        restored.load_state_dict(balancer.state_dict())
        self.assertEqual(restored.steps, 1)
        self.assertEqual(restored.ema_norms, balancer.ema_norms)

    def test_pcgrad_projects_conflicting_tasks(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        components = {
            "left": parameter[0],
            "right": -parameter[0] + parameter[1],
        }
        result = pcgrad_backward(components, {"left": 1.0, "right": 1.0}, [parameter])
        self.assertGreater(result.conflict_count, 0)
        self.assertTrue(bool(torch.all(torch.isfinite(parameter.grad))))

    def test_locally_constant_task_does_not_steal_gradient_scale(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        components = {"active": parameter, "constant": parameter * 0.0}
        result = GradientNormBalancer().combine(
            components, {"active": 1.0, "constant": 1.0}, [parameter]
        )
        self.assertAlmostEqual(result.effective_weights["active"], 1.0)
        self.assertEqual(result.effective_weights["constant"], 0.0)

    def test_relative_near_zero_task_does_not_steal_gradient_scale(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        components = {"active": parameter, "near_zero": parameter * 1e-9}
        result = GradientNormBalancer(
            GradientBalanceConfig(relative_gradient_floor=1e-6)
        ).combine(components, {"active": 1.0, "near_zero": 1.0}, [parameter])
        self.assertAlmostEqual(result.effective_weights["active"], 1.0)
        self.assertEqual(result.effective_weights["near_zero"], 0.0)


class _ToyModel(torch.nn.Module):
    def __init__(self, init_hex, init_conf, noise_std):
        super().__init__()
        self.tables = torch.nn.ModuleDict(
            {"lut": TrainableLUT6(init_hex, (3,), init_conf, noise_std)}
        )

    def hard_inits(self):
        return {"lut": self.tables["lut"].hard_hex()}


class _ToyDesign:
    def __init__(self):
        self.spec = SimpleNamespace(
            name="toy", train_names=("lut",), search_bits={"lut": (3,)}
        )

    def normalize_inits(self, inits):
        value = hex_to_int(inits["lut"])
        if value & ~(1 << 3):
            raise ValueError("frozen toy bit changed")
        return {"lut": int_to_hex(value)}

    def build_model(self, inits, init_conf, noise_std):
        return _ToyModel(self.normalize_inits(inits)["lut"], init_conf, noise_std)

    def artifact(self, inits, metrics=None, extra=None):
        obj = {"design": "toy", "inits": self.normalize_inits(inits)}
        if metrics is not None:
            obj["metrics"] = metrics
        if extra:
            obj.update(extra)
        return obj


class Stage3V2Test(unittest.TestCase):
    def test_all_registered_designs_expose_search_bit_gradients(self):
        for name in choices():
            design = get_design(name)
            model = design.build_model(design.spec.base_inits, 0.60, 0.0)
            sum(parameter.sum() for parameter in model.parameters()).backward()
            ranked = rank_hard_bit_flips(model, design)
            expected = sum(len(design.spec.search_bits[table]) for table in design.spec.train_names)
            self.assertEqual(len(ranked), expected, name)

    def test_gradient_guided_stage3_restores_and_accepts_exact_hard_bit(self):
        design = _ToyDesign()

        def evaluate(inits):
            bit = (hex_to_int(inits["lut"]) >> 3) & 1
            return {"objective_score": float(1 - bit), "bit": bit}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "hard_best.json"
            start = {"lut": int_to_hex(0)}
            tracker = HardBestTracker(
                design, start, evaluate(start), checkpoint_path=checkpoint
            )

            balancer = GradientNormBalancer()

            def loss_closure(model):
                probability = torch.sigmoid(model.tables["lut"].logits[3])
                components = {"target": torch.square(probability - 1.0)}
                return balancer.combine(components, {"target": 1.0}, model.parameters())

            history = run_gradient_guided_stage3(
                design,
                tracker,
                loss_closure,
                evaluate,
                device=torch.device("cpu"),
                cfg=GradientGuidedStage3Config(rounds=3, top_k=1, pair_top_k=0),
            )
            self.assertEqual(tracker.score, 0.0)
            self.assertTrue(history[0]["accepted"])
            self.assertEqual((hex_to_int(tracker.inits["lut"]) >> 3) & 1, 1)
            self.assertTrue(checkpoint.exists())

            loaded = HardBestTracker.load(design, checkpoint)
            restored = loaded.restore_model(device=torch.device("cpu"))
            self.assertEqual(restored.hard_inits(), tracker.inits)


if __name__ == "__main__":
    unittest.main()
