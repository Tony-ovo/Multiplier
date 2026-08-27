import unittest

import torch

from signed88.common import hex_to_int
from signed88.hard_search import (
    MetricConstraint,
    ParetoMetric,
    gradient_ranked_hard_search,
    make_constraint_filter,
    make_pareto_better,
    rank_bit_gradients,
    rank_model_gradients,
)
from signed88.hardware import get_design


DESIGNS = ("aggressive", "fast", "default", "balanced", "quality", "area")


def all_search_bits(design):
    return [
        (name, int(bit))
        for name in design.spec.train_names
        for bit in design.spec.search_bits[name]
    ]


def changed_bits(design, base, candidate):
    changed = set()
    for name, bit in all_search_bits(design):
        if ((hex_to_int(base[name]) ^ hex_to_int(candidate[name])) >> bit) & 1:
            changed.add((name, bit))
    return changed


def terminal_modules(model, design):
    result = {}
    for path, module in model.named_modules():
        terminal = path.rsplit(".", 1)[-1]
        if terminal in design.spec.train_names and hasattr(module, "logits"):
            result[terminal] = module
    return result


class GradientExtractionTest(unittest.TestCase):
    def test_terminal_name_mapping_covers_every_design(self):
        for design_name in DESIGNS:
            with self.subTest(design=design_name):
                design = get_design(design_name)
                model = design.build_model(design.spec.base_inits, 0.75, 0.0)
                modules = terminal_modules(model, design)
                self.assertEqual(set(modules), set(design.spec.train_names))
                for module in modules.values():
                    module.logits.grad = torch.zeros_like(module.logits)

                target_name, target_bit = all_search_bits(design)[0]
                hard = (hex_to_int(design.spec.base_inits[target_name]) >> target_bit) & 1
                # A positive gradient gain is -grad * (+1 for 0->1 / -1 for 1->0).
                modules[target_name].logits.grad[target_bit] = -1.0 if hard == 0 else 1.0
                ranking = rank_model_gradients(model, design, normalization="logit")

                self.assertEqual(len(ranking), len(all_search_bits(design)))
                self.assertEqual(ranking[0].ref, (target_name, target_bit))
                self.assertAlmostEqual(ranking[0].predicted_gain, 1.0)

    def test_table_normalization_recovers_discrete_bit_gradient(self):
        design = get_design("fast")
        model = design.build_model(design.spec.base_inits, 0.80, 0.0)
        modules = terminal_modules(model, design)
        for module in modules.values():
            module.logits.grad = torch.zeros_like(module.logits)

        name, bit = all_search_bits(design)[0]
        module = modules[name]
        with torch.enable_grad():
            table = module.table(1.7)
            sensitivity = torch.autograd.grad(table.sum(), module.logits)[0]
        desired_bit_gradient = -3.25
        module.logits.grad[bit] = sensitivity[bit] * desired_bit_gradient

        ranking = rank_model_gradients(model, design, normalization="table", c_init=1.7)
        item = next(row for row in ranking if row.ref == (name, bit))
        self.assertAlmostEqual(item.bit_gradient, desired_bit_gradient, places=5)


class ExactHardSearchTest(unittest.TestCase):
    def setUp(self):
        self.design = get_design("fast")
        self.base = self.design.normalize_inits(self.design.spec.base_inits)
        self.bits = all_search_bits(self.design)

    def ranking_for(self, gains):
        gradients = {name: [0.0] * 64 for name in self.design.spec.train_names}
        for ref, gain in gains.items():
            name, bit = ref
            hard = (hex_to_int(self.base[name]) >> bit) & 1
            direction = 1 if hard == 0 else -1
            gradients[name][bit] = -float(gain) / direction
        return rank_bit_gradients(self.design, self.base, gradients)

    def test_pair_synergy_is_exactly_evaluated(self):
        first, second = self.bits[:2]
        ranking = self.ranking_for({first: 4.0, second: 3.0})

        def evaluate(inits):
            changed = changed_bits(self.design, self.base, inits)
            if changed == {first, second}:
                return {"score": 4.0}
            if changed in ({first}, {second}):
                return {"score": 11.0}
            return {"score": 10.0}

        result = gradient_ranked_hard_search(
            self.design,
            self.base,
            ranking,
            evaluate=evaluate,
            better=lambda new, old: new["score"] < old["score"],
            top_k=2,
            pair_top_k=2,
        )

        self.assertIsNotNone(result.accepted)
        self.assertEqual(result.accepted.kind, "pair")
        self.assertEqual(set(result.accepted.flips), {first, second})
        self.assertEqual(result.accepted.evaluation["score"], 4.0)
        self.assertEqual(result.evaluations, 4)  # base + two singles + one pair
        self.assertEqual(self.base, self.design.normalize_inits(self.design.spec.base_inits))

    def test_constraints_and_pareto_acceptance(self):
        violating, dominating, tradeoff = self.bits[:3]
        ranking = self.ranking_for({violating: 5.0, dominating: 4.0, tradeoff: 3.0})

        rows = {
            frozenset(): {"score": 10.0, "validation": {"mred": 10.0, "er": 10.0}},
            frozenset((violating,)): {"score": 5.0, "validation": {"mred": 8.0, "er": 12.0}},
            frozenset((dominating,)): {"score": 7.0, "validation": {"mred": 8.0, "er": 9.0}},
            frozenset((tradeoff,)): {"score": 6.0, "validation": {"mred": 11.0, "er": 7.0}},
        }

        def evaluate(inits):
            return rows[frozenset(changed_bits(self.design, self.base, inits))]

        feasible = make_constraint_filter(
            (MetricConstraint("validation.er", max_increase=0.0),)
        )
        dominates = make_pareto_better(
            (
                ParetoMetric("validation.mred"),
                ParetoMetric("validation.er"),
            )
        )

        scalar_result = gradient_ranked_hard_search(
            self.design,
            self.base,
            ranking,
            evaluate=evaluate,
            better=lambda new, old: new["score"] < old["score"],
            feasible=feasible,
            pareto_dominates=dominates,
            top_k=3,
        )
        self.assertEqual(scalar_result.accepted.flips, (tradeoff,))
        self.assertFalse(scalar_result.candidates[0].feasible)
        self.assertEqual(
            {candidate.flips for candidate in scalar_result.pareto_front},
            {(dominating,), (tradeoff,)},
        )

        pareto_result = gradient_ranked_hard_search(
            self.design,
            self.base,
            ranking,
            evaluate=evaluate,
            better=dominates,
            feasible=feasible,
            top_k=3,
        )
        self.assertEqual(pareto_result.accepted.flips, (dominating,))


if __name__ == "__main__":
    unittest.main()
