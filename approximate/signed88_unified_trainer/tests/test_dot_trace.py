import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from signed88.data import CalibrationProfile
from signed88.dot_trace import (
    DOT_TRACE_FORMAT,
    DotProxyLossConfig,
    DotTraceProfile,
    build_dot_group_record,
    compute_delta_y_numpy,
    compute_dot_proxy_loss,
    evaluate_dot_trace,
    load_dot_trace_jsonl,
    load_objective_profile,
    make_dot_trace_task_evaluator,
    to_torch_dot_trace,
    write_dot_trace_jsonl,
)


class DotTraceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.records = [
            {
                'type': 'group', 'id': 'train-0', 'layer': 'ffn.0',
                'channel': '7', 'split': 'train', 'scale': 0.5,
                'sensitivity': 2.0, 'normalizer': 2.0,
                'counts': [[0, 2], [65, 3]],
            },
            {
                'type': 'group', 'id': 'val-0', 'layer': 'ffn.0',
                'channel': '7', 'split': 'validation', 'scale': -0.25,
                'sensitivity': 4.0, 'normalizer': 1.0,
                'counts': [[65, 1], [4095, 4]],
            },
            {
                'type': 'group', 'id': 'val-1', 'layer': 'attn.1',
                'channel': '7', 'split': 'validation', 'scale': 0.125,
                'sensitivity': 1.0, 'normalizer': 0.5,
                'counts': [[0, 1], [4095, 2]],
            },
        ]
        self.path = self.root / 'trace.jsonl'
        write_dot_trace_jsonl(self.path, self.records)
        self.profile = load_dot_trace_jsonl(self.path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_load_select_and_hard_delta(self):
        self.assertEqual(self.profile.group_count, 3)
        self.assertEqual(self.profile.nnz, 6)
        self.assertEqual(self.profile.split_names, ('train', 'validation'))
        error = np.zeros(4096, dtype=np.float64)
        error[0], error[65], error[4095] = 1.0, -2.0, 3.0
        got = compute_delta_y_numpy(error, self.profile)
        expected = np.asarray([-2.0, -2.5, 0.875])
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)
        validation = self.profile.select_split('validation')
        np.testing.assert_allclose(
            compute_delta_y_numpy(error, validation), expected[1:], rtol=0, atol=0
        )
        self.assertEqual(validation.group_index.tolist(), [0, 0, 1, 1])

    def test_torch_delta_gradient_and_numpy_score_match(self):
        validation = self.profile.select_split('validation')
        batch = to_torch_dot_trace(validation, torch.device('cpu'), dtype=torch.float64)
        error = torch.zeros(4096, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            error[0], error[65], error[4095] = 1.0, -2.0, 3.0
        cfg = DotProxyLossConfig(
            huber_delta=0.75,
            channel_bias_weight=0.2,
            layer_bias_weight=0.3,
            tail_weight=0.4,
            tail_fraction=0.5,
            mse_weight=0.1,
        )
        total, terms = compute_dot_proxy_loss(error, batch, cfg)
        hard = evaluate_dot_trace(error.detach().numpy(), validation, cfg=cfg)
        self.assertAlmostEqual(float(total.detach()), hard.proxy_score, places=12)
        task = make_dot_trace_task_evaluator(
            self.profile, split='validation', cfg=cfg
        )(error.detach().numpy())
        self.assertAlmostEqual(task['score'], hard.proxy_score, places=12)
        self.assertIn('dot_channel_bias', terms)
        total.backward()
        self.assertIsNotNone(error.grad)
        self.assertNotEqual(float(error.grad[4095]), 0.0)
        self.assertEqual(float(error.grad[123]), 0.0)

    def test_record_builder_uses_signed_low_state(self):
        record = build_dot_group_record(
            group_id='g', a=[-1, 63, 64, -128], b=[-1, 1, 64, 127],
            scale=1.0, sensitivity=1.0, layer='l', channel='c', split='train',
        )
        expected_states = sorted([
            (63 * 64 + 63), (63 * 64 + 1), (0 * 64 + 0), (0 * 64 + 63)
        ])
        self.assertEqual([x[0] for x in record['counts']], expected_states)
        self.assertTrue(all(x[1] == 1 for x in record['counts']))

    def test_strict_count_validation(self):
        header = {
            'type': 'metadata', 'format': DOT_TRACE_FORMAT,
            'state_count': 4096, 'error_convention': 'approx_minus_exact',
        }
        base = dict(self.records[0])
        invalid = (
            [[4096, 1]],
            [[1, 1], [1, 2]],
            [[1, 1.5]],
            [[1, 0]],
        )
        for index, counts in enumerate(invalid):
            with self.subTest(counts=counts):
                path = self.root / f'invalid-{index}.jsonl'
                path.write_text(
                    json.dumps(header) + '\n' + json.dumps({**base, 'counts': counts}) + '\n',
                    encoding='utf-8',
                )
                with self.assertRaises(ValueError):
                    load_dot_trace_jsonl(path)

    def test_legacy_csv_auto_loader(self):
        csv_path = self.root / 'legacy.csv'
        csv_path.write_text('a,b,count\n-1,2,3\n1,2,1\n', encoding='utf-8')
        loaded = load_objective_profile(csv_path)
        self.assertIsInstance(loaded, CalibrationProfile)
        self.assertIsInstance(load_objective_profile(self.path), DotTraceProfile)


if __name__ == '__main__':
    unittest.main()
