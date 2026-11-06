import copy
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from utils.comparison import load_class_order, task_accuracies, forgetting
from utils.data_manager import DataManager
from models.base import BaseLearner
import trainer


class ComparisonTests(unittest.TestCase):
    def test_all_orders_are_permutations(self):
        for path in Path('class_orders').glob('*.yaml'):
            n = 100 if path.name.startswith('cifar') else 200
            orders = [load_class_order(path, i) for i in range(3)]
            for order in orders:
                self.assertEqual(sorted(order), list(range(n)))
            self.assertEqual(len({tuple(x) for x in orders}), 3)
        with self.assertRaises(ValueError):
            load_class_order('class_orders/cifar100_order.yaml', 3)

    def test_data_and_text_remapping_and_task_boundaries(self):
        data = SimpleNamespace(train_data=np.arange(100), train_targets=np.arange(100),
                               test_data=np.arange(100), test_targets=np.arange(100),
                               use_path=False, train_trsf=[], test_trsf=[], common_trsf=[],
                               class_order=list(range(100)), download_data=lambda: None)
        for index in range(3):
            order = load_class_order('class_orders/cifar100_order.yaml', index)
            with patch('utils.data_manager._get_idata', return_value=data), redirect_stdout(io.StringIO()):
                dm = DataManager('cifar224', True, 1993, 50, 10, class_order=order)
            self.assertEqual(dm._class_order, order)
            self.assertEqual(dm._increments, [50, 10, 10, 10, 10, 10])
            np.testing.assert_array_equal(dm._train_targets[order], np.arange(100))
            np.testing.assert_array_equal(dm._test_targets[order], np.arange(100))
            import json
            with open('utils/labels.json') as stream:
                names = json.load(stream)['cifar224']
            self.assertEqual(dm._class_to_label, [names[i] for i in order])
        with patch('utils.data_manager._get_idata', return_value=data), redirect_stdout(io.StringIO()):
            legacy = DataManager('cifar224', True, 1993, 10, 10)
            with self.assertRaises(ValueError):
                DataManager('cifar224', True, 1993, 10, 10, class_order=[0, 1])
        expected = np.random.RandomState(1993).permutation(100).tolist()
        self.assertEqual(legacy._class_order, expected)

    def test_three_runs_do_not_change_seed_or_config(self):
        args = {'seed': [1993], 'device': ['0'],
                'class_order_file': 'class_orders/cifar100_order.yaml'}
        original = copy.deepcopy(args)
        with patch('trainer._train') as run:
            trainer.train(args)
        self.assertEqual(args, original)
        self.assertEqual(run.call_count, 3)
        for i, call in enumerate(run.call_args_list):
            self.assertEqual(call.args[0]['seed'], 1993)
            self.assertEqual(call.args[0]['class_order'], load_class_order(args['class_order_file'], i))
            self.assertEqual(call.args[0]['device'], ['0'])

    def test_legacy_seed_loop(self):
        with patch('trainer._train') as run:
            trainer.train({'seed': [1, 2], 'device': ['0']})
        self.assertEqual([c.args[0]['seed'] for c in run.call_args_list], [1, 2])
        self.assertTrue(all(c.args[0]['class_order'] is None for c in run.call_args_list))

    def test_three_run_metric_files(self):
        import json
        import os
        import tempfile
        import logging
        import torch

        class FakeModel:
            _network = torch.nn.Linear(1, 1)

            def incremental_train(self, manager):
                pass

            def eval_task(self):
                return ({'top1': 80., 'top5': 100., 'grouped': {},
                         'acc_per_task': [80.]}, None, None, None, None, None)

            def after_task(self):
                pass

        args = json.loads(Path('exps/cifar_seca.json').read_text())
        args['class_order_file'] = str(Path(args['class_order_file']).resolve())
        manager = SimpleNamespace(nb_tasks=1, _class_order=list(range(100)))
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with patch('trainer.DataManager', return_value=manager), \
                     patch('trainer.factory.get_model', side_effect=lambda *a: FakeModel()), \
                     patch('trainer._set_device'), patch('trainer._set_random'), \
                     redirect_stdout(io.StringIO()):
                    trainer.train(args)
                files = list(Path('logs').rglob('*.metrics.jsonl'))
                self.assertEqual(len(files), 3)
                for file in files:
                    records = [json.loads(line) for line in file.read_text().splitlines()]
                    self.assertEqual(records[1]['fg'], 0)
                    self.assertEqual(records[1]['seca_fg'], 0)
                    self.assertIn('_order' + str(records[0]['class_order_idx']), file.name)
                self.assertEqual(len(list(Path('logs').rglob('*.log'))), 3)
                logging.basicConfig(force=True, handlers=[logging.NullHandler()])
        finally:
            os.chdir(previous)

    def test_fg_formulas(self):
        self.assertEqual(forgetting([[90]]), (0, 0))
        self.assertEqual(forgetting([[90], [80, 70]]), (10, 0))
        self.assertEqual(forgetting([[90], [95, 80], [85, 70, 60]]), (10, 2.5))
        # Positive backward transfer is allowed to yield negative forgetting.
        self.assertEqual(forgetting([[50], [60, 70]]), (-10, 0))

    def test_task_accuracy_uses_actual_boundaries_and_global_predictions(self):
        truth = np.array([0, 1, 2, 3, 4, 5])
        pred = np.array([[5], [1], [2], [0], [4], [0]])
        np.testing.assert_allclose(task_accuracies(pred, truth, [3, 2, 1]), [100 * 2 / 3, 50, 0])

    def test_eval_reuses_predictions_once(self):
        learner = BaseLearner({'memory_size': 0, 'device': ['cpu']})
        learner._cur_task = 1
        learner._known_classes, learner._total_classes = 2, 4
        learner.args = {'convnet_type': 'other'}
        learner.test_loader = object()
        learner.data_manager = SimpleNamespace(get_task_size=lambda i: 2)
        truth = np.array([0, 1, 2, 3])
        pred = np.array([[0]*5, [0]*5, [2]*5, [3]*5])
        with patch.object(learner, '_eval_cnn', return_value=(pred, truth)) as inference:
            result = learner.eval_task()[0]
        inference.assert_called_once_with(learner.test_loader)
        self.assertEqual(result['acc_per_task'], [50, 100])


if __name__ == '__main__':
    unittest.main()
