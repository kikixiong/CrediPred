"""Leakage-contract tests for the deterministic clean-UQ baselines."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from credipred.experiments.gnn_experiments.clean_uq_baselines import (
    fit_global_median,
    fit_xgboost_with_selection,
    regression_label_propagation,
    select_label_propagation,
    write_baseline_role_predictions,
)


class _RecordingEstimator:
    def __init__(self, kind: str, config: dict[str, Any], log: list[tuple[str, str, np.ndarray]]) -> None:
        self.kind = kind
        self.config = config
        self.log = log

    def fit(self, features: np.ndarray, targets: np.ndarray) -> '_RecordingEstimator':
        self.log.append((self.kind, 'fit', features.copy()))
        self._level = float(np.median(targets))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.log.append((self.kind, 'predict', features.copy()))
        midpoint = np.full(features.shape[0], self._level, dtype=np.float32)
        if self.kind == 'quantile':
            return np.stack((midpoint - 0.1, midpoint + 0.1), axis=1)
        return midpoint


class _GridEstimator(_RecordingEstimator):
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.log.append((self.kind, 'predict', features.copy()))
        depth = self.config['max_depth']
        if self.kind == 'midpoint':
            value = 0.5 if depth == 2 else 0.0
            return np.full(features.shape[0], value, dtype=np.float32)
        spread = 0.2 if depth == 2 else 0.05
        midpoint = np.full(features.shape[0], 0.5, dtype=np.float32)
        return np.stack((midpoint - spread, midpoint + spread), axis=1)


class CleanUqBaselinesTest(unittest.TestCase):
    def test_global_median_uses_fit_targets_only(self) -> None:
        fit = torch.tensor([0, 2, 4])
        labels = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.8, 0.7])
        changed_nonfit = labels.clone()
        changed_nonfit[torch.tensor([1, 3, 5])] = torch.tensor([0.0, 1.0, 0.0])

        self.assertEqual(fit_global_median(labels, fit), 0.5)
        self.assertEqual(fit_global_median(changed_nonfit, fit), 0.5)
        self.assertNotIn('select', inspect.signature(fit_global_median).parameters)
        self.assertNotIn('calibrate', inspect.signature(fit_global_median).parameters)
        self.assertNotIn('test', inspect.signature(fit_global_median).parameters)

    def test_xgboost_fits_only_fit_and_scores_only_select(self) -> None:
        features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        labels = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])
        fit = torch.tensor([0, 2, 4])
        select = torch.tensor([1, 3])
        log: list[tuple[str, str, np.ndarray]] = []

        result = fit_xgboost_with_selection(
            features,
            labels,
            fit,
            select,
            grid=({'max_depth': 2, 'n_estimators': 3},),
            estimator_factory=lambda kind, config: _RecordingEstimator(kind, config, log),
        )

        fit_rows = [rows for _, operation, rows in log if operation == 'fit']
        score_rows = [rows for _, operation, rows in log if operation == 'predict']
        self.assertEqual(len(fit_rows), 2)
        self.assertTrue(all(np.array_equal(rows, features[fit].numpy()) for rows in fit_rows))
        self.assertTrue(all(np.array_equal(rows, features[select].numpy()) for rows in score_rows))
        self.assertEqual(result.midpoint_config, {'max_depth': 2, 'n_estimators': 3})
        self.assertEqual(result.quantile_config, {'max_depth': 2, 'n_estimators': 3})
        self.assertTrue(result.native_quantiles)
        parameters = inspect.signature(fit_xgboost_with_selection).parameters
        self.assertNotIn('calibrate_ids', parameters)
        self.assertNotIn('test_ids', parameters)

    def test_xgboost_selects_midpoint_and_quantile_configs_independently(self) -> None:
        features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        labels = torch.tensor([0.1, 0.5, 0.9, 0.5])
        log: list[tuple[str, str, np.ndarray]] = []
        result = fit_xgboost_with_selection(
            features,
            labels,
            torch.tensor([0, 2]),
            torch.tensor([1, 3]),
            grid=(
                {'max_depth': 2, 'n_estimators': 3},
                {'max_depth': 4, 'n_estimators': 3},
            ),
            estimator_factory=lambda kind, config: _GridEstimator(kind, config, log),
        )

        self.assertEqual(result.midpoint_config['max_depth'], 2)
        self.assertEqual(result.quantile_config['max_depth'], 4)

    def test_label_propagation_has_full_graph_shape_and_only_explicit_fit_seeds(self) -> None:
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
        )
        fit_ids = torch.tensor([0, 3])
        fit_targets = torch.tensor([0.2, 0.8])

        first = regression_label_propagation(
            edge_index,
            num_nodes=6,
            fit_node_ids=fit_ids,
            fit_targets=fit_targets,
            damping=0.8,
            iterations=4,
        )
        second = regression_label_propagation(
            edge_index,
            num_nodes=6,
            fit_node_ids=fit_ids,
            fit_targets=fit_targets,
            damping=0.8,
            iterations=4,
        )

        self.assertEqual(tuple(first.shape), (6,))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[fit_ids], fit_targets)
        parameters = inspect.signature(regression_label_propagation).parameters
        self.assertNotIn('labels', parameters)
        self.assertNotIn('select_node_ids', parameters)
        self.assertNotIn('calibrate_node_ids', parameters)
        self.assertNotIn('test_node_ids', parameters)

    def test_label_propagation_rejects_nondeterministic_gpu_execution(self) -> None:
        edge_index = torch.empty((2, 0), dtype=torch.long, device='meta')
        with self.assertRaisesRegex(ValueError, 'CPU'):
            regression_label_propagation(
                edge_index,
                num_nodes=2,
                fit_node_ids=torch.tensor([0]),
                fit_targets=torch.tensor([0.5]),
                damping=0.8,
                iterations=1,
            )

    def test_lp_grid_reads_select_only_for_selection_not_as_seeds(self) -> None:
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
        )
        labels = torch.tensor([0.1, 0.4, 0.7, 0.9, -1.0])
        result = select_label_propagation(
            edge_index,
            num_nodes=5,
            labels=labels,
            fit_node_ids=torch.tensor([0, 2]),
            select_node_ids=torch.tensor([1, 3]),
            grid=((0.8, 2), (0.9, 3)),
        )

        self.assertEqual(tuple(result.predictions.shape), (5,))
        torch.testing.assert_close(result.predictions[torch.tensor([0, 2])], labels[torch.tensor([0, 2])])
        self.assertIn((result.damping, result.iterations), {(0.8, 2), (0.9, 3)})

    def test_point_baseline_prediction_artifact_is_single_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'calibrate.pt'
            payload = write_baseline_role_predictions(
                torch.tensor([0.2, 0.4, 0.6, 0.8]),
                torch.tensor([3, 1]),
                torch.tensor([-1.0, 0.3, -1.0, 0.9]),
                output,
                role='calibrate',
                arm='RegressionLP',
            )

            self.assertEqual(payload['seed'], 'deterministic')
            self.assertEqual(tuple(payload['predictions'].shape), (2, 1))
            torch.testing.assert_close(payload['labels'], torch.tensor([0.9, 0.3]))
            self.assertEqual(torch.load(output, weights_only=True)['uq_methods'], ['simple'])


if __name__ == '__main__':
    unittest.main()
