"""Conformalized Quantile Regression (CQR) and evaluation utilities.

Provides:
- CQR calibration: compute qhat from calibration set
- Interval adjustment: widen intervals by qhat
- Coverage and width evaluation
- Stratified conformal prediction for better coverage uniformity
- Mondrian conformal prediction (group by external attribute, e.g. degree)
- Locally adaptive conformal prediction (KNN-based local qhat)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def compute_cqr_scores(
    lower: torch.Tensor, upper: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """CQR conformity scores: max(lower - y, y - upper)."""
    return torch.maximum(y_true - upper, lower - y_true)


def compute_qhat(scores: torch.Tensor, alpha: float) -> float:
    """Compute conformal quantile from calibration scores."""
    n = scores.numel()
    k = max(1, min(math.ceil((n + 1) * (1 - alpha)), n))
    return torch.kthvalue(scores.reshape(-1), k).values.item()


def adjust_intervals(
    lower: torch.Tensor, upper: torch.Tensor, qhat: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Widen prediction intervals by qhat."""
    return lower - qhat, upper + qhat


@dataclass
class IntervalMetrics:
    coverage: float
    avg_width: float
    mae: float
    qhat: float


def evaluate_intervals(
    preds: torch.Tensor,
    labels: torch.Tensor,
    calib_idx: torch.Tensor,
    test_idx: torch.Tensor,
    alpha: float,
) -> IntervalMetrics:
    """Evaluate CQR prediction intervals.

    Args:
        preds: [N, 3] tensor with [mid, lower, upper].
        labels: [N] true values.
        calib_idx: indices for qhat calibration.
        test_idx: indices for evaluation.
        alpha: miscoverage rate.

    Returns:
        IntervalMetrics with coverage, avg_width, mae, qhat.
    """
    mid, lower, upper = preds[:, 0], preds[:, 1], preds[:, 2]

    cal_scores = compute_cqr_scores(lower[calib_idx], upper[calib_idx], labels[calib_idx])
    qhat = compute_qhat(cal_scores, alpha)

    lower_adj, upper_adj = adjust_intervals(lower[test_idx], upper[test_idx], qhat)
    labels_test = labels[test_idx]

    coverage = ((labels_test >= lower_adj) & (labels_test <= upper_adj)).float().mean().item()
    avg_width = (upper_adj - lower_adj).mean().item()
    mae = torch.abs(mid[test_idx] - labels_test).mean().item()

    return IntervalMetrics(coverage=coverage, avg_width=avg_width, mae=mae, qhat=qhat)


class MondrianConformalPredictor:
    """Mondrian CQR: group by an external attribute (e.g. node degree).

    Computes per-group qhat. More semantically meaningful than width-based
    stratification -- groups reflect structural properties of the graph.

    Args:
        n_groups: number of groups (quantile-based binning).
        alpha: miscoverage rate.
    """

    def __init__(self, n_groups: int = 3, alpha: float = 0.05):
        self.n_groups = n_groups
        self.alpha = alpha
        self.qhat_per_group_: np.ndarray | None = None
        self.group_boundaries_: np.ndarray | None = None
        self.global_qhat_: float | None = None

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def fit(
        self,
        lower_calib: np.ndarray | torch.Tensor,
        upper_calib: np.ndarray | torch.Tensor,
        y_true_calib: np.ndarray | torch.Tensor,
        group_feature_calib: np.ndarray | torch.Tensor,
    ) -> 'MondrianConformalPredictor':
        lower_calib = self._to_numpy(lower_calib)
        upper_calib = self._to_numpy(upper_calib)
        y_true_calib = self._to_numpy(y_true_calib)
        group_feature_calib = self._to_numpy(group_feature_calib).astype(float)

        scores = np.maximum(y_true_calib - upper_calib, lower_calib - y_true_calib)

        # Quantile-based grouping on the feature
        percentiles = np.linspace(0, 100, self.n_groups + 1)
        boundaries = np.unique(np.percentile(group_feature_calib, percentiles))
        if len(boundaries) < self.n_groups + 1:
            boundaries = np.linspace(
                group_feature_calib.min(), group_feature_calib.max(),
                self.n_groups + 1,
            )
        self.group_boundaries_ = boundaries
        assignments = np.clip(
            np.digitize(group_feature_calib, boundaries[1:-1]),
            0, self.n_groups - 1,
        )

        # Global qhat as fallback
        n = len(scores)
        q_level = min(max(math.ceil((n + 1) * (1 - self.alpha)) / n, 1e-6), 1.0)
        self.global_qhat_ = float(np.quantile(scores, q_level, method='higher'))

        # Per-group qhat
        self.qhat_per_group_ = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            mask = assignments == g
            n_g = mask.sum()
            if n_g > 0:
                q_g = min(max(math.ceil((n_g + 1) * (1 - self.alpha)) / n_g, 1e-6), 1.0)
                self.qhat_per_group_[g] = np.quantile(scores[mask], q_g, method='higher')
            else:
                self.qhat_per_group_[g] = self.global_qhat_

        return self

    def predict(
        self,
        lower_test: np.ndarray | torch.Tensor,
        upper_test: np.ndarray | torch.Tensor,
        group_feature_test: np.ndarray | torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_test = self._to_numpy(lower_test)
        upper_test = self._to_numpy(upper_test)
        group_feature_test = self._to_numpy(group_feature_test).astype(float)

        assignments = np.clip(
            np.digitize(group_feature_test, self.group_boundaries_[1:-1]),
            0, self.n_groups - 1,
        )
        qhat_per_sample = self.qhat_per_group_[assignments]
        return lower_test - qhat_per_sample, upper_test + qhat_per_sample, assignments
