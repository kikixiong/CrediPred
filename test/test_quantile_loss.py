"""Regression tests for the three quantile-training loss implementations."""

import ast
import unittest
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
LOSS_MODULES = (
    Path('credipred/experiments/gnn_experiments/regression_uq_experiment.py'),
    Path('credipred/experiments/gnn_experiments/uncertainty_gat_experiment.py'),
    Path('credipred/experiments/gnn_experiments/topology_correction_experiment.py'),
)


def _load_quantile_loss(module_path: Path) -> Callable:
    """Load only ``_quantile_loss`` so this unit test needs no GNN runtime."""
    source_path = REPO_ROOT / module_path
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    loss_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_quantile_loss'
    )
    namespace = {'torch': torch, 'F': F, 'Tensor': torch.Tensor}
    exec(
        compile(
            ast.Module(body=[loss_function], type_ignores=[]),
            filename=str(source_path),
            mode='exec',
        ),
        namespace,
    )
    return namespace['_quantile_loss']


def _load_base_evaluate() -> Callable:
    """Load the base quantile loss and evaluator without GNN dependencies."""
    module_path = LOSS_MODULES[0]
    source_path = REPO_ROOT / module_path
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {'_quantile_loss', 'evaluate'}
    ]
    namespace = {
        'torch': torch,
        'F': F,
        'Tensor': torch.Tensor,
        'Tuple': Tuple,
        'NeighborLoader': object,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            filename=str(source_path),
            mode='exec',
        ),
        namespace,
    )
    return namespace['evaluate']


class _SingleBatch:
    def __init__(self) -> None:
        self.x = torch.zeros(1, 1)
        self.edge_index = torch.empty(2, 0, dtype=torch.long)
        self.y = torch.tensor([0.0])
        self.valid_mask = torch.tensor([True])
        self.batch_size = 1

    def to(self, _device: torch.device) -> '_SingleBatch':
        return self


class _FixedPredictionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        _x: torch.Tensor,
        _edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, 1.0]]) + self.anchor * 0


class QuantileLossTest(unittest.TestCase):
    def test_upper_quantile_uses_opposite_tail_weights(self) -> None:
        alpha = 0.05
        targets = torch.tensor([0.0])

        for module_path in LOSS_MODULES:
            with self.subTest(module=str(module_path)):
                loss_function = _load_quantile_loss(module_path)

                upper_above_target = torch.tensor([[0.0, 0.0, 1.0]])
                actual_above = loss_function(
                    upper_above_target,
                    targets,
                    alpha,
                )
                torch.testing.assert_close(actual_above, torch.tensor(alpha))

                upper_below_target = torch.tensor([[0.0, 0.0, -1.0]])
                actual_below = loss_function(
                    upper_below_target,
                    targets,
                    alpha,
                )
                torch.testing.assert_close(
                    actual_below,
                    torch.tensor(1 - alpha),
                )

    def test_base_evaluator_returns_quantile_loss_for_checkpoint_selection(
        self,
    ) -> None:
        evaluate = _load_base_evaluate()
        result = evaluate(
            _FixedPredictionModel(),
            [_SingleBatch()],
            'valid_mask',
            alpha=0.05,
        )

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[3], 0.05, places=6)


if __name__ == '__main__':
    unittest.main()
