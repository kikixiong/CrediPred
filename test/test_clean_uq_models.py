"""Focused contracts for clean graph-UQ model variants."""

import importlib
import sys
import types
import unittest

import torch


def _install_torch_geometric_stub() -> None:
    """Make the real FF implementation importable on the torch-only host."""

    class _UnusedConv(torch.nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor, _edge_index: torch.Tensor) -> torch.Tensor:
            return x

    geometric = types.ModuleType('torch_geometric')
    nn_module = types.ModuleType('torch_geometric.nn')
    for name in ('GATConv', 'GATv2Conv', 'GCNConv', 'GINConv', 'SAGEConv'):
        setattr(nn_module, name, _UnusedConv)
    geometric.nn = nn_module
    sys.modules.setdefault('torch_geometric', geometric)
    sys.modules.setdefault('torch_geometric.nn', nn_module)


class CleanUQModelTest(unittest.TestCase):
    def test_ff_model_forward_needs_no_edge_index(self) -> None:
        """Catches reintroducing a mandatory graph argument on the no-edge arm."""
        _install_torch_geometric_stub()
        model_module = importlib.import_module('credipred.gnn.model')
        model = model_module.Model(
            model_name='FF',
            normalization='none',
            in_channels=4,
            hidden_channels=8,
            out_channels=6,
            num_layers=2,
            dropout=0.0,
            binary=False,
            prediction_dim=3,
        )

        prediction = model(torch.ones(5, 4))

        self.assertEqual(tuple(prediction.shape), (5, 3))

    def test_correction_arms_share_three_column_additive_delta_contract(self) -> None:
        """Catches a scalar/bounded head or a non-zero initial correction."""
        _install_torch_geometric_stub()
        from credipred.conformal_regression.correction_mlp import CorrectionMLP
        from credipred.conformal_regression.correction_gnn import CorrectionGNN

        base_predictions = torch.rand(7, 3)
        edge_index = torch.empty(2, 0, dtype=torch.long)

        for model in (
            CorrectionMLP(hidden_channels=8, num_layers=2, dropout=0.0),
            CorrectionGNN(
                hidden_channels=8,
                num_layers=2,
                dropout=0.0,
                normalization='none',
            ),
        ):
            with self.subTest(model=type(model).__name__):
                delta = model(base_predictions, edge_index)

                self.assertEqual(tuple(delta.shape), (7, 3))
                torch.testing.assert_close(delta, torch.zeros_like(delta))


if __name__ == '__main__':
    unittest.main()
