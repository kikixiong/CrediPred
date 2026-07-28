"""Tests for immutable in-memory model checkpoint snapshots."""

import unittest

import torch

from credipred.utils.checkpoint import snapshot_state_dict


class CheckpointSnapshotTest(unittest.TestCase):
    def test_snapshot_does_not_drift_when_model_keeps_training(self) -> None:
        model = torch.nn.Linear(2, 1)
        snapshot = snapshot_state_dict(model)
        saved_weight = snapshot['weight'].clone()
        saved_bias = snapshot['bias'].clone()

        with torch.no_grad():
            model.weight.add_(10)
            model.bias.sub_(10)

        torch.testing.assert_close(snapshot['weight'], saved_weight)
        torch.testing.assert_close(snapshot['bias'], saved_bias)
        self.assertNotEqual(
            snapshot['weight'].data_ptr(),
            model.state_dict()['weight'].data_ptr(),
        )


if __name__ == '__main__':
    unittest.main()
