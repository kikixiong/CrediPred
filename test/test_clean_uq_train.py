"""Torch-only tests for clean three-head training and role isolation."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from credipred.experiments.gnn_experiments.clean_uq_train import (
    TrainingRoles,
    _build_parser,
    _quantile_loss,
    _run_predict_role,
    fit_with_selection,
    load_training_roles,
    write_cached_role_predictions,
    write_role_predictions,
)


class _Batch:
    def __init__(self) -> None:
        self.x = torch.ones(1, 1)
        self.edge_index = torch.empty(2, 0, dtype=torch.long)
        self.n_id = torch.tensor([0])
        self.batch_size = 1

    def to(self, _device: torch.device) -> '_Batch':
        return self


class _SharedHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: _Batch) -> torch.Tensor:
        return self.value.expand(batch.batch_size, 3)


class CleanUQTrainTest(unittest.TestCase):
    def test_cli_exposes_four_explicit_run_local_seeded_stages(self) -> None:
        """Catches returning to an implicit runner or historical fixed paths."""
        parser = _build_parser()
        stage_arguments = {
            'base-fit': ['--data-pt', 'data.pt', '--labels-pt', 'labels.pt', '--partition-jsonl', 'parts.jsonl', '--model', 'FF'],
            'parent-cache': ['--data-pt', 'data.pt'],
            'correction-fit': ['--data-pt', 'data.pt', '--labels-pt', 'labels.pt', '--partition-jsonl', 'parts.jsonl', '--arm', 'mlp'],
            'predict-role': ['--data-pt', 'data.pt', '--labels-pt', 'labels.pt', '--partition-jsonl', 'parts.jsonl', '--role', 'calibrate', '--arm', 'GAT'],
        }
        for stage, extra in stage_arguments.items():
            with self.subTest(stage=stage):
                parsed = parser.parse_args(
                    [stage, '--run-dir', 'fresh-run', '--seed', '42', *extra]
                )
                self.assertEqual(parsed.stage, stage)
                self.assertEqual(parsed.run_dir, Path('fresh-run'))
                self.assertEqual(parsed.seed, 42)
                self.assertFalse(hasattr(parsed, 'seeds'))

    def test_quantile_heads_target_mid_lower_upper_in_that_order(self) -> None:
        """Catches swapping the .05/.95 heads or using one tail weight twice."""
        predictions = torch.tensor([[0.0, 0.0, 1.0]])
        targets = torch.tensor([0.0])

        loss = _quantile_loss(predictions, targets, alpha=0.05)

        torch.testing.assert_close(loss, torch.tensor(0.05))

    def test_partition_reader_exposes_only_fit_and_select_to_training(self) -> None:
        """Catches adding calibrate/test cohorts to the trainer-facing contract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'clean_uq_partitions.jsonl'
            rows = [
                {'node_id': 3, 'mapped': True, 'labelled': True, 'clean_uq_split': 'fit'},
                {'node_id': 7, 'mapped': True, 'labelled': True, 'clean_uq_split': 'select'},
                {'node_id': 11, 'mapped': True, 'labelled': True, 'clean_uq_split': 'calibrate'},
                {'node_id': 13, 'mapped': True, 'labelled': True, 'clean_uq_split': 'test'},
            ]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

            roles = load_training_roles(path)

            self.assertEqual(tuple(TrainingRoles.__dataclass_fields__), ('fit', 'select'))
            torch.testing.assert_close(roles.fit, torch.tensor([3]))
            torch.testing.assert_close(roles.select, torch.tensor([7]))

    def test_selection_checkpoint_is_an_immutable_best_epoch_snapshot(self) -> None:
        """Catches selecting on fit loss or retaining live state_dict tensors."""
        model = _SharedHead()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        batch = _Batch()

        result = fit_with_selection(
            model,
            fit_loader=[batch],
            select_loader=[batch],
            labels=torch.tensor([1.0]),
            optimizer=optimizer,
            epochs=2,
            predict_batch=lambda current_model, current_batch: current_model(current_batch),
            alpha=0.05,
        )

        self.assertAlmostEqual(float(model.value.detach()), 0.0, places=6)
        self.assertAlmostEqual(float(result.state_dict['value']), 1.5, places=6)
        self.assertLess(result.select_loss, 1.5)
        parameters = inspect.signature(fit_with_selection).parameters
        self.assertNotIn('calibrate_loader', parameters)
        self.assertNotIn('test_loader', parameters)

    def test_role_prediction_artifact_keeps_external_labels_and_node_ids_aligned(self) -> None:
        """Catches writing neighborhood rows instead of requested seed-node rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = _SharedHead()
            with torch.no_grad():
                model.value.fill_(0.25)
            batch = _Batch()
            batch.n_id = torch.tensor([1])
            path = Path(temp_dir) / 'calibrate.pt'

            payload = write_role_predictions(
                model,
                loader=[batch],
                labels=torch.tensor([-1.0, 0.75]),
                predict_batch=lambda current_model, current_batch: current_model(current_batch),
                output_path=path,
                role='calibrate',
                seed=42,
                arm='FF',
            )

            torch.testing.assert_close(payload['node_ids'], torch.tensor([1]))
            torch.testing.assert_close(payload['labels'], torch.tensor([0.75]))
            torch.testing.assert_close(payload['predictions'], torch.full((1, 3), 0.25))
            self.assertEqual(torch.load(path, weights_only=True)['role'], 'calibrate')

    def test_gat_role_prediction_is_an_aligned_frozen_parent_cache_slice(self) -> None:
        """Catches resampling the GAT parent or misaligning cache rows and labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'calibrate.pt'
            parent_predictions = torch.arange(15, dtype=torch.float32).reshape(5, 3)
            labels = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
            node_ids = torch.tensor([3, 1])

            payload = write_cached_role_predictions(
                parent_predictions,
                node_ids,
                labels,
                path,
                role='calibrate',
                seed=42,
            )

            torch.testing.assert_close(payload['node_ids'], node_ids)
            torch.testing.assert_close(payload['predictions'], parent_predictions[node_ids])
            torch.testing.assert_close(payload['labels'], labels[node_ids])
            self.assertEqual(payload['arm'], 'GAT')

    def test_gat_predict_stage_reads_cache_without_a_checkpoint_or_loader(self) -> None:
        """Catches routing the paired GAT arm back through stochastic sampling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / 'data.pt'
            labels_path = root / 'labels.pt'
            partitions_path = root / 'partitions.jsonl'
            cache_path = root / 'parent.pt'
            torch.save(SimpleNamespace(num_nodes=5), data_path)
            labels = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
            torch.save(labels, labels_path)
            partitions_path.write_text(
                ''.join(
                    json.dumps(
                        {
                            'node_id': node_id,
                            'mapped': True,
                            'labelled': True,
                            'clean_uq_split': 'calibrate',
                        }
                    )
                    + '\n'
                    for node_id in (3, 1)
                )
            )
            parent_predictions = torch.arange(15, dtype=torch.float32).reshape(5, 3)
            torch.save(
                {
                    'schema_version': 'clean-uq-parent-cache-v1',
                    'seed': 42,
                    'arm': 'GAT',
                    'predictions': parent_predictions,
                },
                cache_path,
            )

            result = _run_predict_role(
                SimpleNamespace(
                    partition_jsonl=partitions_path,
                    role='calibrate',
                    data_pt=data_path,
                    labels_pt=labels_path,
                    run_dir=root / 'run',
                    arm='GAT',
                    seed=42,
                    output_pt=None,
                    parent_cache_pt=cache_path,
                )
            )

            payload = torch.load(result['predictions'], weights_only=True)
            torch.testing.assert_close(payload['node_ids'], torch.tensor([3, 1]))
            torch.testing.assert_close(
                payload['predictions'], parent_predictions[torch.tensor([3, 1])]
            )
            torch.testing.assert_close(payload['labels'], labels[torch.tensor([3, 1])])
            self.assertEqual(result['source'], 'frozen-parent-cache')


if __name__ == '__main__':
    unittest.main()
