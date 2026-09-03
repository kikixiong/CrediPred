"""Role-masked target artifact tests for the clean-UQ phase boundary."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    load_role_targets,
    write_role_target_artifacts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_partitions(path: Path) -> None:
    rows = [
        {'node_id': 0, 'mapped': True, 'labelled': True, 'clean_uq_split': 'fit'},
        {'node_id': 2, 'mapped': True, 'labelled': True, 'clean_uq_split': 'select'},
        {'node_id': 4, 'mapped': True, 'labelled': True, 'clean_uq_split': 'calibrate'},
        {'node_id': 5, 'mapped': True, 'labelled': True, 'clean_uq_split': 'test'},
    ]
    path.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'
    )


class CleanUqTargetsTest(unittest.TestCase):
    def test_writer_physically_separates_pretest_and_sealed_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            partitions = root / 'partitions.jsonl'
            labels_path = root / 'labels.pt'
            _write_partitions(partitions)
            torch.save(torch.tensor([0.1, -1.0, 0.3, -1.0, 0.7, 0.9]), labels_path)

            paths = write_role_target_artifacts(
                labels_path,
                partitions,
                root / 'protocol' / 'targets',
                root / 'sealed' / 'targets',
                expected_labels_sha256=_sha256(labels_path),
                num_nodes=6,
            )

            fit_select = load_role_targets(
                paths['fit_select'],
                partition_jsonl=partitions,
                expected_roles=('fit', 'select'),
                num_nodes=6,
            )
            calibrate = load_role_targets(
                paths['calibrate'],
                partition_jsonl=partitions,
                expected_roles=('calibrate',),
                num_nodes=6,
            )
            sealed = load_role_targets(
                paths['test'],
                partition_jsonl=partitions,
                expected_roles=('test',),
                num_nodes=6,
            )

            self.assertEqual(torch.nonzero(fit_select >= 0).flatten().tolist(), [0, 2])
            self.assertEqual(torch.nonzero(calibrate >= 0).flatten().tolist(), [4])
            self.assertEqual(torch.nonzero(sealed >= 0).flatten().tolist(), [5])
            self.assertNotEqual(paths['fit_select'].parent, paths['test'].parent)
            self.assertTrue(paths['ledger'].is_file())

    def test_loader_rejects_raw_or_overexposed_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            partitions = root / 'partitions.jsonl'
            _write_partitions(partitions)
            raw = root / 'raw.pt'
            torch.save(torch.tensor([0.1, -1.0, 0.3, -1.0, 0.7, 0.9]), raw)
            with self.assertRaisesRegex(ValueError, 'role-target'):
                load_role_targets(
                    raw,
                    partition_jsonl=partitions,
                    expected_roles=('fit', 'select'),
                    num_nodes=6,
                )

            payload = {
                'schema_version': 'clean-uq-role-targets-v1',
                'authorized_roles': ['fit', 'select'],
                'num_nodes': 6,
                'partition_sha256': _sha256(partitions),
                'source_labels_sha256': '0' * 64,
                'sentinel': -1.0,
                'labels': torch.tensor([0.1, -1.0, 0.3, -1.0, 0.7, -1.0]),
            }
            overexposed = root / 'overexposed.pt'
            torch.save(payload, overexposed)
            with self.assertRaisesRegex(ValueError, 'exposed'):
                load_role_targets(
                    overexposed,
                    partition_jsonl=partitions,
                    expected_roles=('fit', 'select'),
                    num_nodes=6,
                )

    def test_writer_rejects_a_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            partitions = root / 'partitions.jsonl'
            labels_path = root / 'labels.pt'
            _write_partitions(partitions)
            torch.save(torch.tensor([0.1, -1.0, 0.3, -1.0, 0.7, 0.9]), labels_path)

            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                write_role_target_artifacts(
                    labels_path,
                    partitions,
                    root / 'protocol-targets',
                    root / 'sealed-targets',
                    expected_labels_sha256='0' * 64,
                    num_nodes=6,
                )


if __name__ == '__main__':
    unittest.main()
