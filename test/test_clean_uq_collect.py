from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'noether' / 'clean_uq_collect.py'
SPEC = importlib.util.spec_from_file_location('clean_uq_collect', SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


ARMS = ('FF', 'GCN', 'SAGE', 'GAT', 'GAT-mlp', 'GAT-topology')


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


def _build_fixture(root: Path, seeds: list[int], *, count: int = 2) -> None:
    protocol_dir = root / 'protocol'
    protocol_dir.mkdir(parents=True)
    rows = []
    node_id = 0
    for role, role_count in (
        ('fit', 2),
        ('select', 1),
        ('calibrate', count),
        ('test', count),
    ):
        for _ in range(role_count):
            rows.append(
                {
                    'canonical_domain': f'domain-{node_id}.example',
                    'group_id': f'domain:domain-{node_id}.example',
                    'node_id': node_id,
                    'mapped': True,
                    'labelled': True,
                    'clean_uq_split': role,
                }
            )
            node_id += 1
    (protocol_dir / 'clean_uq_partitions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'
    )
    partition_counts = {
        role: sum(row['clean_uq_split'] == role for row in rows)
        for role in ('fit', 'select', 'calibrate', 'test')
    }
    _write_json(
        protocol_dir / 'clean_uq_protocol.json',
        {
            'schema_version': 'clean-uq-protocol-v1',
            'model_seeds': seeds,
            'partition_counts': partition_counts,
        },
    )
    observed = len(rows)
    _write_json(
        protocol_dir / 'clean_uq_population_ledger.json',
        {
            'source_population_count': observed + 2,
            'observed_population_count': observed,
            'excluded_before_observation_count': 2,
            'mapped_count': observed,
            'mapped_labelled_count': observed,
            'excluded_unmapped_count': 0,
            'excluded_unlabelled_count': 0,
            'sealed_test_count': 0,
            'sealed_test_group_count': 0,
        },
    )

    role_rows = {
        role: [row for row in rows if row['clean_uq_split'] == role]
        for role in ('calibrate', 'test')
    }
    for seed in seeds:
        (root / 'cache' / 'GAT').mkdir(parents=True, exist_ok=True)
        (root / 'cache' / 'GAT' / f'seed-{seed}.pt').touch()
        for arm in ARMS:
            checkpoint = root / 'checkpoints' / arm / f'seed-{seed}.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
            for role in ('calibrate', 'test'):
                selected = role_rows[role]
                node_ids = torch.tensor([row['node_id'] for row in selected])
                labels = torch.linspace(0.2, 0.8, len(selected))
                prediction = root / 'predictions' / arm / f'seed-{seed}' / f'{role}.pt'
                prediction.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'schema_version': 'clean-uq-predictions-v1',
                        'role': role,
                        'seed': seed,
                        'arm': arm,
                        'node_ids': node_ids,
                        'labels': labels,
                        'predictions': torch.stack(
                            (labels, labels - 0.1, labels + 0.1), dim=1
                        ),
                    },
                    prediction,
                )
            _write_json(
                root / 'audit' / str(seed) / f'{arm}_calibration.json',
                {
                    'schema_version': 'clean-uq-calibration-v1',
                    'seed': seed,
                    'arm': arm,
                    'alpha': 0.1,
                    'calibration_count': count,
                },
            )
            _write_json(
                root / 'audit' / str(seed) / f'{arm}_test.json',
                [
                    {
                        'schema_version': 'clean-uq-audit-row-v1',
                        'seed': seed,
                        'arm': arm,
                        'method': method,
                        'alpha': 0.1,
                        'test_count': count,
                        'coverage': 1.0,
                    }
                    for method in ('simple', 'cqr')
                ],
            )

    for role in ('calibrate', 'test'):
        selected = role_rows[role]
        node_ids = torch.tensor([row['node_id'] for row in selected])
        labels = torch.linspace(0.2, 0.8, len(selected))
        path = root / 'predictions' / 'GAT-ensemble' / f'{role}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'schema_version': 'clean-uq-predictions-v1',
                'role': role,
                'seed': 'ensemble',
                'arm': 'GAT-ensemble',
                'member_seeds': seeds,
                'node_ids': node_ids,
                'labels': labels,
                'predictions': torch.stack(
                    (labels, labels - 0.1, labels + 0.1), dim=1
                ),
            },
            path,
        )
    _write_json(
        root / 'audit' / 'ensemble' / 'GAT-ensemble_calibration.json',
        {
            'schema_version': 'clean-uq-calibration-v1',
            'seed': 'ensemble',
            'arm': 'GAT-ensemble',
            'member_seeds': seeds,
            'alpha': 0.1,
            'calibration_count': count,
        },
    )
    _write_json(
        root / 'audit' / 'ensemble' / 'GAT-ensemble_test.json',
        [
            {
                'schema_version': 'clean-uq-audit-row-v1',
                'seed': 'ensemble',
                'arm': 'GAT-ensemble',
                'method': method,
                'alpha': 0.1,
                'test_count': count,
                'coverage': 1.0,
            }
            for method in ('simple', 'cqr')
        ],
    )


class CleanUqCollectTest(unittest.TestCase):
    def test_pretest_reads_only_calibration_and_rejects_test_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42, 43])
            for path in (root / 'predictions').rglob('test.pt'):
                path.unlink()
            for path in (root / 'audit').rglob('*_test.json'):
                path.unlink()

            loaded: list[Path] = []
            real_load = torch.load

            def recording_load(path: Path, **kwargs: object) -> object:
                loaded.append(Path(path))
                return real_load(path, **kwargs)

            with mock.patch.object(collect.torch, 'load', side_effect=recording_load):
                ledger = collect.collect_pretest(root)
            self.assertTrue(loaded)
            self.assertTrue(all(path.name == 'calibrate.pt' for path in loaded))
            self.assertEqual(ledger['observed_artifacts']['calibration_predictions'], 13)

            remnants = (
                root / 'predictions' / 'FF' / 'seed-42' / 'test.pt',
                root / 'audit' / '42' / 'FF_test.json',
                root / 'audit' / '42' / 'FF_test.csv',
                root / 'results' / 'clean_uq_metrics.csv',
                root / 'results' / 'clean_uq_exposure_ledger.json',
            )
            for exposed in remnants:
                with self.subTest(exposed=exposed.name):
                    exposed.parent.mkdir(parents=True, exist_ok=True)
                    exposed.write_text('must never be opened', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'sealed-test artifact'):
                        collect.collect_pretest(root, write_output=False)
                    exposed.unlink()

    def test_sealed_test_writes_all_62_unaggregated_rows_for_five_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42, 43, 44, 45, 46])

            ledger = collect.collect_sealed_test(root)
            with (root / 'results' / 'clean_uq_metrics.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 62)
            self.assertEqual(ledger['expected']['metric_rows'], 62)
            self.assertEqual(ledger['observed']['metric_rows'], 62)
            self.assertEqual(ledger['unique_domain_exposure']['calibrate'], 2)
            self.assertEqual(ledger['unique_domain_exposure']['test'], 2)
            self.assertEqual(
                ledger['population_funnel']['mapped_labelled_count'], 7
            )

    def test_identity_and_dynamic_denominator_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42])
            for path in (root / 'predictions').rglob('test.pt'):
                path.unlink()
            for path in (root / 'audit').rglob('*_test.json'):
                path.unlink()

            state_path = root / 'audit' / '42' / 'FF_calibration.json'
            state = json.loads(state_path.read_text())
            state['calibration_count'] = 3
            _write_json(state_path, state)
            with self.assertRaisesRegex(ValueError, 'calibration_count'):
                collect.collect_pretest(root, write_output=False)

            state['calibration_count'] = 2
            state['arm'] = 'GCN'
            _write_json(state_path, state)
            with self.assertRaisesRegex(ValueError, 'identity'):
                collect.collect_pretest(root, write_output=False)


if __name__ == '__main__':
    unittest.main()
