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
BASELINE_METHODS = {
    'GlobalMedian': ('simple',),
    'XGBoost': ('simple', 'cqr'),
    'RegressionLP': ('simple',),
}


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
                    'prediction_kind': 'point-with-endpoints',
                    'methods': ['simple', 'cqr'],
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
                        'qhat': 0.1,
                        'coverage': 1.0,
                        'mean_width': 0.2,
                        'median_width': 0.2,
                        'interval_score': 0.2,
                        'midpoint_mae': 0.0,
                        'midpoint_rmse': 0.0,
                        'midpoint_spearman': 1.0,
                        'raw_crossing_rate': 0.0,
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
            'prediction_kind': 'point-with-endpoints',
            'methods': ['simple', 'cqr'],
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
                'qhat': 0.1,
                'coverage': 1.0,
                'mean_width': 0.2,
                'median_width': 0.2,
                'interval_score': 0.2,
                'midpoint_mae': 0.0,
                'midpoint_rmse': 0.0,
                'midpoint_spearman': 1.0,
                'raw_crossing_rate': 0.0,
            }
            for method in ('simple', 'cqr')
        ],
    )

    num_nodes = len(rows)
    graph_edges = 12
    lp_cache = root / 'cache' / 'RegressionLP' / 'deterministic.pt'
    lp_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'schema_version': 'clean-uq-lp-cache-v1',
            'full_graph': True,
            'directed': True,
            'num_nodes': num_nodes,
            'num_edges': graph_edges,
            'fit_seed_count': 2,
            'estimand': collect.LP_ESTIMAND,
            'predictions': torch.full((num_nodes,), 0.5),
            'reached': torch.ones(num_nodes, dtype=torch.bool),
        },
        lp_cache,
    )
    _write_json(
        root / 'probe' / 'regression_lp.json',
        {
            'schema_version': 'clean-uq-lp-probe-v1',
            'full_graph': True,
            'directed': True,
            'num_nodes': num_nodes,
            'num_edges': graph_edges,
            'estimand': collect.LP_ESTIMAND,
            'projected_passes': collect.LP_GRAPH_PASSES,
            'execution_device': 'cpu',
            'deterministic_algorithms': True,
            'passed': True,
        },
    )
    midpoint_model = root / 'checkpoints' / 'XGBoost' / 'midpoint.ubj'
    quantile_model = root / 'checkpoints' / 'XGBoost' / 'quantiles.ubj'
    midpoint_model.parent.mkdir(parents=True, exist_ok=True)
    midpoint_model.touch()
    quantile_model.touch()
    baseline_states = {
        'GlobalMedian': {
            'fit_count': 2,
            'median': 0.5,
            'prediction_kind': 'point-only',
        },
        'XGBoost': {
            'deterministic': True,
            'xgboost_distribution': 'xgboost-cpu',
            'xgboost_version': '3.2.0',
            'midpoint_model': str(midpoint_model),
            'quantile_model': str(quantile_model),
            'prediction_kind': 'point-with-endpoints',
        },
        'RegressionLP': {
            'full_graph': True,
            'directed': True,
            'num_nodes': num_nodes,
            'num_edges': graph_edges,
            'estimand': collect.LP_ESTIMAND,
            'graph_passes': collect.LP_GRAPH_PASSES,
            'deterministic_algorithms': True,
            'execution_device': 'cpu',
            'fit_seed_count': 2,
            'cache': str(lp_cache),
            'prediction_kind': 'point-only',
        },
    }
    for arm, methods in BASELINE_METHODS.items():
        _write_json(
            root / 'checkpoints' / arm / 'deterministic.json',
            {
                'schema_version': 'clean-uq-baseline-state-v1',
                'seed': 'deterministic',
                'arm': arm,
                'methods': list(methods),
                **baseline_states[arm],
            },
        )
        for role in ('calibrate', 'test'):
            selected = role_rows[role]
            node_ids = torch.tensor([row['node_id'] for row in selected])
            labels = torch.linspace(0.2, 0.8, len(selected))
            if methods == ('simple',):
                predictions = labels[:, None]
            else:
                predictions = torch.stack(
                    (labels, labels - 0.1, labels + 0.1), dim=1
                )
            path = root / 'predictions' / arm / 'deterministic' / f'{role}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'schema_version': 'clean-uq-predictions-v1',
                    'role': role,
                    'seed': 'deterministic',
                    'arm': arm,
                    'node_ids': node_ids,
                    'labels': labels,
                    'predictions': predictions,
                    'uq_methods': list(methods),
                },
                path,
            )
        _write_json(
            root / 'audit' / 'deterministic' / f'{arm}_calibration.json',
            {
                'schema_version': 'clean-uq-calibration-v1',
                'seed': 'deterministic',
                'arm': arm,
                'alpha': 0.1,
                'calibration_count': count,
                'prediction_kind': (
                    'point-only'
                    if methods == ('simple',)
                    else 'point-with-endpoints'
                ),
                'methods': list(methods),
            },
        )
        _write_json(
            root / 'audit' / 'deterministic' / f'{arm}_test.json',
            [
                {
                    'schema_version': 'clean-uq-audit-row-v1',
                    'seed': 'deterministic',
                    'arm': arm,
                    'method': method,
                    'alpha': 0.1,
                    'test_count': count,
                    'qhat': 0.1,
                    'coverage': 1.0,
                    'mean_width': 0.2,
                    'median_width': 0.2,
                    'interval_score': 0.2,
                    'midpoint_mae': 0.0,
                    'midpoint_rmse': 0.0,
                    'midpoint_spearman': None,
                    'raw_crossing_rate': (
                        None if methods == ('simple',) else 0.0
                    ),
                }
                for method in methods
            ],
        )


class CleanUqCollectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._node_count = mock.patch.object(collect, 'FULL_GRAPH_NODE_COUNT', 7)
        self._edge_count = mock.patch.object(collect, 'FULL_GRAPH_EDGE_COUNT', 12)
        self._node_count.start()
        self._edge_count.start()

    def tearDown(self) -> None:
        self._edge_count.stop()
        self._node_count.stop()

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
            self.assertFalse(any(path.name == 'test.pt' for path in loaded))
            self.assertEqual(
                sum(path.name == 'calibrate.pt' for path in loaded), 16
            )
            self.assertEqual(ledger['observed_artifacts']['calibration_predictions'], 16)
            self.assertEqual(ledger['observed_artifacts']['baseline_states'], 3)
            self.assertEqual(ledger['observed_artifacts']['lp_caches'], 1)
            self.assertEqual(ledger['observed_artifacts']['lp_probes'], 1)

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

    def test_sealed_test_writes_all_66_unaggregated_rows_for_five_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42, 43, 44, 45, 46])

            ledger = collect.collect_sealed_test(root)
            with (root / 'results' / 'clean_uq_metrics.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 66)
            self.assertEqual(ledger['expected']['metric_rows'], 66)
            self.assertEqual(ledger['observed']['metric_rows'], 66)
            self.assertEqual(ledger['expected']['test_predictions'], 34)
            self.assertEqual(ledger['observed']['test_predictions'], 34)
            self.assertEqual(ledger['expected']['test_audits'], 34)
            self.assertEqual(ledger['observed']['test_audits'], 34)
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

    def test_prediction_columns_must_match_calibration_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42])
            for path in (root / 'predictions').rglob('test.pt'):
                path.unlink()
            for path in (root / 'audit').rglob('*_test.json'):
                path.unlink()

            state_path = root / 'audit' / '42' / 'FF_calibration.json'
            state = json.loads(state_path.read_text())
            state['methods'] = ['simple']
            state['prediction_kind'] = 'point-only'
            _write_json(state_path, state)
            with self.assertRaisesRegex(ValueError, 'prediction columns'):
                collect.collect_pretest(root, write_output=False)

    def test_pretest_requires_baseline_state_passed_probe_and_lp_cache(self) -> None:
        cases = (
            (
                Path('checkpoints/GlobalMedian/deterministic.json'),
                FileNotFoundError,
                'deterministic.json',
            ),
            (
                Path('cache/RegressionLP/deterministic.pt'),
                FileNotFoundError,
                'deterministic.pt',
            ),
        )
        for relative_path, error, message in cases:
            with self.subTest(path=str(relative_path)):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    _build_fixture(root, [42])
                    for path in (root / 'predictions').rglob('test.pt'):
                        path.unlink()
                    for path in (root / 'audit').rglob('*_test.json'):
                        path.unlink()
                    (root / relative_path).unlink()
                    with self.assertRaisesRegex(error, message):
                        collect.collect_pretest(root, write_output=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42])
            for path in (root / 'predictions').rglob('test.pt'):
                path.unlink()
            for path in (root / 'audit').rglob('*_test.json'):
                path.unlink()
            probe_path = root / 'probe' / 'regression_lp.json'
            probe = json.loads(probe_path.read_text())
            probe['passed'] = False
            _write_json(probe_path, probe)
            with self.assertRaisesRegex(ValueError, 'passed full-graph'):
                collect.collect_pretest(root, write_output=False)

    def test_pretest_rejects_unregistered_checkpoint_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42])
            for path in (root / 'predictions').rglob('test.pt'):
                path.unlink()
            for path in (root / 'audit').rglob('*_test.json'):
                path.unlink()
            extra = root / 'checkpoints' / 'GlobalMedian' / 'unregistered.bin'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'all checkpoint files'):
                collect.collect_pretest(root, write_output=False)

    def test_sealed_test_rejects_an_incomplete_metric_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _build_fixture(root, [42])
            audit_path = root / 'audit' / '42' / 'FF_test.json'
            rows = json.loads(audit_path.read_text(encoding='utf-8'))
            del rows[0]['midpoint_rmse']
            _write_json(audit_path, rows)

            with self.assertRaisesRegex(ValueError, 'midpoint_rmse'):
                collect.collect_sealed_test(root)


if __name__ == '__main__':
    unittest.main()
