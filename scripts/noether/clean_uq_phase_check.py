#!/usr/bin/env python3
"""Check the frozen seed identity before clean-UQ pretest or sealed-test submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from credipred.experiments.gnn_experiments.clean_uq_protocol import sha256_file


ARMS = ('FF', 'GCN', 'SAGE', 'GAT', 'GAT-mlp', 'GAT-topology')
BASELINE_ARMS = ('GlobalMedian', 'XGBoost', 'RegressionLP')
FROZEN_DQR_LABELS_SHA256 = (
    '287d828385dc9ca8f6e21bcb383f824320850c560dbbdc941bafdf2b46c1a623'
)
FULL_GRAPH_NODE_COUNT = 45_030_252
FROZEN_PARTITION_SHA256 = (
    'b896a886a4b8981043820e2326492f5ffd8c20c63006380f9df8f2544231b5d7'
)
FROZEN_PROTOCOL_SHA256 = (
    '52f17cadda18ec2284a31e06ade7a8a3e911e4af63efd8d5255810415fa24131'
)
FROZEN_MODEL_SEEDS = [42, 43, 44, 45, 46]
FROZEN_PARTITION_COUNTS = {
    'fit': 5_519,
    'select': 920,
    'calibrate': 1_380,
    'test': 1_380,
}


def _read_state(path: Path) -> dict[str, object]:
    state = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(state, dict):
        raise ValueError(f'invalid calibration state: {path}')
    return state


def _validate_target_ledger(
    path: Path, *, phase: str, protocol_json: Path, run_root: Path
) -> None:
    ledger = _read_state(path)
    if (
        ledger.get('schema_version') != 'clean-uq-target-ledger-v1'
        or ledger.get('source_labels_sha256') != FROZEN_DQR_LABELS_SHA256
        or ledger.get('num_nodes') != FULL_GRAPH_NODE_COUNT
        or ledger.get('partition_sha256') != FROZEN_PARTITION_SHA256
    ):
        raise ValueError('target ledger differs from the frozen DQR protocol')
    partition_path = protocol_json.parent / 'clean_uq_partitions.jsonl'
    if (
        sha256_file(protocol_json) != FROZEN_PROTOCOL_SHA256
        or sha256_file(partition_path) != FROZEN_PARTITION_SHA256
    ):
        raise ValueError('protocol files differ from the frozen WWW RQ1 split')
    artifacts = ledger.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('target ledger is missing role artifacts')
    expected = {
        'fit_select': (
            run_root / 'protocol' / 'targets' / 'fit_select.pt',
            ['fit', 'select'],
        ),
        'calibrate': (
            run_root / 'protocol' / 'targets' / 'calibrate.pt',
            ['calibrate'],
        ),
        'test': (run_root / 'sealed' / 'targets' / 'test.pt', ['test']),
    }
    allowed = ('fit_select', 'calibrate') if phase == 'pretest' else tuple(expected)
    for name, (expected_path, roles) in expected.items():
        record = artifacts.get(name)
        if (
            not isinstance(record, dict)
            or record.get('path') != str(expected_path.resolve())
            or record.get('authorized_roles') != roles
        ):
            raise ValueError(f'target ledger has an invalid {name} record')
        if name in allowed:
            if not expected_path.is_file() or record.get('sha256') != sha256_file(
                expected_path
            ):
                raise ValueError(f'{name} target artifact differs from its ledger')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pretest', 'sealed-test'))
    parser.add_argument('--protocol-json', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--target-ledger', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol_json.read_text(encoding='utf-8'))
    if (
        protocol.get('split_seed') != 2027
        or protocol.get('model_seeds') != FROZEN_MODEL_SEEDS
        or protocol.get('partition_counts') != FROZEN_PARTITION_COUNTS
        or args.seeds != FROZEN_MODEL_SEEDS
    ):
        raise ValueError('submission differs from the frozen WWW RQ1 protocol')
    _validate_target_ledger(
        args.target_ledger,
        phase=args.phase,
        protocol_json=args.protocol_json,
        run_root=args.run_root,
    )
    if args.phase == 'pretest':
        print(json.dumps({'phase': args.phase, 'seeds': args.seeds}, sort_keys=True))
        return

    for seed in args.seeds:
        for arm in ARMS:
            state_path = (
                args.run_root / 'audit' / str(seed) / f'{arm}_calibration.json'
            )
            state = _read_state(state_path)
            if str(state.get('seed')) != str(seed) or state.get('arm') != arm:
                raise ValueError(f'calibration state identity mismatch: {state_path}')
        for arm in ARMS:
            checkpoint_path = (
                args.run_root / 'checkpoints' / arm / f'seed-{seed}.pt'
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
        cache_path = args.run_root / 'cache' / 'GAT' / f'seed-{seed}.pt'
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)

    for arm in BASELINE_ARMS:
        baseline_state = _read_state(
            args.run_root / 'checkpoints' / arm / 'deterministic.json'
        )
        calibration_state = _read_state(
            args.run_root / 'audit' / 'deterministic' / f'{arm}_calibration.json'
        )
        if (
            baseline_state.get('arm') != arm
            or baseline_state.get('seed') != 'deterministic'
            or calibration_state.get('arm') != arm
            or calibration_state.get('seed') != 'deterministic'
            or baseline_state.get('methods') != calibration_state.get('methods')
        ):
            raise ValueError(f'baseline state identity mismatch: {arm}')
    lp_probe = _read_state(args.run_root / 'probe' / 'regression_lp.json')
    if lp_probe.get('passed') is not True or lp_probe.get('full_graph') is not True:
        raise ValueError('sealed test requires a passed full-graph LP probe')
    if not (
        args.run_root / 'cache' / 'RegressionLP' / 'deterministic.pt'
    ).is_file():
        raise FileNotFoundError('RegressionLP deterministic cache')

    pretest_ledger = _read_state(
        args.run_root / 'results' / 'clean_uq_pretest_ledger.json'
    )
    expected_pretest_artifacts = {
        'calibration_predictions': 34,
        'calibration_states': 34,
        'checkpoints': 30,
        'baseline_states': 3,
        'xgboost_models': 2,
        'parent_caches': 5,
        'lp_caches': 1,
        'lp_probes': 1,
    }
    if (
        pretest_ledger.get('schema_version') != 'clean-uq-pretest-ledger-v1'
        or pretest_ledger.get('phase') != 'pretest'
        or pretest_ledger.get('model_seeds') != FROZEN_MODEL_SEEDS
        or pretest_ledger.get('arms') != list(ARMS)
        or pretest_ledger.get('ensemble_arm') != 'GAT-ensemble'
        or pretest_ledger.get('baseline_arms') != list(BASELINE_ARMS)
        or pretest_ledger.get('expected_artifacts') != expected_pretest_artifacts
        or pretest_ledger.get('observed_artifacts') != expected_pretest_artifacts
        or pretest_ledger.get('sealed_test_artifacts_present') is not False
    ):
        raise ValueError('sealed test requires a passed pretest exposure ledger')

    ensemble_path = (
        args.run_root / 'predictions' / 'GAT-ensemble' / 'calibrate.pt'
    )
    ensemble = torch.load(ensemble_path, map_location='cpu', weights_only=True)
    if ensemble.get('member_seeds') != args.seeds:
        raise ValueError('calibration ensemble members differ from frozen model seeds')
    ensemble_state = _read_state(
        args.run_root / 'audit' / 'ensemble' / 'GAT-ensemble_calibration.json'
    )
    if ensemble_state.get('member_seeds') != args.seeds:
        raise ValueError('ensemble calibration state has different member seeds')
    print(json.dumps({'phase': args.phase, 'seeds': args.seeds}, sort_keys=True))


if __name__ == '__main__':
    main()
