#!/usr/bin/env python3
"""Run the clean graph-UQ pipeline end to end on a tiny synthetic PyG graph."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    load_role_targets,
    sha256_file,
    write_clean_uq_protocol,
    write_role_target_artifacts,
)
from credipred.experiments.gnn_experiments.clean_uq_baselines import (
    fit_global_median,
    fit_xgboost_with_selection,
    predict_xgboost,
    select_label_propagation,
    write_baseline_role_predictions,
)


TRAIN_MODULE = 'credipred.experiments.gnn_experiments.clean_uq_train'
AUDIT_MODULE = 'credipred.experiments.gnn_experiments.clean_uq_audit'


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def _make_inputs(root: Path, seed: int) -> tuple[Path, dict[str, Path], Path]:
    generator = torch.Generator().manual_seed(seed)
    node_count = 40
    x = torch.randn(node_count, 8, generator=generator)
    nodes = torch.arange(node_count)
    edge_index = torch.stack(
        [
            torch.cat((nodes, nodes, nodes)),
            torch.cat(((nodes + 1) % node_count, (nodes - 1) % node_count, (nodes + 7) % node_count)),
        ]
    )
    labels = torch.sigmoid(0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2])

    input_dir = root / 'inputs'
    input_dir.mkdir(parents=True)
    data_path = input_dir / 'data.pt'
    labels_path = input_dir / 'source-labels.pt'
    torch.save(Data(x=x, edge_index=edge_index), data_path)
    torch.save(labels, labels_path)

    rows = [
        {
            'canonical_domain': f'smoke-{node_id}.example',
            'group_id': f'domain:smoke-{node_id}.example',
            'node_id': node_id,
            'mapped': True,
            'labelled': True,
        }
        for node_id in range(node_count)
    ]
    protocol_dir = root / 'protocol'
    write_clean_uq_protocol(
        rows,
        protocol_dir,
        split_seed=2027,
        model_seeds=[seed],
        source_population_count=node_count,
    )
    partitions_path = protocol_dir / 'clean_uq_partitions.jsonl'
    target_paths = write_role_target_artifacts(
        labels_path,
        partitions_path,
        protocol_dir / 'targets',
        root / 'sealed' / 'targets',
        expected_labels_sha256=sha256_file(labels_path),
        num_nodes=node_count,
    )
    return data_path, target_paths, partitions_path


def _train_arguments(
    stage: str,
    root: Path,
    seed: int,
    device: str,
    data_path: Path,
    targets_path: Path,
    partitions_path: Path,
) -> list[str]:
    common = [
        '-m', TRAIN_MODULE, stage,
        '--run-dir', str(root),
        '--seed', str(seed),
        '--data-pt', str(data_path),
        '--batch-size', '8',
        '--neighbor-k', '2',
        '--num-workers', '0',
        '--device', device,
    ]
    if stage != 'parent-cache':
        common.extend(
            ['--targets-pt', str(targets_path), '--partition-jsonl', str(partitions_path)]
        )
    return common


def _role_nodes(partitions_path: Path, role: str) -> torch.Tensor:
    rows = [
        json.loads(line)
        for line in partitions_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    return torch.tensor(
        [row['node_id'] for row in rows if row['clean_uq_split'] == role],
        dtype=torch.long,
    )


def _smoke_baselines(
    root: Path,
    data_path: Path,
    target_paths: dict[str, Path],
    partitions_path: Path,
) -> None:
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    fit_select = load_role_targets(
        target_paths['fit_select'],
        partition_jsonl=partitions_path,
        expected_roles=('fit', 'select'),
        num_nodes=int(data.num_nodes),
    )
    fit_ids = _role_nodes(partitions_path, 'fit')
    select_ids = _role_nodes(partitions_path, 'select')
    median = fit_global_median(fit_select, fit_ids)
    xgboost = fit_xgboost_with_selection(
        data.x, fit_select, fit_ids, select_ids
    )
    label_propagation = select_label_propagation(
        data.edge_index,
        num_nodes=int(data.num_nodes),
        labels=fit_select,
        fit_node_ids=fit_ids,
        select_node_ids=select_ids,
    )
    xgboost_full = predict_xgboost(
        xgboost.midpoint_model,
        xgboost.quantile_model,
        data.x,
        torch.arange(int(data.num_nodes)),
    )
    full_predictions = {
        'GlobalMedian': torch.full((int(data.num_nodes),), median),
        'XGBoost': xgboost_full,
        'RegressionLP': label_propagation.predictions,
    }
    for role, target_name in (('calibrate', 'calibrate'), ('test', 'test')):
        targets = load_role_targets(
            target_paths[target_name],
            partition_jsonl=partitions_path,
            expected_roles=(role,),
            num_nodes=int(data.num_nodes),
        )
        node_ids = _role_nodes(partitions_path, role)
        for arm, predictions in full_predictions.items():
            path = root / 'predictions' / arm / 'deterministic' / f'{role}.pt'
            write_baseline_role_predictions(
                predictions,
                node_ids,
                targets,
                path,
                role=role,
                arm=arm,
            )
    for arm in full_predictions:
        prediction_dir = root / 'predictions' / arm / 'deterministic'
        _run(
            '-m', AUDIT_MODULE, 'calibrate',
            '--run-dir', str(root), '--seed', 'deterministic', '--arm', arm,
            '--predictions-pt', str(prediction_dir / 'calibrate.pt'),
        )
        _run(
            '-m', AUDIT_MODULE, 'test',
            '--run-dir', str(root), '--seed', 'deterministic', '--arm', arm,
            '--predictions-pt', str(prediction_dir / 'test.pt'),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f'smoke output already exists: {args.output_dir}')
    args.output_dir.mkdir(parents=True)

    data_path, target_paths, partitions_path = _make_inputs(
        args.output_dir, args.seed
    )

    _run(
        *_train_arguments(
            'base-fit', args.output_dir, args.seed, args.device,
            data_path, target_paths['fit_select'], partitions_path
        ),
        '--model', 'GAT',
        '--epochs', '2',
        '--hidden-channels', '16',
        '--embedding-dim', '8',
        '--num-layers', '2',
        '--dropout', '0.0',
        '--normalization', 'none',
    )
    _run(
        *_train_arguments(
            'parent-cache', args.output_dir, args.seed, args.device,
            data_path, target_paths['fit_select'], partitions_path
        )
    )
    for correction_arm in ('mlp', 'topology'):
        _run(
            *_train_arguments(
                'correction-fit',
                args.output_dir,
                args.seed,
                args.device,
                data_path,
                target_paths['fit_select'],
                partitions_path,
            ),
            '--arm', correction_arm,
            '--epochs', '2',
            '--hidden-channels', '8',
            '--num-layers', '2',
            '--dropout', '0.0',
            '--normalization', 'none',
        )

    for arm in ('GAT', 'GAT-mlp', 'GAT-topology'):
        for role in ('calibrate', 'test'):
            _run(
                *_train_arguments(
                    'predict-role',
                    args.output_dir,
                    args.seed,
                    args.device,
                    data_path,
                    target_paths[role],
                    partitions_path,
                ),
                '--arm', arm,
                '--role', role,
            )
        calibration_predictions = (
            args.output_dir / 'predictions' / arm / f'seed-{args.seed}' / 'calibrate.pt'
        )
        test_predictions = (
            args.output_dir / 'predictions' / arm / f'seed-{args.seed}' / 'test.pt'
        )
        _run(
            '-m', AUDIT_MODULE, 'calibrate',
            '--run-dir', str(args.output_dir),
            '--seed', str(args.seed),
            '--arm', arm,
            '--predictions-pt', str(calibration_predictions),
        )
        _run(
            '-m', AUDIT_MODULE, 'test',
            '--run-dir', str(args.output_dir),
            '--seed', str(args.seed),
            '--arm', arm,
            '--predictions-pt', str(test_predictions),
        )

    for role in ('calibrate', 'test'):
        _run(
            '-m', AUDIT_MODULE, 'ensemble',
            '--run-dir', str(args.output_dir),
            '--role', role,
            '--predictions-pt', str(
                args.output_dir / 'predictions' / 'GAT' / f'seed-{args.seed}' / f'{role}.pt'
            ),
        )
    ensemble_dir = args.output_dir / 'predictions' / 'GAT-ensemble'
    _run(
        '-m', AUDIT_MODULE, 'calibrate',
        '--run-dir', str(args.output_dir),
        '--seed', 'ensemble',
        '--arm', 'GAT-ensemble',
        '--predictions-pt', str(ensemble_dir / 'calibrate.pt'),
    )
    _run(
        '-m', AUDIT_MODULE, 'test',
        '--run-dir', str(args.output_dir),
        '--seed', 'ensemble',
        '--arm', 'GAT-ensemble',
        '--predictions-pt', str(ensemble_dir / 'test.pt'),
    )

    _smoke_baselines(
        args.output_dir, data_path, target_paths, partitions_path
    )

    seed_audit_dir = args.output_dir / 'audit' / str(args.seed)
    ensemble_audit_dir = args.output_dir / 'audit' / 'ensemble'
    audit_files = sorted(seed_audit_dir.glob('*_test.json'))
    audit_files.extend(sorted(ensemble_audit_dir.glob('*_test.json')))
    deterministic_audit_dir = args.output_dir / 'audit' / 'deterministic'
    audit_files.extend(sorted(deterministic_audit_dir.glob('*_test.json')))
    if len(audit_files) != 7:
        raise RuntimeError(f'expected seven audited arms, found {len(audit_files)}')
    print(
        json.dumps(
            {
                'status': 'passed',
                'seed': args.seed,
                'epochs': 2,
                'arms': [path.name.removesuffix('_test.json') for path in audit_files],
                'ensemble': 'provisional-single-seed',
            },
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
