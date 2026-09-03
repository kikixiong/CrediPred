#!/usr/bin/env python3
"""Freeze the production clean-UQ protocol from a mapped DQR registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    sha256_file,
    write_clean_uq_protocol,
    write_role_target_artifacts,
)


FROZEN_DQR_LABELS_SHA256 = (
    '287d828385dc9ca8f6e21bcb383f824320850c560dbbdc941bafdf2b46c1a623'
)
FROZEN_DQR_REGISTRY_SHA256 = (
    'b6ea754aa1f27fe67ffe7fa7738811ce22af641c3ad23e4ec74634a8c74b4931'
)
FROZEN_PARTITION_SHA256 = (
    'b896a886a4b8981043820e2326492f5ffd8c20c63006380f9df8f2544231b5d7'
)
FROZEN_PROTOCOL_SHA256 = (
    '52f17cadda18ec2284a31e06ade7a8a3e911e4af63efd8d5255810415fa24131'
)
FULL_GRAPH_NODE_COUNT = 45_030_252
FROZEN_OBSERVED_COUNT = 9_199
FROZEN_SOURCE_POPULATION_COUNT = 11_520
FROZEN_SPLIT_SEED = 2027
FROZEN_MODEL_SEEDS = (42, 43, 44, 45, 46)
FROZEN_PARTITION_COUNTS = {
    'fit': 5_519,
    'select': 920,
    'calibrate': 1_380,
    'test': 1_380,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path} must contain one JSON object per line')
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-jsonl', type=Path, required=True)
    parser.add_argument('--labels-pt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sealed-target-output-dir', type=Path)
    args = parser.parse_args()

    if sha256_file(args.registry_jsonl) != FROZEN_DQR_REGISTRY_SHA256:
        raise ValueError('registry differs from the frozen WWW RQ1 population')
    population = _read_jsonl(args.registry_jsonl)
    if len(population) != FROZEN_OBSERVED_COUNT:
        raise ValueError(
            f'expected {FROZEN_OBSERVED_COUNT} registry rows, found {len(population)}'
        )
    protocol = write_clean_uq_protocol(
        population,
        args.output_dir,
        split_seed=FROZEN_SPLIT_SEED,
        model_seeds=FROZEN_MODEL_SEEDS,
        source_population_count=FROZEN_SOURCE_POPULATION_COUNT,
    )
    partition_path = args.output_dir / 'clean_uq_partitions.jsonl'
    protocol_path = args.output_dir / 'clean_uq_protocol.json'
    if (
        protocol.get('partition_counts') != FROZEN_PARTITION_COUNTS
        or sha256_file(partition_path) != FROZEN_PARTITION_SHA256
        or sha256_file(protocol_path) != FROZEN_PROTOCOL_SHA256
    ):
        raise ValueError('generated protocol differs from the frozen WWW RQ1 split')
    target_paths = write_role_target_artifacts(
        args.labels_pt,
        partition_path,
        args.output_dir / 'targets',
        args.sealed_target_output_dir or args.output_dir.parent / 'sealed' / 'targets',
        expected_labels_sha256=FROZEN_DQR_LABELS_SHA256,
        num_nodes=FULL_GRAPH_NODE_COUNT,
    )
    print(
        json.dumps(
            {
                **protocol,
                'target_artifacts': {
                    name: str(path) for name, path in target_paths.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
