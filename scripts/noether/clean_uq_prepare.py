#!/usr/bin/env python3
"""Freeze the production clean-UQ protocol from a mapped DQR registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    write_clean_uq_protocol,
)


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
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sealed-test-jsonl', type=Path)
    parser.add_argument('--source-population-count', type=int, default=11_520)
    parser.add_argument('--expected-observed-count', type=int, default=9_199)
    parser.add_argument('--split-seed', type=int, default=2027)
    parser.add_argument(
        '--model-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46]
    )
    args = parser.parse_args()

    population = _read_jsonl(args.registry_jsonl)
    if len(population) != args.expected_observed_count:
        raise ValueError(
            f'expected {args.expected_observed_count} registry rows, found {len(population)}'
        )
    sealed = _read_jsonl(args.sealed_test_jsonl) if args.sealed_test_jsonl else None
    protocol = write_clean_uq_protocol(
        population,
        args.output_dir,
        split_seed=args.split_seed,
        model_seeds=args.model_seeds,
        sealed_test_rows=sealed,
        source_population_count=args.source_population_count,
    )
    print(json.dumps(protocol, sort_keys=True))


if __name__ == '__main__':
    main()
