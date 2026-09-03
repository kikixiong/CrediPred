"""Freeze the domain-group split required before honest mu-star training."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_PARTITIONS = ('base_fit', 'head_fit', 'head_tune', 'audit')


def _canonical_registry_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate the minimal official registry contract needed for a frozen split."""
    result: list[dict[str, object]] = []
    seen_domains: set[str] = set()
    seen_nodes: set[int] = set()
    for row in rows:
        raw_domain = row.get('canonical_domain')
        raw_group = row.get('group_id')
        node_id = row.get('node_id')
        if not isinstance(raw_domain, str) or not raw_domain.strip():
            raise ValueError('registry rows require nonblank canonical_domain')
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError('registry rows require nonnegative integer node_id')
        if not isinstance(raw_group, str) or not raw_group.strip():
            raise ValueError('registry rows require nonblank group_id')
        domain = raw_domain.strip().lower()
        if domain in seen_domains or node_id in seen_nodes:
            raise ValueError('registry domains and node IDs must each be unique')
        seen_domains.add(domain)
        seen_nodes.add(node_id)
        result.append(
            {
                'canonical_domain': domain,
                'group_id': raw_group.strip(),
                'node_id': node_id,
            }
        )
    if not result:
        raise ValueError('registry must contain at least one domain')
    return result


def _split_rank(domain: str, seed: int) -> str:
    return hashlib.sha256(f'mu-star-base-fit-v1:{seed}:{domain}'.encode()).hexdigest()


def assign_mu_star_partitions(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    """Assign canonical domain groups to deterministic 40/35/10/15 partitions."""
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    registry = _canonical_registry_rows(rows)
    groups = sorted(
        {str(row['group_id']) for row in registry},
        key=lambda group: _split_rank(group, seed),
    )
    count = len(groups)
    base_end = count * 40 // 100
    head_fit_end = count * 75 // 100
    head_tune_end = count * 85 // 100
    group_partition: dict[str, str] = {}
    for index, group in enumerate(groups):
        if index < base_end:
            partition = 'base_fit'
        elif index < head_fit_end:
            partition = 'head_fit'
        elif index < head_tune_end:
            partition = 'head_tune'
        else:
            partition = 'audit'
        group_partition[group] = partition
    assignments = [
        {**row, 'quality_split': group_partition[str(row['group_id'])]}
        for row in registry
    ]
    return sorted(assignments, key=lambda row: str(row['canonical_domain']))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'registry JSONL row {line_number} must be an object')
        rows.append(row)
    return rows


def write_mu_star_plan(
    registry_path: Path,
    output_dir: Path,
    *,
    seed: int,
    data_path: Path | None = None,
    mapping_path: Path | None = None,
    expected_data_sha256: str | None = None,
    expected_mapping_sha256: str | None = None,
    labels_path: Path | None = None,
    expected_labels_sha256: str | None = None,
) -> dict[str, object]:
    """Write an immutable split/lineage plan without loading graph training data."""
    if not registry_path.is_file():
        raise ValueError(f'official registry is missing: {registry_path}')
    if output_dir.exists():
        raise ValueError(f'output directory already exists: {output_dir}')
    assignments = assign_mu_star_partitions(_read_jsonl(registry_path), seed=seed)
    output_dir.mkdir(parents=True)
    assignment_path = output_dir / 'mu_star_partitions.jsonl'
    assignment_path.write_text(
        ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in assignments),
        encoding='utf-8',
    )
    graph_inputs = (
        data_path,
        mapping_path,
        expected_data_sha256,
        expected_mapping_sha256,
        labels_path,
        expected_labels_sha256,
    )
    if any(value is not None for value in graph_inputs) and not all(
        value is not None for value in graph_inputs
    ):
        raise ValueError('graph approval requires both paths and both reviewed SHA-256 values')
    graph_approved = all(value is not None for value in graph_inputs)
    if graph_approved:
        assert data_path is not None and mapping_path is not None
        if not data_path.is_file() or _sha256(data_path) != expected_data_sha256:
            raise ValueError('data.pt does not match its reviewed SHA-256')
        if not mapping_path.is_file() or _sha256(mapping_path) != expected_mapping_sha256:
            raise ValueError('mapping.pt does not match its reviewed SHA-256')
    plan = {
        'schema_version': 'mu-star-base-fit-v1',
        'seed': seed,
        'partition_rule': 'sha256(mu-star-base-fit-v1:seed:group_id)',
        'partition_counts': {
            partition: len(
                {
                    row['group_id']
                    for row in assignments
                    if row['quality_split'] == partition
                }
            )
            for partition in _PARTITIONS
        },
        'training_partitions': ['base_fit'],
        'excluded_partitions': ['head_fit', 'head_tune', 'audit'],
        'registry_sha256': _sha256(registry_path),
        'partition_sha256': _sha256(assignment_path),
        'official_group_registry_sha256': _sha256(registry_path),
        'quality_partition_registry_sha256': _sha256(assignment_path),
        'data_sha256': expected_data_sha256 if graph_approved else None,
        'mapping_sha256': expected_mapping_sha256 if graph_approved else None,
        'labels_sha256': expected_labels_sha256 if graph_approved else None,
        'training_status': (
            'approved_verified_graph_and_base_fit_trainer'
            if graph_approved
            else 'blocked_pending_verified_graph_data_and_trainer_contract'
        ),
    }
    (output_dir / 'mu_star_lineage.json').write_text(
        json.dumps(plan, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8'
    )
    return plan


def main() -> None:
    """Freeze the only registry split eligible for a future mu-star base fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-jsonl', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-pt', type=Path)
    parser.add_argument('--mapping-pt', type=Path)
    parser.add_argument('--expected-data-sha256')
    parser.add_argument('--expected-mapping-sha256')
    parser.add_argument('--labels-pt', type=Path)
    parser.add_argument('--expected-labels-sha256')
    args = parser.parse_args()
    plan = write_mu_star_plan(
        args.registry_jsonl,
        args.output_dir,
        seed=args.seed,
        data_path=args.data_pt,
        mapping_path=args.mapping_pt,
        expected_data_sha256=args.expected_data_sha256,
        expected_mapping_sha256=args.expected_mapping_sha256,
        labels_path=args.labels_pt,
        expected_labels_sha256=args.expected_labels_sha256,
    )
    print(json.dumps(plan, sort_keys=True))


if __name__ == '__main__':
    main()
