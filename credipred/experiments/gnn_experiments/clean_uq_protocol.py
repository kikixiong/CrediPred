"""Freeze the clean graph-UQ split protocol and population ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


_PARTITIONS = ('fit', 'select', 'calibrate', 'test')
_TARGET_RATIOS = {'fit': 60, 'select': 10, 'calibrate': 15, 'test': 15}


def _canonical_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate and normalize rows used by the clean UQ protocol."""
    normalized: list[dict[str, object]] = []
    domains: set[str] = set()
    node_ids: set[int] = set()
    for row in rows:
        raw_domain = row.get('canonical_domain')
        raw_group = row.get('group_id')
        node_id = row.get('node_id')
        if not isinstance(raw_domain, str) or not raw_domain.strip():
            raise ValueError('protocol rows require nonblank canonical_domain')
        if not isinstance(raw_group, str) or not raw_group.strip():
            raise ValueError('protocol rows require nonblank group_id')
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError('protocol rows require nonnegative integer node_id')
        domain = raw_domain.strip().lower()
        if domain in domains or node_id in node_ids:
            raise ValueError('protocol domains and node IDs must each be unique')
        domains.add(domain)
        node_ids.add(node_id)
        normalized.append(
            {
                'canonical_domain': domain,
                'group_id': raw_group.strip(),
                'node_id': node_id,
                'mapped': row.get('mapped', True) is True,
                'labelled': row.get('labelled', True) is True,
            }
        )
    return normalized


def _split_rank(group_id: str, split_seed: int) -> str:
    return hashlib.sha256(
        f'clean-uq-protocol-v1:{split_seed}:{group_id}'.encode()
    ).hexdigest()


def assign_clean_uq_partitions(
    rows: Sequence[Mapping[str, object]], *, split_seed: int
) -> list[dict[str, object]]:
    """Assign every supplied group once to deterministic 60/10/15/15 partitions."""
    if split_seed < 0:
        raise ValueError('split_seed must be nonnegative')
    registry = _canonical_rows(rows)
    if not registry:
        raise ValueError('split protocol requires at least one row')
    groups = sorted(
        {str(row['group_id']) for row in registry},
        key=lambda group_id: _split_rank(group_id, split_seed),
    )
    group_count = len(groups)
    fit_end = group_count * 60 // 100
    select_end = group_count * 70 // 100
    calibrate_end = group_count * 85 // 100
    group_partition = {
        group_id: (
            'fit'
            if index < fit_end
            else 'select'
            if index < select_end
            else 'calibrate'
            if index < calibrate_end
            else 'test'
        )
        for index, group_id in enumerate(groups)
    }
    return sorted(
        [
            {**row, 'clean_uq_split': group_partition[str(row['group_id'])]}
            for row in registry
        ],
        key=lambda row: str(row['canonical_domain']),
    )


def _eligible_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row['mapped'] and row['labelled']]


def _population_ledger(
    population: Sequence[dict[str, object]], *, source_population_count: int, sealed_test: Sequence[dict[str, object]]
) -> dict[str, int]:
    """Describe the full population funnel and all excluded records."""
    if source_population_count < len(population):
        raise ValueError('source_population_count cannot be smaller than observed population')
    mapped = [row for row in population if row['mapped']]
    mapped_labelled = _eligible_rows(population)
    return {
        'source_population_count': source_population_count,
        'observed_population_count': len(population),
        'excluded_before_observation_count': source_population_count - len(population),
        'mapped_count': len(mapped),
        'mapped_labelled_count': len(mapped_labelled),
        'excluded_unmapped_count': len(population) - len(mapped),
        'excluded_unlabelled_count': len(mapped) - len(mapped_labelled),
        'sealed_test_count': len(sealed_test),
        'sealed_test_group_count': len({str(row['group_id']) for row in sealed_test}),
    }


def write_clean_uq_protocol(
    population_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    split_seed: int,
    model_seeds: Sequence[int],
    sealed_test_rows: Sequence[Mapping[str, object]] | None = None,
    source_population_count: int | None = None,
) -> dict[str, object]:
    """Write a compact protocol, ledger, and group-preserving assignments.

    ``sealed_test_rows`` is an explicitly supplied unused mapped/labelled cohort.
    When present it is used as the sealed test cohort; the frozen four-way
    assignment remains available in the output for complete population audit.
    """
    if output_dir.exists():
        raise ValueError(f'output directory already exists: {output_dir}')
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in model_seeds):
        raise ValueError('model_seeds must contain integers')

    population = _canonical_rows(population_rows)
    eligible = _eligible_rows(population)
    if not eligible:
        raise ValueError('protocol requires at least one mapped and labelled row')
    sealed_test = _canonical_rows(sealed_test_rows or [])
    if any(not row['mapped'] or not row['labelled'] for row in sealed_test):
        raise ValueError('sealed test cohort must be mapped and labelled')
    population_domains = {str(row['canonical_domain']) for row in population}
    population_node_ids = {int(row['node_id']) for row in population}
    if population_domains & {str(row['canonical_domain']) for row in sealed_test}:
        raise ValueError('sealed test cohort must not reuse a population canonical_domain')
    if population_node_ids & {int(row['node_id']) for row in sealed_test}:
        raise ValueError('sealed test cohort must not reuse a population node_id')
    population_groups = {str(row['group_id']) for row in eligible}
    sealed_groups = {str(row['group_id']) for row in sealed_test}
    if population_groups & sealed_groups:
        raise ValueError('sealed test cohort must be group-disjoint from protocol population')

    assignments = assign_clean_uq_partitions(eligible, split_seed=split_seed)
    if sealed_test:
        assignments = [
            {
                **row,
                'clean_uq_split': (
                    'audit_only_frozen_test'
                    if row['clean_uq_split'] == 'test'
                    else row['clean_uq_split']
                ),
            }
            for row in assignments
        ]
        assignments.extend(
            {**row, 'clean_uq_split': 'test'} for row in sealed_test
        )
    assignments = sorted(assignments, key=lambda row: str(row['canonical_domain']))
    source_count = len(population) if source_population_count is None else source_population_count
    ledger = _population_ledger(
        population, source_population_count=source_count, sealed_test=sealed_test
    )
    protocol: dict[str, object] = {
        'schema_version': 'clean-uq-protocol-v1',
        'split_seed': split_seed,
        'model_seeds': list(model_seeds),
        'partition_rule': 'sha256(clean-uq-protocol-v1:split_seed:group_id)',
        'target_group_ratios': _TARGET_RATIOS,
        'group_disjoint': True,
        'test_source': (
            'sealed_unused_mapped_labelled_cohort' if sealed_test else 'frozen_four_way_split'
        ),
        'partition_counts': {
            partition: len(
                {row['group_id'] for row in assignments if row['clean_uq_split'] == partition}
            )
            for partition in _PARTITIONS
        },
        'audit_only_frozen_test_group_count': len(
            {
                row['group_id']
                for row in assignments
                if row['clean_uq_split'] == 'audit_only_frozen_test'
            }
        ),
    }

    output_dir.mkdir(parents=True)
    (output_dir / 'clean_uq_partitions.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in assignments),
        encoding='utf-8',
    )
    (output_dir / 'clean_uq_protocol.json').write_text(
        json.dumps(protocol, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8'
    )
    (output_dir / 'clean_uq_population_ledger.json').write_text(
        json.dumps(ledger, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8'
    )
    return protocol
