"""Behavior tests for the clean graph-UQ split protocol and population ledger."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    assign_clean_uq_partitions,
    write_clean_uq_protocol,
)


def _rows(count: int, *, group_prefix: str = 'group') -> list[dict[str, object]]:
    return [
        {
            'canonical_domain': f'domain-{index}.example',
            'group_id': f'{group_prefix}-{index}',
            'node_id': index,
            'mapped': True,
            'labelled': True,
        }
        for index in range(count)
    ]


def test_partition_assignment_is_deterministic_group_disjoint_and_once_only() -> None:
    rows = _rows(20) + [
        {
            'canonical_domain': 'alias.example',
            'group_id': 'group-0',
            'node_id': 20,
            'mapped': True,
            'labelled': True,
        }
    ]

    first = assign_clean_uq_partitions(rows, split_seed=17)
    second = assign_clean_uq_partitions(list(reversed(rows)), split_seed=17)

    assert first == second
    assert len(first) == len(rows)
    assert {row['clean_uq_split'] for row in first} == {
        'fit',
        'select',
        'calibrate',
        'test',
    }
    partition_groups = {
        partition: {row['group_id'] for row in first if row['clean_uq_split'] == partition}
        for partition in {'fit', 'select', 'calibrate', 'test'}
    }
    assert {partition: len(groups) for partition, groups in partition_groups.items()} == {
        'fit': 12,
        'select': 2,
        'calibrate': 3,
        'test': 3,
    }
    for left, right in combinations(partition_groups.values(), 2):
        assert left.isdisjoint(right)
    assert len({row['clean_uq_split'] for row in first if row['group_id'] == 'group-0'}) == 1
    assert {row['canonical_domain'] for row in first} == {
        row['canonical_domain'] for row in rows
    }


def test_protocol_writes_population_funnel_exclusions_and_sealed_test(tmp_path: Path) -> None:
    population = _rows(20)
    population.extend(
        [
            {
                'canonical_domain': 'unmapped.example',
                'group_id': 'unmapped-group',
                'node_id': 20,
                'mapped': False,
                'labelled': True,
            },
            {
                'canonical_domain': 'unlabelled.example',
                'group_id': 'unlabelled-group',
                'node_id': 21,
                'mapped': True,
                'labelled': False,
            },
        ]
    )
    sealed_test = _rows(2, group_prefix='sealed')
    for offset, row in enumerate(sealed_test, start=22):
        row['canonical_domain'] = f'sealed-{offset}.example'
        row['node_id'] = offset

    protocol = write_clean_uq_protocol(
        population,
        tmp_path / 'protocol',
        split_seed=17,
        model_seeds=[101, 202],
        sealed_test_rows=sealed_test,
        source_population_count=11_520,
    )

    output_dir = tmp_path / 'protocol'
    ledger = json.loads((output_dir / 'clean_uq_population_ledger.json').read_text())
    assignments = [
        json.loads(line)
        for line in (output_dir / 'clean_uq_partitions.jsonl').read_text().splitlines()
    ]

    assert protocol['split_seed'] == 17
    assert protocol['model_seeds'] == [101, 202]
    assert protocol['test_source'] == 'sealed_unused_mapped_labelled_cohort'
    assert ledger == {
        'source_population_count': 11_520,
        'observed_population_count': 22,
        'excluded_before_observation_count': 11_498,
        'mapped_count': 21,
        'mapped_labelled_count': 20,
        'excluded_unmapped_count': 1,
        'excluded_unlabelled_count': 1,
        'sealed_test_count': 2,
        'sealed_test_group_count': 2,
    }
    assert {row['clean_uq_split'] for row in assignments if row['group_id'].startswith('sealed-')} == {
        'test'
    }
    assert {
        row['group_id']
        for row in assignments
        if row['clean_uq_split'] == 'test'
    } == {'sealed-0', 'sealed-1'}
    assert sum(row['clean_uq_split'] == 'audit_only_frozen_test' for row in assignments) == 3
    output_partition_groups = {
        partition: {
            row['group_id'] for row in assignments if row['clean_uq_split'] == partition
        }
        for partition in {row['clean_uq_split'] for row in assignments}
    }
    for left, right in combinations(output_partition_groups.values(), 2):
        assert left.isdisjoint(right)
    assert (output_dir / 'clean_uq_protocol.json').is_file()


@pytest.mark.parametrize(
    ('sealed_row', 'error'),
    [
        (
            {
                'canonical_domain': 'domain-0.example',
                'group_id': 'sealed-domain',
                'node_id': 99,
                'mapped': True,
                'labelled': True,
            },
            'canonical_domain',
        ),
        (
            {
                'canonical_domain': 'sealed-node.example',
                'group_id': 'sealed-node',
                'node_id': 0,
                'mapped': True,
                'labelled': True,
            },
            'node_id',
        ),
    ],
)
def test_protocol_rejects_domain_or_node_reused_by_sealed_cohort(
    tmp_path: Path, sealed_row: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        write_clean_uq_protocol(
            _rows(4),
            tmp_path / 'protocol',
            split_seed=17,
            model_seeds=[101],
            sealed_test_rows=[sealed_row],
        )
