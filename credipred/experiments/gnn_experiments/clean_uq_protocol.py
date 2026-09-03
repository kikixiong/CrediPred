"""Freeze the clean graph-UQ split protocol and population ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch


_PARTITIONS = ('fit', 'select', 'calibrate', 'test')
_TARGET_RATIOS = {'fit': 60, 'select': 10, 'calibrate': 15, 'test': 15}
_TARGET_SCHEMA = 'clean-uq-role-targets-v1'
_TARGET_SENTINEL = -1.0


def sha256_file(path: Path) -> str:
    """Hash one frozen input or generated artifact without loading it in memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_role_nodes(path: Path) -> dict[str, list[int]]:
    role_nodes: dict[str, list[int]] = {role: [] for role in _PARTITIONS}
    seen: set[int] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('clean-UQ partition rows must be JSON objects')
        role = row.get('clean_uq_split')
        if role not in _PARTITIONS:
            continue
        node_id = row.get('node_id')
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError('clean-UQ partition node IDs must be nonnegative integers')
        if row.get('mapped') is not True or row.get('labelled') is not True:
            raise ValueError(f'{role} contains an ineligible target row')
        if node_id in seen:
            raise ValueError('clean-UQ partition node IDs must be unique')
        seen.add(node_id)
        role_nodes[role].append(node_id)
    if any(not role_nodes[role] for role in _PARTITIONS):
        raise ValueError('fit/select/calibrate/test target roles must all be nonempty')
    return role_nodes


def _target_payload(
    full_labels: torch.Tensor,
    role_nodes: dict[str, list[int]],
    roles: tuple[str, ...],
    *,
    partition_sha256: str,
    source_labels_sha256: str,
) -> dict[str, Any]:
    labels = torch.full_like(full_labels, _TARGET_SENTINEL, dtype=torch.float32)
    authorized = torch.tensor(
        sorted(node_id for role in roles for node_id in role_nodes[role]),
        dtype=torch.long,
    )
    labels[authorized] = full_labels[authorized].float()
    return {
        'schema_version': _TARGET_SCHEMA,
        'authorized_roles': list(roles),
        'num_nodes': int(full_labels.numel()),
        'partition_sha256': partition_sha256,
        'source_labels_sha256': source_labels_sha256,
        'sentinel': _TARGET_SENTINEL,
        'labels': labels,
    }


def write_role_target_artifacts(
    source_labels_path: Path,
    partition_jsonl: Path,
    pretest_output_dir: Path,
    sealed_output_dir: Path,
    *,
    expected_labels_sha256: str,
    num_nodes: int,
) -> dict[str, Path]:
    """Materialize role-masked pretest targets and a separate sealed-test target."""
    actual_labels_sha256 = sha256_file(source_labels_path)
    if actual_labels_sha256 != expected_labels_sha256:
        raise ValueError('source labels SHA-256 differs from the frozen anchor')
    full_labels = torch.load(
        source_labels_path, map_location='cpu', weights_only=True, mmap=True
    )
    if not isinstance(full_labels, torch.Tensor) or full_labels.ndim != 1:
        raise ValueError('source labels must be a one-dimensional tensor')
    if full_labels.numel() != num_nodes:
        raise ValueError('source labels length differs from the full graph node count')
    role_nodes = _partition_role_nodes(partition_jsonl)
    expected_nodes = torch.tensor(
        sorted(node_id for nodes in role_nodes.values() for node_id in nodes),
        dtype=torch.long,
    )
    if int(expected_nodes.max()) >= num_nodes:
        raise ValueError('partition node ID exceeds the full labels tensor')
    exposed_nodes = torch.nonzero(full_labels >= 0.0).flatten()
    if not torch.equal(exposed_nodes, expected_nodes):
        raise ValueError('source labels do not expose exactly the frozen population')
    population_targets = full_labels[expected_nodes].float()
    if not bool(torch.isfinite(population_targets).all()) or bool(
        (population_targets > 1.0).any()
    ):
        raise ValueError('frozen targets must be finite values in [0, 1]')
    unavailable = full_labels < 0.0
    if not bool((full_labels[unavailable] == _TARGET_SENTINEL).all()):
        raise ValueError('unavailable source labels must use the -1 sentinel')

    paths = {
        'fit_select': pretest_output_dir / 'fit_select.pt',
        'calibrate': pretest_output_dir / 'calibrate.pt',
        'test': sealed_output_dir / 'test.pt',
        'ledger': pretest_output_dir / 'clean_uq_target_ledger.json',
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('refusing to overwrite clean-UQ target artifacts')
    pretest_output_dir.mkdir(parents=True, exist_ok=True)
    sealed_output_dir.mkdir(parents=True, exist_ok=True)
    partition_sha256 = sha256_file(partition_jsonl)
    role_sets = {
        'fit_select': ('fit', 'select'),
        'calibrate': ('calibrate',),
        'test': ('test',),
    }
    for name, roles in role_sets.items():
        torch.save(
            _target_payload(
                full_labels,
                role_nodes,
                roles,
                partition_sha256=partition_sha256,
                source_labels_sha256=actual_labels_sha256,
            ),
            paths[name],
        )
    ledger = {
        'schema_version': 'clean-uq-target-ledger-v1',
        'source_labels_path': str(source_labels_path.resolve()),
        'source_labels_sha256': actual_labels_sha256,
        'partition_path': str(partition_jsonl.resolve()),
        'partition_sha256': partition_sha256,
        'num_nodes': num_nodes,
        'artifacts': {
            name: {
                'path': str(paths[name].resolve()),
                'authorized_roles': list(role_sets[name]),
                'sha256': sha256_file(paths[name]),
            }
            for name in role_sets
        },
    }
    paths['ledger'].write_text(
        json.dumps(ledger, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    return paths


def load_role_targets(
    path: Path,
    *,
    partition_jsonl: Path,
    expected_roles: tuple[str, ...],
    num_nodes: int,
) -> torch.Tensor:
    """Load exactly the roles authorized for one stage and reject overexposure."""
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != _TARGET_SCHEMA:
        raise ValueError('not a clean-UQ role-target artifact')
    if payload.get('authorized_roles') != list(expected_roles):
        raise ValueError('role-target artifact authorizes different roles')
    if payload.get('num_nodes') != num_nodes or payload.get('sentinel') != _TARGET_SENTINEL:
        raise ValueError('role-target artifact has incompatible graph metadata')
    if payload.get('partition_sha256') != sha256_file(partition_jsonl):
        raise ValueError('role-target artifact belongs to a different partition')
    labels = payload.get('labels')
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError('role-target labels must be a one-dimensional tensor')
    if labels.numel() != num_nodes:
        raise ValueError('role-target labels length differs from the full graph')
    role_nodes = _partition_role_nodes(partition_jsonl)
    expected_nodes = torch.tensor(
        sorted(node_id for role in expected_roles for node_id in role_nodes[role]),
        dtype=torch.long,
    )
    exposed_nodes = torch.nonzero(labels != _TARGET_SENTINEL).flatten()
    if not torch.equal(exposed_nodes, expected_nodes):
        raise ValueError('role-target artifact exposed unauthorized targets')
    targets = labels[expected_nodes].float()
    if not bool(torch.isfinite(targets).all()) or bool(
        ((targets < 0.0) | (targets > 1.0)).any()
    ):
        raise ValueError('authorized role targets must be finite values in [0, 1]')
    return labels.float()


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
    population_node_ids = {cast(int, row['node_id']) for row in population}
    if population_domains & {str(row['canonical_domain']) for row in sealed_test}:
        raise ValueError('sealed test cohort must not reuse a population canonical_domain')
    if population_node_ids & {cast(int, row['node_id']) for row in sealed_test}:
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
