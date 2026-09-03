#!/usr/bin/env python3
"""Validate clean-UQ phase boundaries and collect unaggregated sealed metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from credipred.experiments.gnn_experiments.clean_uq_baselines import (
    FULL_GRAPH_EDGE_COUNT,
    FULL_GRAPH_NODE_COUNT,
    LP_ESTIMAND,
    LP_GRAPH_PASSES,
)


ARMS = ('FF', 'GCN', 'SAGE', 'GAT', 'GAT-mlp', 'GAT-topology')
ENSEMBLE_ARM = 'GAT-ensemble'
BASELINE_ARMS = ('GlobalMedian', 'XGBoost', 'RegressionLP')
DETERMINISTIC_RUN_ID = 'deterministic'
BASELINE_METHODS = {
    'GlobalMedian': ('simple',),
    'XGBoost': ('simple', 'cqr'),
    'RegressionLP': ('simple',),
}
ROLES = ('fit', 'select', 'calibrate', 'test')


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path.name} must contain JSON objects')
            rows.append(row)
    return rows


def _load_protocol(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    protocol_dir = run_root / 'protocol'
    protocol = _read_json(protocol_dir / 'clean_uq_protocol.json')
    partitions = _read_jsonl(protocol_dir / 'clean_uq_partitions.jsonl')
    population = _read_json(protocol_dir / 'clean_uq_population_ledger.json')
    if not isinstance(protocol, dict) or protocol.get('schema_version') != 'clean-uq-protocol-v1':
        raise ValueError('not a clean-UQ protocol')
    seeds = protocol.get('model_seeds')
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError('protocol model_seeds must be unique integers')
    if not isinstance(population, dict):
        raise ValueError('invalid clean-UQ population ledger')
    funnel_keys = (
        'source_population_count',
        'observed_population_count',
        'excluded_before_observation_count',
        'mapped_count',
        'mapped_labelled_count',
        'excluded_unmapped_count',
        'excluded_unlabelled_count',
        'sealed_test_count',
        'sealed_test_group_count',
    )
    funnel: dict[str, int] = {}
    for key in funnel_keys:
        value = population.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'population ledger has invalid {key}')
        funnel[key] = value
    return protocol, partitions, funnel


def _partition_surface(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, int]], dict[str, set[int]], dict[str, set[str]]]:
    node_sets: dict[str, set[int]] = {role: set() for role in ROLES}
    domain_sets: dict[str, set[str]] = {role: set() for role in ROLES}
    group_sets: dict[str, set[str]] = {role: set() for role in ROLES}
    for row in rows:
        role = row.get('clean_uq_split')
        if role not in ROLES:
            continue
        node_id = row.get('node_id')
        domain = row.get('canonical_domain')
        group = row.get('group_id')
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError('partition rows require nonnegative integer node_id')
        if not isinstance(domain, str) or not domain or not isinstance(group, str) or not group:
            raise ValueError('partition rows require domain and group identities')
        if node_id in node_sets[role] or domain in domain_sets[role]:
            raise ValueError(f'duplicate node or domain in {role} partition')
        node_sets[role].add(node_id)
        domain_sets[role].add(domain)
        group_sets[role].add(group)
    if any(not node_sets[role] for role in ROLES):
        raise ValueError('fit/select/calibrate/test partitions must all be nonempty')
    if any(node_sets[left] & node_sets[right] for index, left in enumerate(ROLES) for right in ROLES[index + 1 :]):
        raise ValueError('partition node IDs are not disjoint')
    if any(domain_sets[left] & domain_sets[right] for index, left in enumerate(ROLES) for right in ROLES[index + 1 :]):
        raise ValueError('partition canonical domains are not disjoint')
    if any(group_sets[left] & group_sets[right] for index, left in enumerate(ROLES) for right in ROLES[index + 1 :]):
        raise ValueError('partition groups are not disjoint')
    frozen_counts = protocol.get('partition_counts')
    if not isinstance(frozen_counts, dict):
        raise ValueError('protocol is missing partition_counts')
    for role in ROLES:
        if frozen_counts.get(role) != len(group_sets[role]):
            raise ValueError(f'protocol partition_counts mismatch for {role}')
    counts = {
        role: {
            'groups': len(group_sets[role]),
            'domains': len(domain_sets[role]),
            'nodes': len(node_sets[role]),
        }
        for role in ROLES
    }
    return counts, node_sets, domain_sets


def _prediction_path(run_root: Path, arm: str, seed: int | str, role: str) -> Path:
    if arm == ENSEMBLE_ARM:
        return run_root / 'predictions' / arm / f'{role}.pt'
    if seed == DETERMINISTIC_RUN_ID:
        return run_root / 'predictions' / arm / DETERMINISTIC_RUN_ID / f'{role}.pt'
    return run_root / 'predictions' / arm / f'seed-{seed}' / f'{role}.pt'


def _state_path(run_root: Path, arm: str, seed: int | str) -> Path:
    return run_root / 'audit' / str(seed) / f'{arm}_calibration.json'


def _audit_path(run_root: Path, arm: str, seed: int | str) -> Path:
    return run_root / 'audit' / str(seed) / f'{arm}_test.json'


def _load_prediction(
    path: Path,
    *,
    role: str,
    seed: int | str,
    arm: str,
    expected_nodes: set[int],
    methods: tuple[str, ...],
    member_seeds: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 'clean-uq-predictions-v1':
        raise ValueError(f'not a clean-UQ prediction artifact: {path.name}')
    if payload.get('role') != role or payload.get('seed') != seed or payload.get('arm') != arm:
        raise ValueError(f'prediction identity mismatch for {arm}/{seed}/{role}')
    if member_seeds is not None and payload.get('member_seeds') != member_seeds:
        raise ValueError(f'ensemble members mismatch for {role}')
    predictions = payload.get('predictions')
    labels = payload.get('labels')
    node_ids = payload.get('node_ids')
    expected_columns = 3 if methods == ('simple', 'cqr') else 1
    if (
        not isinstance(predictions, torch.Tensor)
        or predictions.ndim != 2
        or predictions.shape[1] != expected_columns
        or not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or not isinstance(node_ids, torch.Tensor)
        or node_ids.ndim != 1
        or predictions.shape[0] != labels.numel()
        or labels.numel() != node_ids.numel()
    ):
        raise ValueError(
            f'prediction columns or tensors differ from calibration methods '
            f'for {arm}/{seed}/{role}'
        )
    prediction_methods = payload.get('uq_methods')
    if prediction_methods is not None and prediction_methods != list(methods):
        raise ValueError(f'prediction methods mismatch for {arm}/{seed}/{role}')
    if node_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f'node IDs are not integer-valued for {arm}/{seed}/{role}')
    if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError(f'nonfinite prediction or label for {arm}/{seed}/{role}')
    node_values = [int(value) for value in node_ids.tolist()]
    if len(set(node_values)) != len(node_values) or set(node_values) != expected_nodes:
        raise ValueError(f'node IDs differ from frozen {role} partition for {arm}/{seed}')
    order = torch.argsort(node_ids)
    return node_ids[order].to(torch.int64), labels[order]


def _load_calibration_state(
    path: Path,
    *,
    seed: int | str,
    arm: str,
    expected_count: int,
    member_seeds: list[int] | None = None,
) -> dict[str, Any]:
    state = _read_json(path)
    if not isinstance(state, dict) or state.get('schema_version') != 'clean-uq-calibration-v1':
        raise ValueError(f'not a clean-UQ calibration state: {path.name}')
    if state.get('seed') != seed or state.get('arm') != arm:
        raise ValueError(f'calibration state identity mismatch for {arm}/{seed}')
    if state.get('calibration_count') != expected_count:
        raise ValueError(f'calibration_count mismatch for {arm}/{seed}')
    if member_seeds is not None and state.get('member_seeds') != member_seeds:
        raise ValueError('ensemble calibration members differ from protocol')
    _state_methods(state, arm=arm, seed=seed)
    return state


def _state_methods(
    state: dict[str, Any], *, arm: str, seed: int | str
) -> tuple[str, ...]:
    raw_methods = state.get('methods')
    if raw_methods not in (['simple'], ['simple', 'cqr']):
        raise ValueError(f'invalid calibration methods for {arm}/{seed}')
    expected_kind = (
        'point-only' if raw_methods == ['simple'] else 'point-with-endpoints'
    )
    if state.get('prediction_kind') != expected_kind:
        raise ValueError(f'prediction kind differs from methods for {arm}/{seed}')
    return tuple(raw_methods)


def _same_path(value: object, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _load_baseline_state(run_root: Path, arm: str) -> dict[str, Any]:
    path = run_root / 'checkpoints' / arm / f'{DETERMINISTIC_RUN_ID}.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    state = _read_json(path)
    if (
        not isinstance(state, dict)
        or state.get('schema_version') != 'clean-uq-baseline-state-v1'
        or state.get('seed') != DETERMINISTIC_RUN_ID
        or state.get('arm') != arm
    ):
        raise ValueError(f'baseline state identity mismatch for {arm}')
    methods = _state_methods(state, arm=arm, seed=DETERMINISTIC_RUN_ID)
    if methods != BASELINE_METHODS[arm]:
        raise ValueError(f'baseline methods mismatch for {arm}')
    if arm == 'GlobalMedian':
        median = state.get('median')
        if isinstance(median, bool) or not isinstance(median, (float, int)) or not math.isfinite(median):
            raise ValueError('GlobalMedian state has an invalid fit median')
    elif arm == 'XGBoost':
        midpoint = run_root / 'checkpoints' / arm / 'midpoint.ubj'
        quantiles = run_root / 'checkpoints' / arm / 'quantiles.ubj'
        if not _same_path(state.get('midpoint_model'), midpoint) or not _same_path(
            state.get('quantile_model'), quantiles
        ):
            raise ValueError('XGBoost state references foreign selected models')
        for model in (midpoint, quantiles):
            if not model.is_file():
                raise FileNotFoundError(model)
    else:
        cache = run_root / 'cache' / arm / f'{DETERMINISTIC_RUN_ID}.pt'
        if (
            state.get('full_graph') is not True
            or state.get('directed') is not True
            or state.get('num_nodes') != FULL_GRAPH_NODE_COUNT
            or state.get('num_edges') != FULL_GRAPH_EDGE_COUNT
            or state.get('estimand') != LP_ESTIMAND
            or state.get('graph_passes') != LP_GRAPH_PASSES
            or state.get('deterministic_algorithms') is not True
            or state.get('execution_device') != 'cpu'
            or not _same_path(state.get('cache'), cache)
        ):
            raise ValueError('RegressionLP state is not bound to the full-graph cache')
    return state


def _validate_lp_support(
    run_root: Path, *, node_sets: dict[str, set[int]]
) -> tuple[Path, Path]:
    probe_path = run_root / 'probe' / 'regression_lp.json'
    if not probe_path.is_file():
        raise FileNotFoundError(probe_path)
    probe = _read_json(probe_path)
    if (
        not isinstance(probe, dict)
        or probe.get('schema_version') != 'clean-uq-lp-probe-v1'
        or probe.get('full_graph') is not True
        or probe.get('directed') is not True
        or probe.get('num_nodes') != FULL_GRAPH_NODE_COUNT
        or probe.get('num_edges') != FULL_GRAPH_EDGE_COUNT
        or probe.get('estimand') != LP_ESTIMAND
        or probe.get('projected_passes') != LP_GRAPH_PASSES
        or probe.get('execution_device') != 'cpu'
        or probe.get('deterministic_algorithms') is not True
        or probe.get('passed') is not True
    ):
        raise ValueError('RegressionLP requires a passed full-graph probe receipt')
    cache_path = run_root / 'cache' / 'RegressionLP' / f'{DETERMINISTIC_RUN_ID}.pt'
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    predictions = cache.get('predictions') if isinstance(cache, dict) else None
    reached = cache.get('reached') if isinstance(cache, dict) else None
    if (
        not isinstance(cache, dict)
        or cache.get('schema_version') != 'clean-uq-lp-cache-v1'
        or cache.get('full_graph') is not True
        or cache.get('directed') is not True
        or cache.get('num_nodes') != FULL_GRAPH_NODE_COUNT
        or cache.get('num_edges') != FULL_GRAPH_EDGE_COUNT
        or cache.get('estimand') != LP_ESTIMAND
        or cache.get('fit_seed_count') != len(node_sets['fit'])
        or not isinstance(predictions, torch.Tensor)
        or predictions.shape != (FULL_GRAPH_NODE_COUNT,)
        or not bool(torch.isfinite(predictions).all())
        or not isinstance(reached, torch.Tensor)
        or reached.dtype != torch.bool
        or reached.shape != (FULL_GRAPH_NODE_COUNT,)
    ):
        raise ValueError('not a valid RegressionLP full-graph cache')
    return probe_path, cache_path


def _assert_aligned(
    reference: tuple[torch.Tensor, torch.Tensor] | None,
    current: tuple[torch.Tensor, torch.Tensor],
    *,
    role: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference is None:
        return current
    if not torch.equal(reference[0], current[0]) or not torch.equal(reference[1], current[1]):
        raise ValueError(f'{role} node/label alignment differs across arms or seeds')
    return reference


def _sealed_artifacts(run_root: Path) -> list[Path]:
    exposed: list[Path] = []
    predictions = run_root / 'predictions'
    audit = run_root / 'audit'
    if predictions.exists():
        exposed.extend(predictions.rglob('test.pt'))
    if audit.exists():
        exposed.extend(audit.rglob('*_test.json'))
        exposed.extend(audit.rglob('*_test.csv'))
    results = run_root / 'results'
    for name in ('clean_uq_metrics.csv', 'clean_uq_exposure_ledger.json'):
        path = results / name
        if path.exists():
            exposed.append(path)
    return exposed


def _check_inventory(label: str, expected: set[Path], observed: set[Path]) -> int:
    if expected != observed:
        raise ValueError(
            f'{label} inventory mismatch: expected {len(expected)}, observed {len(observed)}'
        )
    return len(observed)


def _validate_pretest(
    run_root: Path, *, reject_test_artifacts: bool
) -> dict[str, Any]:
    if reject_test_artifacts and _sealed_artifacts(run_root):
        raise ValueError('sealed-test artifact exists before the sealed-test phase')
    protocol, partitions, funnel = _load_protocol(run_root)
    counts, node_sets, domain_sets = _partition_surface(protocol, partitions)
    seeds: list[int] = protocol['model_seeds']
    calibration_count = len(node_sets['calibrate'])
    reference: tuple[torch.Tensor, torch.Tensor] | None = None
    calibration_states: dict[tuple[int | str, str], dict[str, Any]] = {}
    expected_predictions: set[Path] = set()
    expected_states: set[Path] = set()
    expected_checkpoints: set[Path] = set()
    expected_caches: set[Path] = set()
    for seed in seeds:
        cache_path = run_root / 'cache' / 'GAT' / f'seed-{seed}.pt'
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        expected_caches.add(cache_path)
        for arm in ARMS:
            checkpoint = run_root / 'checkpoints' / arm / f'seed-{seed}.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            expected_checkpoints.add(checkpoint)
            state_path = _state_path(run_root, arm, seed)
            state = _load_calibration_state(
                state_path,
                seed=seed, arm=arm, expected_count=calibration_count,
            )
            methods = _state_methods(state, arm=arm, seed=seed)
            prediction_path = _prediction_path(run_root, arm, seed, 'calibrate')
            current = _load_prediction(
                prediction_path,
                role='calibrate', seed=seed, arm=arm,
                expected_nodes=node_sets['calibrate'], methods=methods,
            )
            reference = _assert_aligned(reference, current, role='calibrate')
            expected_predictions.add(prediction_path)
            calibration_states[(seed, arm)] = state
            expected_states.add(state_path)
    ensemble_state_path = _state_path(run_root, ENSEMBLE_ARM, 'ensemble')
    ensemble_state = _load_calibration_state(
        ensemble_state_path,
        seed='ensemble', arm=ENSEMBLE_ARM, expected_count=calibration_count,
        member_seeds=seeds,
    )
    ensemble_methods = _state_methods(
        ensemble_state, arm=ENSEMBLE_ARM, seed='ensemble'
    )
    ensemble_prediction_path = _prediction_path(
        run_root, ENSEMBLE_ARM, 'ensemble', 'calibrate'
    )
    current = _load_prediction(
        ensemble_prediction_path,
        role='calibrate', seed='ensemble', arm=ENSEMBLE_ARM,
        expected_nodes=node_sets['calibrate'], methods=ensemble_methods,
        member_seeds=seeds,
    )
    reference = _assert_aligned(reference, current, role='calibrate')
    expected_predictions.add(ensemble_prediction_path)
    calibration_states[('ensemble', ENSEMBLE_ARM)] = ensemble_state
    expected_states.add(ensemble_state_path)

    expected_baseline_states: set[Path] = set()
    expected_xgboost_models = {
        run_root / 'checkpoints' / 'XGBoost' / 'midpoint.ubj',
        run_root / 'checkpoints' / 'XGBoost' / 'quantiles.ubj',
    }
    for arm in BASELINE_ARMS:
        baseline_state = _load_baseline_state(run_root, arm)
        if arm == 'GlobalMedian' and baseline_state.get('fit_count') != len(
            node_sets['fit']
        ):
            raise ValueError('GlobalMedian state fit_count differs from the fit role')
        if arm == 'RegressionLP' and baseline_state.get(
            'fit_seed_count'
        ) != len(node_sets['fit']):
            raise ValueError('RegressionLP seed count differs from the fit role')
        baseline_state_path = (
            run_root / 'checkpoints' / arm / f'{DETERMINISTIC_RUN_ID}.json'
        )
        expected_baseline_states.add(baseline_state_path)
        state_path = _state_path(run_root, arm, DETERMINISTIC_RUN_ID)
        state = _load_calibration_state(
            state_path,
            seed=DETERMINISTIC_RUN_ID, arm=arm,
            expected_count=calibration_count,
        )
        methods = _state_methods(
            state, arm=arm, seed=DETERMINISTIC_RUN_ID
        )
        if methods != _state_methods(
            baseline_state, arm=arm, seed=DETERMINISTIC_RUN_ID
        ):
            raise ValueError(f'calibration methods differ from baseline state for {arm}')
        prediction_path = _prediction_path(
            run_root, arm, DETERMINISTIC_RUN_ID, 'calibrate'
        )
        current = _load_prediction(
            prediction_path,
            role='calibrate', seed=DETERMINISTIC_RUN_ID, arm=arm,
            expected_nodes=node_sets['calibrate'], methods=methods,
        )
        reference = _assert_aligned(reference, current, role='calibrate')
        expected_predictions.add(prediction_path)
        calibration_states[(DETERMINISTIC_RUN_ID, arm)] = state
        expected_states.add(state_path)

    lp_probe_path, lp_cache_path = _validate_lp_support(
        run_root, node_sets=node_sets
    )
    prediction_count = _check_inventory(
        'calibration prediction', expected_predictions,
        set((run_root / 'predictions').rglob('calibrate.pt')),
    )
    state_count = _check_inventory(
        'calibration state', expected_states,
        set((run_root / 'audit').rglob('*_calibration.json')),
    )
    checkpoint_count = _check_inventory(
        'checkpoint', expected_checkpoints,
        set((run_root / 'checkpoints').rglob('seed-*.pt')),
    )
    baseline_state_count = _check_inventory(
        'baseline state', expected_baseline_states,
        set((run_root / 'checkpoints').rglob(f'{DETERMINISTIC_RUN_ID}.json')),
    )
    xgboost_model_count = _check_inventory(
        'XGBoost selected model', expected_xgboost_models,
        set((run_root / 'checkpoints').rglob('*.ubj')),
    )
    all_checkpoint_files = {
        path for path in (run_root / 'checkpoints').rglob('*') if path.is_file()
    }
    _check_inventory(
        'all checkpoint files',
        expected_checkpoints | expected_baseline_states | expected_xgboost_models,
        all_checkpoint_files,
    )
    cache_count = _check_inventory(
        'parent cache', expected_caches,
        set((run_root / 'cache' / 'GAT').glob('seed-*.pt')),
    )
    _check_inventory(
        'all cache files', expected_caches | {lp_cache_path},
        {path for path in (run_root / 'cache').rglob('*') if path.is_file()},
    )
    _check_inventory(
        'LP probe', {lp_probe_path},
        {path for path in (run_root / 'probe').rglob('*') if path.is_file()},
    )
    expected_seed_arms = len(seeds) * len(ARMS)
    expected_evaluations = expected_seed_arms + 1 + len(BASELINE_ARMS)
    return {
        'schema_version': 'clean-uq-pretest-ledger-v1',
        'phase': 'pretest',
        'model_seeds': seeds,
        'arms': list(ARMS),
        'ensemble_arm': ENSEMBLE_ARM,
        'baseline_arms': list(BASELINE_ARMS),
        'baseline_run_id': DETERMINISTIC_RUN_ID,
        'partition_counts': counts,
        'population_funnel': funnel,
        'expected_artifacts': {
            'calibration_predictions': expected_evaluations,
            'calibration_states': expected_evaluations,
            'checkpoints': expected_seed_arms,
            'baseline_states': len(BASELINE_ARMS),
            'xgboost_models': len(expected_xgboost_models),
            'parent_caches': len(seeds),
            'lp_caches': 1,
            'lp_probes': 1,
        },
        'observed_artifacts': {
            'calibration_predictions': prediction_count,
            'calibration_states': state_count,
            'checkpoints': checkpoint_count,
            'baseline_states': baseline_state_count,
            'xgboost_models': xgboost_model_count,
            'parent_caches': cache_count,
            'lp_caches': 1,
            'lp_probes': 1,
        },
        'calibration_unique_domain_exposure': len(domain_sets['calibrate']),
        'sealed_test_artifacts_present': False,
        '_calibration_states': calibration_states,
        '_node_sets': node_sets,
        '_domain_sets': domain_sets,
    }


def _public_pretest_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in ledger.items() if not key.startswith('_')}


def _write_json_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite result: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )


def collect_pretest(
    run_root: Path, *, output_dir: Path | None = None, write_output: bool = True
) -> dict[str, Any]:
    """Validate that calibration is complete and the sealed test is untouched."""
    ledger = _public_pretest_ledger(
        _validate_pretest(run_root, reject_test_artifacts=True)
    )
    if write_output:
        destination = (output_dir or run_root / 'results') / 'clean_uq_pretest_ledger.json'
        _write_json_once(destination, ledger)
    return ledger


def _load_audit_rows(
    path: Path,
    *,
    seed: int | str,
    arm: str,
    expected_count: int,
    calibration_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _read_json(path)
    methods = _state_methods(calibration_state, arm=arm, seed=seed)
    if not isinstance(rows, list) or len(rows) != len(methods):
        raise ValueError(
            f'test audit row count differs from calibration methods for {arm}/{seed}'
        )
    if [row.get('method') for row in rows if isinstance(row, dict)] != list(methods):
        raise ValueError(f'test audit methods mismatch for {arm}/{seed}')
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get('schema_version') != 'clean-uq-audit-row-v1'
            or row.get('seed') != seed
            or row.get('arm') != arm
        ):
            raise ValueError(f'test audit identity mismatch for {arm}/{seed}')
        if row.get('test_count') != expected_count:
            raise ValueError(f'test_count mismatch for {arm}/{seed}')
        if row.get('alpha') != calibration_state.get('alpha'):
            raise ValueError(f'test audit alpha differs from calibration for {arm}/{seed}')
        required_metrics = (
            'qhat',
            'coverage',
            'mean_width',
            'median_width',
            'interval_score',
            'midpoint_mae',
            'midpoint_rmse',
        )
        for metric in required_metrics:
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f'missing or nonfinite {metric} for {arm}/{seed}')
        if not 0.0 <= float(row['coverage']) <= 1.0:
            raise ValueError(f'coverage outside [0, 1] for {arm}/{seed}')
        nonnegative = (
            'qhat',
            'mean_width',
            'median_width',
            'interval_score',
            'midpoint_mae',
            'midpoint_rmse',
        )
        if any(float(row[metric]) < 0.0 for metric in nonnegative):
            raise ValueError(f'negative test metric for {arm}/{seed}')
        if 'midpoint_spearman' not in row or 'raw_crossing_rate' not in row:
            raise ValueError(f'test audit omits a required nullable metric for {arm}/{seed}')
        spearman = row['midpoint_spearman']
        if spearman is not None and (
            isinstance(spearman, bool)
            or not isinstance(spearman, (int, float))
            or not math.isfinite(float(spearman))
            or not -1.0 <= float(spearman) <= 1.0
        ):
            raise ValueError(f'invalid midpoint_spearman for {arm}/{seed}')
        crossing = row['raw_crossing_rate']
        if calibration_state.get('prediction_kind') == 'point-only':
            if crossing is not None:
                raise ValueError(f'point-only arm reports crossing for {arm}/{seed}')
        elif (
            isinstance(crossing, bool)
            or not isinstance(crossing, (int, float))
            or not math.isfinite(float(crossing))
            or not 0.0 <= float(crossing) <= 1.0
        ):
            raise ValueError(f'invalid raw_crossing_rate for {arm}/{seed}')
    return rows


def _write_metrics_once(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite result: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_sealed_test(
    run_root: Path, *, output_dir: Path | None = None
) -> dict[str, Any]:
    """Validate sealed artifacts and copy every audit row without aggregation."""
    results_dir = output_dir or run_root / 'results'
    metrics_path = results_dir / 'clean_uq_metrics.csv'
    ledger_path = results_dir / 'clean_uq_exposure_ledger.json'
    if metrics_path.exists() or ledger_path.exists():
        raise FileExistsError('refusing to overwrite sealed-test results')
    pretest = _validate_pretest(run_root, reject_test_artifacts=False)
    seeds: list[int] = pretest['model_seeds']
    node_sets: dict[str, set[int]] = pretest['_node_sets']
    domain_sets: dict[str, set[str]] = pretest['_domain_sets']
    states: dict[tuple[int | str, str], dict[str, Any]] = pretest['_calibration_states']
    test_count = len(node_sets['test'])
    reference: tuple[torch.Tensor, torch.Tensor] | None = None
    metric_rows: list[dict[str, Any]] = []
    expected_predictions: set[Path] = set()
    expected_audits: set[Path] = set()
    for seed in seeds:
        for arm in ARMS:
            state = states[(seed, arm)]
            methods = _state_methods(state, arm=arm, seed=seed)
            prediction_path = _prediction_path(run_root, arm, seed, 'test')
            current = _load_prediction(
                prediction_path,
                role='test', seed=seed, arm=arm,
                expected_nodes=node_sets['test'], methods=methods,
            )
            reference = _assert_aligned(reference, current, role='test')
            expected_predictions.add(prediction_path)
            audit_path = _audit_path(run_root, arm, seed)
            metric_rows.extend(
                _load_audit_rows(
                    audit_path,
                    seed=seed, arm=arm, expected_count=test_count,
                    calibration_state=state,
                )
            )
            expected_audits.add(audit_path)
    ensemble_prediction_path = _prediction_path(
        run_root, ENSEMBLE_ARM, 'ensemble', 'test'
    )
    ensemble_state = states[('ensemble', ENSEMBLE_ARM)]
    ensemble_methods = _state_methods(
        ensemble_state, arm=ENSEMBLE_ARM, seed='ensemble'
    )
    current = _load_prediction(
        ensemble_prediction_path,
        role='test', seed='ensemble', arm=ENSEMBLE_ARM,
        expected_nodes=node_sets['test'], methods=ensemble_methods,
        member_seeds=seeds,
    )
    reference = _assert_aligned(reference, current, role='test')
    expected_predictions.add(ensemble_prediction_path)
    ensemble_audit_path = _audit_path(run_root, ENSEMBLE_ARM, 'ensemble')
    metric_rows.extend(
        _load_audit_rows(
            ensemble_audit_path,
            seed='ensemble', arm=ENSEMBLE_ARM, expected_count=test_count,
            calibration_state=ensemble_state,
        )
    )
    expected_audits.add(ensemble_audit_path)
    for arm in BASELINE_ARMS:
        state = states[(DETERMINISTIC_RUN_ID, arm)]
        methods = _state_methods(
            state, arm=arm, seed=DETERMINISTIC_RUN_ID
        )
        prediction_path = _prediction_path(
            run_root, arm, DETERMINISTIC_RUN_ID, 'test'
        )
        current = _load_prediction(
            prediction_path,
            role='test', seed=DETERMINISTIC_RUN_ID, arm=arm,
            expected_nodes=node_sets['test'], methods=methods,
        )
        reference = _assert_aligned(reference, current, role='test')
        expected_predictions.add(prediction_path)
        audit_path = _audit_path(run_root, arm, DETERMINISTIC_RUN_ID)
        metric_rows.extend(
            _load_audit_rows(
                audit_path,
                seed=DETERMINISTIC_RUN_ID, arm=arm,
                expected_count=test_count, calibration_state=state,
            )
        )
        expected_audits.add(audit_path)
    prediction_count = _check_inventory(
        'test prediction', expected_predictions,
        set((run_root / 'predictions').rglob('test.pt')),
    )
    audit_count = _check_inventory(
        'test audit', expected_audits,
        set((run_root / 'audit').rglob('*_test.json')),
    )
    expected_artifacts = len(seeds) * len(ARMS) + 1 + len(BASELINE_ARMS)
    expected_rows = sum(len(_state_methods(state, arm=arm, seed=seed)) for (seed, arm), state in states.items())
    if len(metric_rows) != expected_rows:
        raise ValueError('sealed metric row count is incomplete')
    ledger = {
        'schema_version': 'clean-uq-exposure-ledger-v1',
        'phase': 'sealed-test',
        'population_funnel': pretest['population_funnel'],
        'partition_counts': pretest['partition_counts'],
        'model_seeds': seeds,
        'arms': list(ARMS),
        'ensemble_arm': ENSEMBLE_ARM,
        'baseline_arms': list(BASELINE_ARMS),
        'baseline_run_id': DETERMINISTIC_RUN_ID,
        'expected': {
            **pretest['expected_artifacts'],
            'test_predictions': expected_artifacts,
            'test_audits': expected_artifacts,
            'metric_rows': expected_rows,
        },
        'observed': {
            **pretest['observed_artifacts'],
            'test_predictions': prediction_count,
            'test_audits': audit_count,
            'metric_rows': len(metric_rows),
        },
        'unique_domain_exposure': {
            'calibrate': len(domain_sets['calibrate']),
            'test': len(domain_sets['test']),
        },
        'population_counting_policy': 'unique canonical domains; repeated arm and seed evaluations are not additional population',
        'metric_policy': 'unaggregated audit rows; no pooled or newly computed statistics',
    }
    _write_metrics_once(metrics_path, metric_rows)
    _write_json_once(ledger_path, ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pretest', 'sealed-test'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.phase == 'pretest':
        result = collect_pretest(args.run_root, output_dir=args.output_dir)
    else:
        result = collect_sealed_test(args.run_root, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
