"""Fit and predict the three pre-registered strong clean-UQ baselines."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    load_role_targets,
    sha256_file,
)


DETERMINISTIC_RUN_ID = 'deterministic'
XGBOOST_VERSION = '3.2.0'
XGBOOST_GRID: tuple[dict[str, int], ...] = (
    {'max_depth': 4, 'n_estimators': 400},
    {'max_depth': 8, 'n_estimators': 400},
)
LP_GRID: tuple[tuple[float, int], ...] = tuple(
    (damping, iterations)
    for damping in (0.50, 0.80, 0.95)
    for iterations in (5, 10, 20)
)
LP_GRAPH_PASSES = sum(
    max(iterations for current, iterations in LP_GRID if current == damping)
    for damping in {candidate[0] for candidate in LP_GRID}
)
LP_ESTIMAND = (
    'directed_source_to_target_outdegree_two_channel_restart_'
    'fit_seeds_only_no_self_loops'
)
FULL_GRAPH_SHA256 = '0c09d2b5aeefae0c5f5306be19dbf0ee9b1b4df85ea02436f14da2280487aff1'
FULL_GRAPH_SIZE_BYTES = 28_120_435_333
FULL_GRAPH_NODE_COUNT = 45_030_252
FULL_GRAPH_EDGE_COUNT = 1_014_523_551

FloatArray = npt.NDArray[np.float32]


class XGBoostEstimator(Protocol):
    """Minimal estimator interface needed by the fixed XGBoost grid."""

    def fit(self, features: FloatArray, targets: FloatArray) -> XGBoostEstimator:
        """Fit on the frozen fit role."""

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict on a role without updating the estimator."""

    def save_model(self, path: str | Path) -> None:
        """Persist the selected estimator."""


EstimatorFactory = Callable[[str, dict[str, int]], XGBoostEstimator]


@dataclass(frozen=True)
class XGBoostSelection:
    """Selected fit-only midpoint and native-quantile XGBoost models."""

    midpoint_config: dict[str, int]
    quantile_config: dict[str, int]
    midpoint_model: XGBoostEstimator
    quantile_model: XGBoostEstimator
    select_mae: float
    select_pinball: float
    native_quantiles: bool = True


@dataclass(frozen=True)
class LabelPropagationSelection:
    """Selected full-graph deterministic regression propagation state."""

    damping: float
    iterations: int
    select_mae: float
    predictions: torch.Tensor
    reached: torch.Tensor
    fit_median: float


def _validate_node_ids(node_ids: torch.Tensor, num_nodes: int, role: str) -> torch.Tensor:
    ids = node_ids.detach().to(dtype=torch.long, device='cpu')
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError(f'{role} node IDs must be a nonempty vector')
    if int(ids.min()) < 0 or int(ids.max()) >= num_nodes:
        raise ValueError(f'{role} node IDs fall outside the full graph')
    if len(set(int(value) for value in ids.tolist())) != ids.numel():
        raise ValueError(f'{role} node IDs must be unique')
    return ids


def _validated_targets(
    labels: torch.Tensor, node_ids: torch.Tensor, role: str
) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError('targets must be a one-dimensional tensor')
    ids = _validate_node_ids(node_ids, labels.numel(), role)
    targets = labels[ids].float()
    if not bool(torch.isfinite(targets).all()) or bool(
        ((targets < 0.0) | (targets > 1.0)).any()
    ):
        raise ValueError(f'{role} targets must be finite values in [0, 1]')
    return targets


def fit_global_median(labels: torch.Tensor, fit_node_ids: torch.Tensor) -> float:
    """Estimate the constant baseline exclusively from fit targets."""
    fit_targets = _validated_targets(labels, fit_node_ids, 'fit')
    return float(torch.quantile(fit_targets, 0.5))


def _default_estimator_factory(
    kind: str, config: dict[str, int]
) -> XGBoostEstimator:
    import xgboost

    if xgboost.__version__ != XGBOOST_VERSION:
        raise RuntimeError(
            f'clean-UQ requires xgboost-cpu=={XGBOOST_VERSION}, found '
            f'{xgboost.__version__}'
        )
    common: dict[str, Any] = {
        **config,
        'learning_rate': 0.05,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'tree_method': 'hist',
        'n_jobs': 16,
        'random_state': 42,
    }
    if kind == 'midpoint':
        return cast(
            XGBoostEstimator,
            xgboost.XGBRegressor(objective='reg:squarederror', **common),
        )
    if kind == 'quantile':
        return cast(
            XGBoostEstimator,
            xgboost.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=np.asarray((0.05, 0.95), dtype=np.float32),
                **common,
            ),
        )
    raise ValueError(f'unsupported XGBoost estimator kind: {kind}')


def _feature_rows(features: torch.Tensor, node_ids: torch.Tensor) -> FloatArray:
    if features.ndim != 2:
        raise ValueError('XGBoost features must have shape [N, D]')
    ids = _validate_node_ids(node_ids, features.shape[0], 'feature')
    rows = features[ids].detach().cpu().float().numpy()
    if not np.isfinite(rows).all():
        raise ValueError('XGBoost features must be finite')
    return rows


def _quantile_pinball(
    predictions: FloatArray, targets: FloatArray
) -> float:
    residual = targets[:, None] - predictions
    quantiles = np.asarray((0.05, 0.95), dtype=np.float32)
    return float(np.maximum(quantiles * residual, (quantiles - 1.0) * residual).mean())


def fit_xgboost_with_selection(
    features: torch.Tensor,
    labels: torch.Tensor,
    fit_node_ids: torch.Tensor,
    select_node_ids: torch.Tensor,
    *,
    grid: Iterable[Mapping[str, int]] = XGBOOST_GRID,
    estimator_factory: EstimatorFactory | None = None,
) -> XGBoostSelection:
    """Fit candidates on fit and select one using only the select role."""
    if labels.numel() != features.shape[0]:
        raise ValueError('XGBoost labels and features must share the full node axis')
    fit_ids = _validate_node_ids(fit_node_ids, features.shape[0], 'fit')
    select_ids = _validate_node_ids(select_node_ids, features.shape[0], 'select')
    if bool(torch.isin(fit_ids, select_ids).any()):
        raise ValueError('fit and select node IDs must be disjoint')
    fit_x = _feature_rows(features, fit_ids)
    select_x = _feature_rows(features, select_ids)
    fit_y = _validated_targets(labels, fit_ids, 'fit').numpy()
    select_y = _validated_targets(labels, select_ids, 'select').numpy()
    factory = estimator_factory or _default_estimator_factory
    best_midpoint: tuple[tuple[float, int, int], dict[str, int], XGBoostEstimator] | None = None
    best_quantile: tuple[tuple[float, int, int], dict[str, int], XGBoostEstimator] | None = None
    for raw_config in grid:
        config = {str(key): int(value) for key, value in raw_config.items()}
        if set(config) != {'max_depth', 'n_estimators'}:
            raise ValueError('XGBoost grid entries require max_depth and n_estimators')
        midpoint = factory('midpoint', config)
        quantile = factory('quantile', config)
        midpoint.fit(fit_x, fit_y)
        quantile.fit(fit_x, fit_y)
        select_midpoint = np.asarray(midpoint.predict(select_x), dtype=np.float32)
        select_quantiles = np.asarray(quantile.predict(select_x), dtype=np.float32)
        if select_midpoint.shape != (len(select_y),):
            raise RuntimeError('XGBoost midpoint prediction has an unexpected shape')
        if select_quantiles.shape != (len(select_y), 2):
            raise RuntimeError(
                'xgboost-cpu 3.2.0 did not produce native quantile endpoints'
            )
        mae = float(np.mean(np.abs(select_midpoint - select_y)))
        pinball = _quantile_pinball(select_quantiles, select_y)
        midpoint_key = (mae, config['max_depth'], config['n_estimators'])
        quantile_key = (pinball, config['max_depth'], config['n_estimators'])
        if best_midpoint is None or midpoint_key < best_midpoint[0]:
            best_midpoint = (midpoint_key, config, midpoint)
        if best_quantile is None or quantile_key < best_quantile[0]:
            best_quantile = (quantile_key, config, quantile)
    if best_midpoint is None or best_quantile is None:
        raise ValueError('XGBoost grid must contain at least one configuration')
    return XGBoostSelection(
        midpoint_config=best_midpoint[1],
        quantile_config=best_quantile[1],
        midpoint_model=best_midpoint[2],
        quantile_model=best_quantile[2],
        select_mae=best_midpoint[0][0],
        select_pinball=best_quantile[0][0],
    )


def predict_xgboost(
    midpoint_model: XGBoostEstimator,
    quantile_model: XGBoostEstimator,
    features: torch.Tensor,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    """Return columns [midpoint, native lower, native upper] for one role."""
    rows = _feature_rows(features, node_ids)
    midpoint = np.asarray(midpoint_model.predict(rows), dtype=np.float32)
    endpoints = np.asarray(quantile_model.predict(rows), dtype=np.float32)
    if midpoint.shape != (rows.shape[0],) or endpoints.shape != (rows.shape[0], 2):
        raise RuntimeError('selected XGBoost models returned incompatible predictions')
    return torch.from_numpy(np.column_stack((midpoint, endpoints)))


def _lp_seed_state(
    num_nodes: int,
    fit_node_ids: torch.Tensor,
    fit_targets: torch.Tensor,
) -> torch.Tensor:
    state = torch.zeros((num_nodes, 2), dtype=torch.float32)
    fit_ids = fit_node_ids.cpu()
    state[fit_ids, 0] = fit_targets.cpu()
    state[fit_ids, 1] = 1.0
    return state


def _lp_out_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(edge_index[0], minlength=num_nodes).to(dtype=torch.float32)


@contextmanager
def _deterministic_cpu_lp(edge_index: torch.Tensor) -> Iterator[None]:
    if edge_index.device.type != 'cpu':
        raise ValueError('deterministic RegressionLP must execute on CPU')
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _lp_pass(
    edge_index: torch.Tensor,
    state: torch.Tensor,
    out_degree: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError('LP chunk_size must be positive')
    propagated = torch.zeros_like(state)
    edge_count = int(edge_index.shape[1])
    for start in range(0, edge_count, chunk_size):
        edge_chunk = edge_index[:, start : start + chunk_size]
        source = edge_chunk[0]
        target = edge_chunk[1]
        contribution = state[source] / out_degree[source].clamp_min(1.0)[:, None]
        propagated.index_add_(0, target, contribution)
    return propagated


def _lp_predictions(
    state: torch.Tensor,
    fit_median: float,
    fit_node_ids: torch.Tensor,
    fit_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mass = state[:, 1]
    predictions = torch.full_like(mass, fit_median)
    reached = mass > torch.finfo(state.dtype).eps
    predictions[reached] = state[reached, 0] / mass[reached]
    predictions.clamp_(0.0, 1.0)
    predictions[fit_node_ids] = fit_targets
    return predictions, reached


def regression_label_propagation(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    fit_node_ids: torch.Tensor,
    fit_targets: torch.Tensor,
    damping: float,
    iterations: int,
    chunk_size: int = 8_388_608,
) -> torch.Tensor:
    """Run directed two-channel diffusion with fit labels as the only seeds."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('full graph edge_index must have shape [2, E]')
    if not 0.0 < damping < 1.0 or iterations < 1:
        raise ValueError('LP damping and iterations must be positive and bounded')
    fit_ids = _validate_node_ids(fit_node_ids, num_nodes, 'fit')
    targets = fit_targets.detach().float().cpu()
    if targets.shape != fit_ids.shape or not bool(torch.isfinite(targets).all()):
        raise ValueError('fit targets must align with LP fit node IDs')
    seed = _lp_seed_state(num_nodes, fit_ids, targets)
    state = seed.clone()
    with _deterministic_cpu_lp(edge_index):
        out_degree = _lp_out_degree(edge_index, num_nodes)
        for _ in range(iterations):
            propagated = _lp_pass(
                edge_index, state, out_degree, chunk_size=chunk_size
            )
            state = damping * propagated + (1.0 - damping) * seed
    fit_median = float(torch.quantile(targets, 0.5))
    predictions, _ = _lp_predictions(state, fit_median, fit_ids, targets)
    return predictions


def select_label_propagation(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    labels: torch.Tensor,
    fit_node_ids: torch.Tensor,
    select_node_ids: torch.Tensor,
    grid: Iterable[tuple[float, int]] = LP_GRID,
    chunk_size: int = 8_388_608,
) -> LabelPropagationSelection:
    """Choose damping/iterations on select while injecting fit targets only."""
    if labels.numel() != num_nodes:
        raise ValueError('LP labels must align with the full graph node axis')
    fit_ids = _validate_node_ids(fit_node_ids, num_nodes, 'fit')
    select_ids = _validate_node_ids(select_node_ids, num_nodes, 'select')
    if bool(torch.isin(fit_ids, select_ids).any()):
        raise ValueError('fit and select node IDs must be disjoint')
    fit_targets = _validated_targets(labels, fit_ids, 'fit')
    select_targets = _validated_targets(labels, select_ids, 'select')
    fit_median = float(torch.quantile(fit_targets, 0.5))
    candidates = tuple((float(damping), int(iterations)) for damping, iterations in grid)
    if not candidates:
        raise ValueError('LP grid must contain at least one configuration')
    if any(not 0.0 < damping < 1.0 or iterations < 1 for damping, iterations in candidates):
        raise ValueError('LP grid contains an invalid configuration')
    if edge_index.device.type != 'cpu':
        raise ValueError('deterministic RegressionLP must execute on CPU')
    out_degree = _lp_out_degree(edge_index, num_nodes)
    seed = _lp_seed_state(num_nodes, fit_ids, fit_targets)
    best_key: tuple[float, float, int] | None = None
    best_predictions: torch.Tensor | None = None
    best_reached: torch.Tensor | None = None
    with _deterministic_cpu_lp(edge_index):
        for damping in sorted({candidate[0] for candidate in candidates}):
            requested = {
                iterations for current_damping, iterations in candidates
                if current_damping == damping
            }
            state = seed.clone()
            for iteration in range(1, max(requested) + 1):
                propagated = _lp_pass(
                    edge_index, state, out_degree, chunk_size=chunk_size
                )
                state = damping * propagated + (1.0 - damping) * seed
                if iteration not in requested:
                    continue
                predictions, reached = _lp_predictions(
                    state, fit_median, fit_ids, fit_targets
                )
                mae = float(
                    torch.mean(torch.abs(predictions[select_ids] - select_targets))
                )
                key = (mae, damping, iteration)
                if best_key is None or key < best_key:
                    best_key = key
                    best_predictions = predictions.clone()
                    best_reached = reached.clone()
    if best_key is None or best_predictions is None or best_reached is None:
        raise RuntimeError('LP selection did not produce a candidate')
    return LabelPropagationSelection(
        damping=best_key[1],
        iterations=best_key[2],
        select_mae=best_key[0],
        predictions=best_predictions,
        reached=best_reached,
        fit_median=fit_median,
    )


def write_baseline_role_predictions(
    predictions: torch.Tensor,
    node_ids: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    *,
    role: str,
    arm: str,
) -> dict[str, Any]:
    """Write an aligned deterministic prediction artifact for one role."""
    ids = _validate_node_ids(node_ids, labels.numel(), role)
    role_labels = _validated_targets(labels, ids, role)
    if predictions.ndim == 1:
        if predictions.numel() != labels.numel():
            raise ValueError('point predictions must align with the full node axis')
        selected_predictions = predictions[ids].float().reshape(-1, 1)
        methods = ['simple']
    elif predictions.ndim == 2 and predictions.shape == (labels.numel(), 3):
        selected_predictions = predictions[ids].float()
        methods = ['simple', 'cqr']
    else:
        raise ValueError('baseline predictions must have shape [N] or [N, 3]')
    payload: dict[str, Any] = {
        'schema_version': 'clean-uq-predictions-v1',
        'role': role,
        'seed': DETERMINISTIC_RUN_ID,
        'arm': arm,
        'node_ids': ids,
        'labels': role_labels,
        'predictions': selected_predictions,
        'uq_methods': methods,
    }
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite predictions: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def _read_partition_role(path: Path, role: str) -> torch.Tensor:
    node_ids: list[int] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get('clean_uq_split') == role:
            node_id = row.get('node_id')
            if isinstance(node_id, bool) or not isinstance(node_id, int):
                raise ValueError('partition node IDs must be integers')
            node_ids.append(node_id)
    if not node_ids:
        raise ValueError(f'partition role {role!r} is empty')
    return torch.tensor(node_ids, dtype=torch.long)


def _load_graph(path: Path, *, mmap: bool) -> Any:
    loaded = torch.load(
        path, map_location='cpu', weights_only=False, mmap=mmap
    )
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded[0]
    return loaded


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite state: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )


def _state_path(run_dir: Path, arm: str) -> Path:
    return run_dir / 'checkpoints' / arm / f'{DETERMINISTIC_RUN_ID}.json'


def _prediction_path(run_dir: Path, arm: str, role: str) -> Path:
    return run_dir / 'predictions' / arm / DETERMINISTIC_RUN_ID / f'{role}.pt'


def _load_targets(
    path: Path,
    partition_jsonl: Path,
    expected_roles: tuple[str, ...],
    num_nodes: int,
) -> torch.Tensor:
    return load_role_targets(
        path,
        partition_jsonl=partition_jsonl,
        expected_roles=expected_roles,
        num_nodes=num_nodes,
    )


def _run_median_fit(args: argparse.Namespace) -> dict[str, Any]:
    labels = _load_targets(
        args.targets_pt, args.partition_jsonl, ('fit', 'select'), args.num_nodes
    )
    fit_ids = _read_partition_role(args.partition_jsonl, 'fit')
    median = fit_global_median(labels, fit_ids)
    state_path = _state_path(args.run_dir, 'GlobalMedian')
    state = {
        'schema_version': 'clean-uq-baseline-state-v1',
        'arm': 'GlobalMedian',
        'seed': DETERMINISTIC_RUN_ID,
        'fit_count': int(fit_ids.numel()),
        'median': median,
        'prediction_kind': 'point-only',
        'methods': ['simple'],
    }
    _write_json_once(state_path, state)
    return {**state, 'state_path': str(state_path)}


def _run_xgboost_fit(args: argparse.Namespace) -> dict[str, Any]:
    data = _load_graph(args.data_pt, mmap=True)
    num_nodes = int(data.num_nodes)
    labels = _load_targets(
        args.targets_pt, args.partition_jsonl, ('fit', 'select'), num_nodes
    )
    fit_ids = _read_partition_role(args.partition_jsonl, 'fit')
    select_ids = _read_partition_role(args.partition_jsonl, 'select')
    selection = fit_xgboost_with_selection(
        data.x, labels, fit_ids, select_ids
    )
    state_path = _state_path(args.run_dir, 'XGBoost')
    model_dir = state_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    midpoint_path = model_dir / 'midpoint.ubj'
    quantile_path = model_dir / 'quantiles.ubj'
    if midpoint_path.exists() or quantile_path.exists():
        raise FileExistsError('refusing to overwrite selected XGBoost model')
    selection.midpoint_model.save_model(midpoint_path)
    selection.quantile_model.save_model(quantile_path)
    state = {
        'schema_version': 'clean-uq-baseline-state-v1',
        'arm': 'XGBoost',
        'seed': DETERMINISTIC_RUN_ID,
        'xgboost_distribution': 'xgboost-cpu',
        'xgboost_version': XGBOOST_VERSION,
        'deterministic': True,
        'selected_midpoint_config': selection.midpoint_config,
        'selected_quantile_config': selection.quantile_config,
        'select_mae': selection.select_mae,
        'select_pinball': selection.select_pinball,
        'midpoint_model': str(midpoint_path),
        'quantile_model': str(quantile_path),
        'prediction_kind': 'point-with-endpoints',
        'methods': ['simple', 'cqr'],
    }
    _write_json_once(state_path, state)
    return {**state, 'state_path': str(state_path)}


def _lp_grid_payload() -> list[dict[str, float | int]]:
    return [
        {'damping': damping, 'iterations': iterations}
        for damping, iterations in LP_GRID
    ]


def _validated_full_graph_identity(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    if size_bytes != FULL_GRAPH_SIZE_BYTES:
        raise ValueError('RegressionLP data.pt size differs from the frozen full graph')
    digest = sha256_file(path)
    if digest != FULL_GRAPH_SHA256:
        raise ValueError('RegressionLP data.pt SHA-256 differs from the frozen full graph')
    return {
        'data_path': str(path.resolve()),
        'data_size_bytes': size_bytes,
        'data_sha256': digest,
    }


def _lp_provenance(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_size < 1 or args.threads < 1:
        raise ValueError('RegressionLP chunk size and thread count must be positive')
    return {
        **_validated_full_graph_identity(args.data_pt),
        'targets_path': str(args.targets_pt.resolve()),
        'targets_sha256': sha256_file(args.targets_pt),
        'partition_path': str(args.partition_jsonl.resolve()),
        'partition_sha256': sha256_file(args.partition_jsonl),
        'chunk_size': args.chunk_size,
        'threads': args.threads,
        'grid': _lp_grid_payload(),
        'projected_passes': LP_GRAPH_PASSES,
        'estimand': LP_ESTIMAND,
        'execution_device': 'cpu',
        'deterministic_algorithms': True,
    }


def _write_lp_probe_failure(
    receipt_path: Path, provenance: dict[str, Any], error: Exception
) -> None:
    receipt = {
        'schema_version': 'clean-uq-lp-probe-v1',
        'full_graph': True,
        'directed': True,
        **provenance,
        'passed': False,
        'error_type': type(error).__name__,
        'error': str(error),
    }
    _write_json_once(receipt_path, receipt)


def _run_lp_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.passes < 1 or args.passes > LP_GRAPH_PASSES:
        raise ValueError('LP probe passes must be between 1 and the fixed grid total')
    receipt_path = args.run_dir / 'probe' / 'regression_lp.json'
    provenance: dict[str, Any] = {
        'data_path': str(args.data_pt.resolve()),
        'targets_path': str(args.targets_pt.resolve()),
        'partition_path': str(args.partition_jsonl.resolve()),
        'chunk_size': args.chunk_size,
        'threads': args.threads,
        'grid': _lp_grid_payload(),
        'projected_passes': LP_GRAPH_PASSES,
        'estimand': LP_ESTIMAND,
        'execution_device': 'cpu',
        'deterministic_algorithms': True,
    }
    try:
        probe_start = time.monotonic()
        provenance = _lp_provenance(args)
        load_start = time.monotonic()
        data = _load_graph(args.data_pt, mmap=True)
        mmap_load_seconds = time.monotonic() - load_start
        num_nodes = int(data.num_nodes)
        num_edges = int(data.edge_index.shape[1])
        if num_nodes != FULL_GRAPH_NODE_COUNT or num_edges != FULL_GRAPH_EDGE_COUNT:
            raise ValueError('RegressionLP graph dimensions differ from the frozen full graph')
        fit_ids = _read_partition_role(args.partition_jsonl, 'fit')
        labels = _load_targets(
            args.targets_pt, args.partition_jsonl, ('fit', 'select'), num_nodes
        )
        fit_targets = _validated_targets(labels, fit_ids, 'fit')
        torch.set_num_threads(args.threads)
        degree_start = time.monotonic()
        with _deterministic_cpu_lp(data.edge_index):
            out_degree = _lp_out_degree(data.edge_index, num_nodes)
            degree_seconds = time.monotonic() - degree_start
            seed = _lp_seed_state(num_nodes, fit_ids, fit_targets)
            state = seed.clone()
            pass_seconds: list[float] = []
            for _ in range(args.passes):
                pass_start = time.monotonic()
                propagated = _lp_pass(
                    data.edge_index, state, out_degree, chunk_size=args.chunk_size
                )
                state = 0.8 * propagated + 0.2 * seed
                pass_seconds.append(time.monotonic() - pass_start)
            snapshot_start = time.monotonic()
            predictions, reached = _lp_predictions(
                state, float(torch.quantile(fit_targets, 0.5)), fit_ids, fit_targets
            )
            predictions.clone()
            reached.clone()
            snapshot_seconds = time.monotonic() - snapshot_start
        median_pass_seconds = float(np.median(np.asarray(pass_seconds)))
        projected_seconds = (
            time.monotonic() - probe_start
            + median_pass_seconds * (LP_GRAPH_PASSES - args.passes)
            + snapshot_seconds * 9
        )
        projected_hours = projected_seconds / 3600.0
        receipt = {
            'schema_version': 'clean-uq-lp-probe-v1',
            'full_graph': True,
            'directed': True,
            **provenance,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'fit_seed_count': int(fit_ids.numel()),
            'mmap_load_seconds': mmap_load_seconds,
            'degree_seconds': degree_seconds,
            'pass_seconds': pass_seconds,
            'snapshot_seconds': snapshot_seconds,
            'edges_per_second': int(num_edges / median_pass_seconds),
            'projected_hours': projected_hours,
            'max_projected_hours': args.max_projected_hours,
            'host_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'passed': projected_hours <= args.max_projected_hours,
        }
    except Exception as error:
        _write_lp_probe_failure(receipt_path, provenance, error)
        raise
    _write_json_once(receipt_path, receipt)
    if not receipt['passed']:
        raise RuntimeError(
            'full-graph LP throughput exceeds the pre-registered runtime limit'
        )
    return {**receipt, 'receipt_path': str(receipt_path)}


def _run_lp_fit(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = args.run_dir / 'probe' / 'regression_lp.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    provenance = _lp_provenance(args)
    if (
        receipt.get('schema_version') != 'clean-uq-lp-probe-v1'
        or receipt.get('passed') is not True
        or any(receipt.get(key) != value for key, value in provenance.items())
    ):
        raise ValueError('RegressionLP requires a passed full-graph throughput probe')
    data = _load_graph(args.data_pt, mmap=True)
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.shape[1])
    if (
        receipt.get('num_nodes') != num_nodes
        or receipt.get('num_edges') != num_edges
        or num_nodes != FULL_GRAPH_NODE_COUNT
        or num_edges != FULL_GRAPH_EDGE_COUNT
    ):
        raise ValueError('RegressionLP probe dimensions differ from the loaded graph')
    labels = _load_targets(
        args.targets_pt, args.partition_jsonl, ('fit', 'select'), num_nodes
    )
    fit_ids = _read_partition_role(args.partition_jsonl, 'fit')
    select_ids = _read_partition_role(args.partition_jsonl, 'select')
    if receipt.get('fit_seed_count') != int(fit_ids.numel()):
        raise ValueError('RegressionLP probe fit seed count differs from the split')
    torch.set_num_threads(args.threads)
    selection = select_label_propagation(
        data.edge_index,
        num_nodes=num_nodes,
        labels=labels,
        fit_node_ids=fit_ids,
        select_node_ids=select_ids,
        chunk_size=args.chunk_size,
    )
    reachability = {
        role: {
            'reached': int(selection.reached[_read_partition_role(args.partition_jsonl, role)].sum()),
            'fallback': int((~selection.reached[_read_partition_role(args.partition_jsonl, role)]).sum()),
        }
        for role in ('fit', 'select', 'calibrate', 'test')
    }
    cache_path = args.run_dir / 'cache' / 'RegressionLP' / f'{DETERMINISTIC_RUN_ID}.pt'
    if cache_path.exists():
        raise FileExistsError(f'refusing to overwrite LP cache: {cache_path}')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'schema_version': 'clean-uq-lp-cache-v1',
            'full_graph': True,
            'directed': True,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'fit_seed_count': int(fit_ids.numel()),
            'estimand': LP_ESTIMAND,
            'chunk_size': args.chunk_size,
            'predictions': selection.predictions,
            'reached': selection.reached,
        },
        cache_path,
    )
    state_path = _state_path(args.run_dir, 'RegressionLP')
    state = {
        'schema_version': 'clean-uq-baseline-state-v1',
        'arm': 'RegressionLP',
        'seed': DETERMINISTIC_RUN_ID,
        'full_graph': True,
        'directed': True,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'estimand': LP_ESTIMAND,
        'chunk_size': args.chunk_size,
        'deterministic_algorithms': True,
        'execution_device': 'cpu',
        'grid': _lp_grid_payload(),
        'graph_passes': LP_GRAPH_PASSES,
        'fit_seed_count': int(fit_ids.numel()),
        'fit_median': selection.fit_median,
        'selected_damping': selection.damping,
        'selected_iterations': selection.iterations,
        'select_mae': selection.select_mae,
        'cache': str(cache_path),
        'reached_count': int(selection.reached.sum()),
        'fallback_count': int((~selection.reached).sum()),
        'partition_reachability': reachability,
        'prediction_kind': 'point-only',
        'methods': ['simple'],
    }
    _write_json_once(state_path, state)
    return {**state, 'state_path': str(state_path)}


def _load_xgboost_models(state: dict[str, Any]) -> tuple[XGBoostEstimator, XGBoostEstimator]:
    import xgboost

    if xgboost.__version__ != state.get('xgboost_version'):
        raise RuntimeError('XGBoost prediction environment differs from fit')
    midpoint = xgboost.XGBRegressor()
    quantile = xgboost.XGBRegressor()
    midpoint.load_model(state['midpoint_model'])
    quantile.load_model(state['quantile_model'])
    return midpoint, quantile


def _load_baseline_state(run_dir: Path, arm: str, num_nodes: int) -> dict[str, Any]:
    state_path = _state_path(run_dir, arm)
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if (
        not isinstance(state, dict)
        or state.get('schema_version') != 'clean-uq-baseline-state-v1'
        or state.get('arm') != arm
        or state.get('seed') != DETERMINISTIC_RUN_ID
    ):
        raise ValueError('baseline state identity mismatch')
    expected_methods = ['simple', 'cqr'] if arm == 'XGBoost' else ['simple']
    expected_kind = 'point-with-endpoints' if arm == 'XGBoost' else 'point-only'
    if state.get('methods') != expected_methods or state.get('prediction_kind') != expected_kind:
        raise ValueError('baseline state prediction contract mismatch')
    if arm == 'GlobalMedian':
        median = state.get('median')
        if (
            isinstance(median, bool)
            or not isinstance(median, (int, float))
            or not 0.0 <= float(median) <= 1.0
        ):
            raise ValueError('GlobalMedian state has an invalid fit constant')
    elif arm == 'XGBoost':
        midpoint_path = run_dir / 'checkpoints' / arm / 'midpoint.ubj'
        quantile_path = run_dir / 'checkpoints' / arm / 'quantiles.ubj'
        if (
            state.get('xgboost_distribution') != 'xgboost-cpu'
            or state.get('xgboost_version') != XGBOOST_VERSION
            or state.get('deterministic') is not True
            or Path(str(state.get('midpoint_model'))).resolve() != midpoint_path.resolve()
            or Path(str(state.get('quantile_model'))).resolve() != quantile_path.resolve()
            or not midpoint_path.is_file()
            or not quantile_path.is_file()
        ):
            raise ValueError('XGBoost state is not bound to the selected run-local models')
    else:
        expected_cache = run_dir / 'cache' / arm / f'{DETERMINISTIC_RUN_ID}.pt'
        if (
            state.get('full_graph') is not True
            or state.get('directed') is not True
            or state.get('num_nodes') != num_nodes
            or state.get('estimand') != LP_ESTIMAND
            or state.get('deterministic_algorithms') is not True
            or state.get('execution_device') != 'cpu'
            or Path(str(state.get('cache'))).resolve() != expected_cache.resolve()
        ):
            raise ValueError('RegressionLP state is not bound to the deterministic full graph')
    return state


def _run_predict_role(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_baseline_state(args.run_dir, args.arm, args.num_nodes)
    node_ids = _read_partition_role(args.partition_jsonl, args.role)
    expected_roles = (args.role,)
    labels = _load_targets(
        args.targets_pt, args.partition_jsonl, expected_roles, args.num_nodes
    )
    if args.arm == 'GlobalMedian':
        full_predictions = torch.full((args.num_nodes,), float(state['median']))
    elif args.arm == 'RegressionLP':
        cache_path = args.run_dir / 'cache' / args.arm / f'{DETERMINISTIC_RUN_ID}.pt'
        cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
        predictions = cache.get('predictions') if isinstance(cache, dict) else None
        reached = cache.get('reached') if isinstance(cache, dict) else None
        if (
            not isinstance(cache, dict)
            or cache.get('schema_version') != 'clean-uq-lp-cache-v1'
            or cache.get('full_graph') is not True
            or cache.get('directed') is not True
            or cache.get('num_nodes') != args.num_nodes
            or cache.get('estimand') != LP_ESTIMAND
            or not isinstance(predictions, torch.Tensor)
            or predictions.shape != (args.num_nodes,)
            or not bool(torch.isfinite(predictions).all())
            or not isinstance(reached, torch.Tensor)
            or reached.dtype != torch.bool
            or reached.shape != (args.num_nodes,)
        ):
            raise ValueError('not a deterministic RegressionLP full-graph cache')
        full_predictions = predictions
    else:
        if args.data_pt is None:
            raise ValueError('XGBoost prediction requires --data-pt')
        data = _load_graph(args.data_pt, mmap=True)
        if int(data.num_nodes) != args.num_nodes:
            raise ValueError('XGBoost graph differs from the frozen target node axis')
        midpoint, quantile = _load_xgboost_models(state)
        role_predictions = predict_xgboost(midpoint, quantile, data.x, node_ids)
        full_predictions = torch.empty((args.num_nodes, 3))
        full_predictions[node_ids] = role_predictions
    output_path = args.output_pt or _prediction_path(
        args.run_dir, args.arm, args.role
    )
    payload = write_baseline_role_predictions(
        full_predictions,
        node_ids,
        labels,
        output_path,
        role=args.role,
        arm=args.arm,
    )
    return {
        'stage': 'predict-role',
        'arm': args.arm,
        'role': args.role,
        'seed': DETERMINISTIC_RUN_ID,
        'predictions': str(output_path),
        'methods': payload['uq_methods'],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('median-fit', 'xgboost-fit', 'lp-probe', 'lp-fit'):
        stage = stages.add_parser(name)
        stage.add_argument('--run-dir', type=Path, required=True)
        stage.add_argument('--targets-pt', type=Path, required=True)
        stage.add_argument('--partition-jsonl', type=Path, required=True)
        if name == 'median-fit':
            stage.add_argument('--num-nodes', type=int, required=True)
        else:
            stage.add_argument('--data-pt', type=Path, required=True)
        if name in ('lp-probe', 'lp-fit'):
            stage.add_argument('--chunk-size', type=int, default=8_388_608)
            stage.add_argument('--threads', type=int, default=16)
        if name == 'lp-probe':
            stage.add_argument('--passes', type=int, default=2)
            stage.add_argument('--max-projected-hours', type=float, default=20.0)
    predict = stages.add_parser('predict-role')
    predict.add_argument('--run-dir', type=Path, required=True)
    predict.add_argument('--targets-pt', type=Path, required=True)
    predict.add_argument('--partition-jsonl', type=Path, required=True)
    predict.add_argument('--num-nodes', type=int, required=True)
    predict.add_argument('--data-pt', type=Path)
    predict.add_argument('--role', choices=('calibrate', 'test'), required=True)
    predict.add_argument(
        '--arm', choices=('GlobalMedian', 'XGBoost', 'RegressionLP'), required=True
    )
    predict.add_argument('--output-pt', type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    runners = {
        'median-fit': _run_median_fit,
        'xgboost-fit': _run_xgboost_fit,
        'lp-probe': _run_lp_probe,
        'lp-fit': _run_lp_fit,
        'predict-role': _run_predict_role,
    }
    result = runners[args.stage](args)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
