"""Leakage-free three-head training stages for the clean graph-UQ rerun."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from credipred.experiments.gnn_experiments.clean_uq_protocol import (
    load_role_targets,
)
from credipred.utils.checkpoint import snapshot_state_dict


QUANTILE_ALPHA = 0.05
SUPPORTED_MODELS = ('FF', 'GCN', 'SAGE', 'GAT')


@dataclass(frozen=True)
class TrainingRoles:
    """The complete role surface available to an optimization loop."""

    fit: torch.Tensor
    select: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    state_dict: dict[str, torch.Tensor]
    select_loss: float
    best_epoch: int


def _partition_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError('clean-UQ partition rows must be JSON objects')
        rows.append(raw)
    return rows


def _role_node_ids(rows: Iterable[dict[str, Any]], role: str) -> torch.Tensor:
    node_ids: list[int] = []
    for row in rows:
        if row.get('clean_uq_split') != role:
            continue
        node_id = row.get('node_id')
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
            raise ValueError('clean-UQ rows require nonnegative integer node_id')
        if row.get('mapped') is not True or row.get('labelled') is not True:
            raise ValueError(f'{role} contains an ineligible row')
        node_ids.append(node_id)
    if not node_ids:
        raise ValueError(f'clean-UQ role {role!r} is empty')
    return torch.tensor(node_ids, dtype=torch.long)


def load_training_roles(path: Path) -> TrainingRoles:
    """Load only the two roles authorized for optimization and selection."""
    rows = _partition_rows(path)
    return TrainingRoles(
        fit=_role_node_ids(rows, 'fit'),
        select=_role_node_ids(rows, 'select'),
    )


def _quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = QUANTILE_ALPHA,
) -> torch.Tensor:
    """Pinball objective for columns [midpoint, lower, upper]."""
    quantiles = predictions.new_tensor((0.5, alpha, 1.0 - alpha))
    residual = targets[:, None] - predictions
    return torch.maximum(quantiles * residual, (quantiles - 1.0) * residual).mean(0).sum()


def _loader_loss(
    model: torch.nn.Module,
    loader: Iterable[Any],
    labels: torch.Tensor,
    predict_batch: Callable[[torch.nn.Module, Any], torch.Tensor],
    alpha: float,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()  # type: ignore[no-untyped-call]
    with context:
        for batch in loader:
            batch = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            predictions = predict_batch(model, batch)[: batch.batch_size]
            seed_nodes = batch.n_id[: batch.batch_size].detach().cpu()
            targets = labels[seed_nodes].to(device)
            loss = _quantile_loss(predictions, targets, alpha)
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            total += float(loss.detach()) * batch.batch_size
            count += batch.batch_size
    if count == 0:
        raise ValueError('training loader yielded no seed nodes')
    return total / count


def fit_with_selection(
    model: torch.nn.Module,
    fit_loader: Iterable[Any],
    select_loader: Iterable[Any],
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    predict_batch: Callable[[torch.nn.Module, Any], torch.Tensor],
    alpha: float = QUANTILE_ALPHA,
) -> TrainingResult:
    """Optimize on fit and checkpoint solely by select pinball loss."""
    if epochs < 1:
        raise ValueError('epochs must be positive')
    best_loss = float('inf')
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        _loader_loss(model, fit_loader, labels, predict_batch, alpha, optimizer)
        select_loss = _loader_loss(
            model,
            select_loader,
            labels,
            predict_batch,
            alpha,
            optimizer=None,
        )
        if select_loss < best_loss:
            best_loss = select_loss
            best_epoch = epoch
            best_state = snapshot_state_dict(model)
    if best_state is None:
        raise RuntimeError('training did not produce a checkpoint')
    return TrainingResult(best_state, best_loss, best_epoch)


def write_role_predictions(
    model: torch.nn.Module,
    loader: Iterable[Any],
    labels: torch.Tensor,
    predict_batch: Callable[[torch.nn.Module, Any], torch.Tensor],
    output_path: Path,
    *,
    role: str,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    """Write aligned seed-node predictions and external labels for one role."""
    model.eval()
    device = next(model.parameters()).device
    node_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_ids = batch.n_id[: batch.batch_size].detach().cpu()
            prediction_parts.append(
                predict_batch(model, batch)[: batch.batch_size].detach().cpu()
            )
            node_parts.append(node_ids)
    if not node_parts:
        raise ValueError(f'{role} loader yielded no seed nodes')
    node_ids = torch.cat(node_parts)
    payload: dict[str, Any] = {
        'schema_version': 'clean-uq-predictions-v1',
        'role': role,
        'seed': seed,
        'arm': arm,
        'node_ids': node_ids,
        'predictions': torch.cat(prediction_parts),
        'labels': labels[node_ids].detach().cpu(),
        'uq_methods': ['simple', 'cqr'],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite predictions: {output_path}')
    torch.save(payload, output_path)
    return payload


def write_cached_role_predictions(
    parent_predictions: torch.Tensor,
    node_ids: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    *,
    role: str,
    seed: int,
) -> dict[str, Any]:
    """Write one GAT role as an exact slice of its frozen parent cache."""
    if parent_predictions.ndim != 2 or parent_predictions.shape[1] != 3:
        raise ValueError('parent predictions must have shape [N, 3]')
    if labels.ndim != 1 or labels.numel() != parent_predictions.shape[0]:
        raise ValueError('labels must align with the frozen parent cache')
    payload: dict[str, Any] = {
        'schema_version': 'clean-uq-predictions-v1',
        'role': role,
        'seed': seed,
        'arm': 'GAT',
        'node_ids': node_ids.detach().cpu(),
        'predictions': parent_predictions[node_ids].detach().cpu(),
        'labels': labels[node_ids].detach().cpu(),
        'uq_methods': ['simple', 'cqr'],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite predictions: {output_path}')
    torch.save(payload, output_path)
    return payload


def _load_graph(path: Path) -> Any:
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded[0]
    return loaded


def _load_labels(
    path: Path,
    num_nodes: int,
    partition_jsonl: Path,
    expected_roles: tuple[str, ...],
) -> torch.Tensor:
    """Load a role-masked artifact and reject any unauthorized finite target."""
    return load_role_targets(
        path,
        partition_jsonl=partition_jsonl,
        expected_roles=expected_roles,
        num_nodes=num_nodes,
    )


def _validate_role_labels(labels: torch.Tensor, node_ids: torch.Tensor, role: str) -> None:
    if int(node_ids.max()) >= labels.numel():
        raise ValueError(f'{role} node_id exceeds labels.pt')
    if bool((labels[node_ids] < 0).any()):
        raise ValueError(f'{role} contains an unlabeled external target')


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str | None) -> torch.device:
    return torch.device(name or ('cuda:0' if torch.cuda.is_available() else 'cpu'))


def _loader(
    data: Any,
    nodes: torch.Tensor,
    *,
    shuffle: bool,
    batch_size: int,
    neighbor_k: int,
    num_layers: int,
    num_workers: int,
) -> Any:
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(
        data,
        input_nodes=nodes,
        num_neighbors=[neighbor_k] * num_layers,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _base_checkpoint_path(run_dir: Path, model: str, seed: int) -> Path:
    return run_dir / 'checkpoints' / model / f'seed-{seed}.pt'


def _parent_cache_path(run_dir: Path, seed: int) -> Path:
    return run_dir / 'cache' / 'GAT' / f'seed-{seed}.pt'


def _correction_checkpoint_path(run_dir: Path, arm: str, seed: int) -> Path:
    return run_dir / 'checkpoints' / f'GAT-{arm}' / f'seed-{seed}.pt'


def _prediction_path(run_dir: Path, arm: str, seed: int, role: str) -> Path:
    return run_dir / 'predictions' / arm / f'seed-{seed}' / f'{role}.pt'


def _base_predict(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    return cast(torch.Tensor, model(batch.x, batch.edge_index))


def _correction_predict(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    base_predictions = batch.base_preds
    return cast(torch.Tensor, base_predictions + model(base_predictions, batch.edge_index))


def _build_base_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from credipred.gnn.model import Model

    return Model(
        model_name=config['model'],
        normalization=config['normalization'],
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['embedding_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        binary=False,
        prediction_dim=3,
    ).to(device)


def _build_correction_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if config['arm'] == 'mlp':
        from credipred.conformal_regression.correction_mlp import CorrectionMLP

        model: torch.nn.Module = CorrectionMLP(
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
        )
    else:
        from credipred.conformal_regression.correction_gnn import CorrectionGNN

        model = CorrectionGNN(
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            normalization=config['normalization'],
        )
    return model.to(device)


def _save_checkpoint(
    path: Path,
    *,
    kind: str,
    arm: str,
    seed: int,
    config: dict[str, Any],
    result: TrainingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite checkpoint: {path}')
    torch.save(
        {
            'schema_version': 'clean-uq-checkpoint-v1',
            'kind': kind,
            'arm': arm,
            'seed': seed,
            'config': config,
            'state_dict': result.state_dict,
            'select_pinball_loss': result.select_loss,
            'best_epoch': result.best_epoch,
        },
        path,
    )


def _load_checkpoint(path: Path, *, expected_kind: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 'clean-uq-checkpoint-v1':
        raise ValueError('not a clean-UQ checkpoint')
    if checkpoint.get('kind') != expected_kind:
        raise ValueError(f'expected a {expected_kind} checkpoint')
    return checkpoint


def _run_base_fit(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    data = _load_graph(args.data_pt)
    labels = _load_labels(
        args.targets_pt,
        int(data.num_nodes),
        args.partition_jsonl,
        ('fit', 'select'),
    )
    roles = load_training_roles(args.partition_jsonl)
    _validate_role_labels(labels, roles.fit, 'fit')
    _validate_role_labels(labels, roles.select, 'select')
    config = {
        'model': args.model,
        'normalization': args.normalization,
        'in_channels': int(data.num_features),
        'hidden_channels': args.hidden_channels,
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
    }
    device = _device(args.device)
    model = _build_base_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    fit_loader = _loader(
        data, roles.fit, shuffle=True, batch_size=args.batch_size,
        neighbor_k=args.neighbor_k, num_layers=args.num_layers,
        num_workers=args.num_workers,
    )
    select_loader = _loader(
        data, roles.select, shuffle=False, batch_size=args.batch_size,
        neighbor_k=args.neighbor_k, num_layers=args.num_layers,
        num_workers=args.num_workers,
    )
    result = fit_with_selection(
        model, fit_loader, select_loader, labels, optimizer, args.epochs,
        _base_predict, QUANTILE_ALPHA,
    )
    checkpoint_path = _base_checkpoint_path(args.run_dir, args.model, args.seed)
    _save_checkpoint(
        checkpoint_path, kind='base', arm=args.model, seed=args.seed,
        config=config, result=result,
    )
    return {
        'stage': 'base-fit',
        'arm': args.model,
        'seed': args.seed,
        'checkpoint': str(checkpoint_path),
        'best_epoch': result.best_epoch,
        'select_pinball_loss': result.select_loss,
    }


def _full_graph_predictions(
    model: torch.nn.Module,
    loader: Iterable[Any],
    num_nodes: int,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    predictions = torch.empty(num_nodes, 3)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_ids = batch.n_id[: batch.batch_size].detach().cpu()
            predictions[node_ids] = _base_predict(model, batch)[: batch.batch_size].cpu()
    return predictions


def _run_parent_cache(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    data = _load_graph(args.data_pt)
    checkpoint_path = args.checkpoint_pt or _base_checkpoint_path(
        args.run_dir, 'GAT', args.seed
    )
    checkpoint = _load_checkpoint(checkpoint_path, expected_kind='base')
    if checkpoint.get('arm') != 'GAT' or checkpoint.get('seed') != args.seed:
        raise ValueError('parent cache requires the matching frozen GAT checkpoint')
    config = checkpoint['config']
    device = _device(args.device)
    model = _build_base_model(config, device)
    model.load_state_dict(checkpoint['state_dict'])
    loader = _loader(
        data, torch.arange(int(data.num_nodes)), shuffle=False,
        batch_size=args.batch_size, neighbor_k=args.neighbor_k,
        num_layers=config['num_layers'], num_workers=args.num_workers,
    )
    output_path = args.output_pt or _parent_cache_path(args.run_dir, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite parent cache: {output_path}')
    torch.save(
        {
            'schema_version': 'clean-uq-parent-cache-v1',
            'seed': args.seed,
            'arm': 'GAT',
            'predictions': _full_graph_predictions(model, loader, int(data.num_nodes)),
        },
        output_path,
    )
    return {'stage': 'parent-cache', 'seed': args.seed, 'cache': str(output_path)}


def _load_parent_cache(path: Path, seed: int, num_nodes: int) -> torch.Tensor:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 'clean-uq-parent-cache-v1':
        raise ValueError('not a clean-UQ parent cache')
    predictions = payload.get('predictions')
    if payload.get('seed') != seed or payload.get('arm') != 'GAT':
        raise ValueError('parent cache identifies a different GAT seed')
    if not isinstance(predictions, torch.Tensor) or tuple(predictions.shape) != (num_nodes, 3):
        raise ValueError('parent cache has the wrong prediction shape')
    return predictions


def _run_correction_fit(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    data = _load_graph(args.data_pt)
    labels = _load_labels(
        args.targets_pt,
        int(data.num_nodes),
        args.partition_jsonl,
        ('fit', 'select'),
    )
    roles = load_training_roles(args.partition_jsonl)
    _validate_role_labels(labels, roles.fit, 'fit')
    _validate_role_labels(labels, roles.select, 'select')
    cache_path = args.parent_cache_pt or _parent_cache_path(args.run_dir, args.seed)
    data.base_preds = _load_parent_cache(cache_path, args.seed, int(data.num_nodes))
    config = {
        'arm': args.arm,
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'normalization': args.normalization,
    }
    device = _device(args.device)
    model = _build_correction_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    fit_loader = _loader(
        data, roles.fit, shuffle=True, batch_size=args.batch_size,
        neighbor_k=args.neighbor_k, num_layers=args.num_layers,
        num_workers=args.num_workers,
    )
    select_loader = _loader(
        data, roles.select, shuffle=False, batch_size=args.batch_size,
        neighbor_k=args.neighbor_k, num_layers=args.num_layers,
        num_workers=args.num_workers,
    )
    result = fit_with_selection(
        model, fit_loader, select_loader, labels, optimizer, args.epochs,
        _correction_predict, QUANTILE_ALPHA,
    )
    checkpoint_path = _correction_checkpoint_path(args.run_dir, args.arm, args.seed)
    _save_checkpoint(
        checkpoint_path, kind='correction', arm=f'GAT-{args.arm}', seed=args.seed,
        config=config, result=result,
    )
    return {
        'stage': 'correction-fit',
        'arm': f'GAT-{args.arm}',
        'seed': args.seed,
        'checkpoint': str(checkpoint_path),
        'best_epoch': result.best_epoch,
        'select_pinball_loss': result.select_loss,
    }


def _run_predict_role(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    rows = _partition_rows(args.partition_jsonl)
    node_ids = _role_node_ids(rows, args.role)
    data = _load_graph(args.data_pt)
    labels = _load_labels(
        args.targets_pt,
        int(data.num_nodes),
        args.partition_jsonl,
        (args.role,),
    )
    _validate_role_labels(labels, node_ids, args.role)
    output_path = args.output_pt or _prediction_path(
        args.run_dir, args.arm, args.seed, args.role
    )
    if args.arm == 'GAT':
        cache_path = args.parent_cache_pt or _parent_cache_path(args.run_dir, args.seed)
        parent_predictions = _load_parent_cache(
            cache_path, args.seed, int(data.num_nodes)
        )
        payload = write_cached_role_predictions(
            parent_predictions,
            node_ids,
            labels,
            output_path,
            role=args.role,
            seed=args.seed,
        )
        return {
            'stage': 'predict-role',
            'arm': args.arm,
            'seed': args.seed,
            'role': args.role,
            'count': int(payload['labels'].numel()),
            'predictions': str(output_path),
            'source': 'frozen-parent-cache',
        }
    device = _device(args.device)
    if args.arm in SUPPORTED_MODELS:
        checkpoint_path = args.checkpoint_pt or _base_checkpoint_path(
            args.run_dir, args.arm, args.seed
        )
        checkpoint = _load_checkpoint(checkpoint_path, expected_kind='base')
        if checkpoint.get('arm') != args.arm or checkpoint.get('seed') != args.seed:
            raise ValueError('base checkpoint identifies a different arm or seed')
        config = checkpoint['config']
        model = _build_base_model(config, device)
        predict_batch = _base_predict
    else:
        correction_arm = args.arm.removeprefix('GAT-')
        checkpoint_path = args.checkpoint_pt or _correction_checkpoint_path(
            args.run_dir, correction_arm, args.seed
        )
        checkpoint = _load_checkpoint(checkpoint_path, expected_kind='correction')
        if checkpoint.get('arm') != args.arm or checkpoint.get('seed') != args.seed:
            raise ValueError('correction checkpoint identifies a different arm or seed')
        config = checkpoint['config']
        cache_path = args.parent_cache_pt or _parent_cache_path(args.run_dir, args.seed)
        data.base_preds = _load_parent_cache(cache_path, args.seed, int(data.num_nodes))
        model = _build_correction_model(config, device)
        predict_batch = _correction_predict
    model.load_state_dict(checkpoint['state_dict'])
    loader = _loader(
        data, node_ids, shuffle=False, batch_size=args.batch_size,
        neighbor_k=args.neighbor_k, num_layers=config['num_layers'],
        num_workers=args.num_workers,
    )
    payload = write_role_predictions(
        model, loader, labels, predict_batch, output_path,
        role=args.role, seed=args.seed, arm=args.arm,
    )
    return {
        'stage': 'predict-role',
        'arm': args.arm,
        'seed': args.seed,
        'role': args.role,
        'count': int(payload['labels'].numel()),
        'predictions': str(output_path),
    }


def _add_common_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--neighbor-k', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)

    base = stages.add_parser('base-fit')
    _add_common_stage_arguments(base)
    base.add_argument('--data-pt', type=Path, required=True)
    base.add_argument('--targets-pt', type=Path, required=True)
    base.add_argument('--partition-jsonl', type=Path, required=True)
    base.add_argument('--model', choices=SUPPORTED_MODELS, required=True)
    base.add_argument('--epochs', type=int, default=200)
    base.add_argument('--lr', type=float, default=0.001)
    base.add_argument('--weight-decay', type=float, default=2.36e-5)
    base.add_argument('--hidden-channels', type=int, default=256)
    base.add_argument('--embedding-dim', type=int, default=128)
    base.add_argument('--num-layers', type=int, default=3)
    base.add_argument('--dropout', type=float, default=0.1)
    base.add_argument('--normalization', default='BatchNorm')

    parent = stages.add_parser('parent-cache')
    _add_common_stage_arguments(parent)
    parent.add_argument('--data-pt', type=Path, required=True)
    parent.add_argument('--checkpoint-pt', type=Path)
    parent.add_argument('--output-pt', type=Path)

    correction = stages.add_parser('correction-fit')
    _add_common_stage_arguments(correction)
    correction.add_argument('--data-pt', type=Path, required=True)
    correction.add_argument('--targets-pt', type=Path, required=True)
    correction.add_argument('--partition-jsonl', type=Path, required=True)
    correction.add_argument('--parent-cache-pt', type=Path)
    correction.add_argument('--arm', choices=('mlp', 'topology'), required=True)
    correction.add_argument('--epochs', type=int, default=200)
    correction.add_argument('--lr', type=float, default=0.001)
    correction.add_argument('--weight-decay', type=float, default=2.36e-5)
    correction.add_argument('--hidden-channels', type=int, default=64)
    correction.add_argument('--num-layers', type=int, default=2)
    correction.add_argument('--dropout', type=float, default=0.1)
    correction.add_argument('--normalization', default='BatchNorm')

    predict = stages.add_parser('predict-role')
    _add_common_stage_arguments(predict)
    predict.add_argument('--data-pt', type=Path, required=True)
    predict.add_argument('--targets-pt', type=Path, required=True)
    predict.add_argument('--partition-jsonl', type=Path, required=True)
    predict.add_argument('--role', choices=('calibrate', 'test'), required=True)
    predict.add_argument('--arm', choices=(*SUPPORTED_MODELS, 'GAT-mlp', 'GAT-topology'), required=True)
    predict.add_argument('--checkpoint-pt', type=Path)
    predict.add_argument('--parent-cache-pt', type=Path)
    predict.add_argument('--output-pt', type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    runners = {
        'base-fit': _run_base_fit,
        'parent-cache': _run_parent_cache,
        'correction-fit': _run_correction_fit,
        'predict-role': _run_predict_role,
    }
    result = runners[args.stage](args)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
