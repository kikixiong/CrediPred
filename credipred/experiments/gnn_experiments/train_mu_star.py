"""Train a base-fit-only midpoint GAT and export full-graph predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError('mu-star partition rows must be JSON objects')
        rows.append({str(key): value for key, value in raw.items()})
    return rows


def _base_fit_groups_digest(rows: list[dict[str, object]]) -> str:
    groups = sorted(
        {
            str(row['group_id'])
            for row in rows
            if row.get('quality_split') == 'base_fit'
        }
    )
    return hashlib.sha256(''.join(f'{group}\n' for group in groups).encode()).hexdigest()


def _node_id(row: Mapping[str, object]) -> int:
    value = row.get('node_id')
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('mu-star rows require nonnegative integer node_id')
    return value


def _internal_base_split(rows: list[dict[str, object]], seed: int) -> tuple[list[int], list[int]]:
    base_rows = [row for row in rows if row.get('quality_split') == 'base_fit']
    groups = sorted(
        {str(row['group_id']) for row in base_rows},
        key=lambda group: hashlib.sha256(f'mu-star-internal:{seed}:{group}'.encode()).hexdigest(),
    )
    if len(groups) < 5:
        raise ValueError('mu-star base_fit requires at least five independent groups')
    validation_count = max(1, len(groups) // 5)
    validation_groups = set(groups[:validation_count])
    train = [
        _node_id(row)
        for row in base_rows
        if str(row['group_id']) not in validation_groups
    ]
    validation = [
        _node_id(row)
        for row in base_rows
        if str(row['group_id']) in validation_groups
    ]
    if not train or not validation:
        raise ValueError('mu-star internal base split is empty')
    return train, validation


def _load_graph(path: Path) -> Any:
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded[0]
    return loaded


def _validate_mapping(
    rows: list[dict[str, object]], mapping: Mapping[object, object], num_nodes: int
) -> None:
    for row in rows:
        domain = str(row['canonical_domain'])
        node_id = _node_id(row)
        mapping_key = '.'.join(reversed(domain.split('.')))
        if node_id < 0 or node_id >= num_nodes or mapping.get(mapping_key) != node_id:
            raise ValueError(f'partition/mapping mismatch for {domain}')


def _loader(
    data: Any,
    nodes: list[int] | torch.Tensor,
    *,
    shuffle: bool,
    batch_size: int,
    neighbor_k: int = 30,
    num_workers: int = 4,
) -> Any:
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(
        data,
        input_nodes=torch.as_tensor(nodes, dtype=torch.long),
        num_neighbors=[neighbor_k, neighbor_k, neighbor_k],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _validation_mae(
    model: torch.nn.Module,
    loader: Any,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index)[: batch.batch_size, 0]
            seed_nodes = batch.n_id[: batch.batch_size].detach().cpu()
            targets = labels[seed_nodes].to(device)
            total += F.l1_loss(predictions, targets, reduction='sum').item()
            count += batch.batch_size
    return total / count


def train_mu_star(
    data_path: Path,
    mapping_path: Path,
    partition_path: Path,
    plan_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    seed: int = 42,
    neighbor_k: int = 30,
    num_workers: int = 4,
) -> dict[str, object]:
    """Train only on base-fit labels; head cohorts are never evaluated or selected on."""
    if output_dir.exists():
        raise FileExistsError(f'refusing to reuse mu-star output: {output_dir}')
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch size must be positive')
    plan_raw = json.loads(plan_path.read_text(encoding='utf-8'))
    if not isinstance(plan_raw, dict):
        raise ValueError('mu-star plan must be a JSON object')
    plan = {str(key): value for key, value in plan_raw.items()}
    if plan.get('partition_sha256') != _sha256(partition_path):
        raise ValueError('mu-star partition file does not match its frozen plan')
    if plan.get('training_partitions') != ['base_fit']:
        raise ValueError('mu-star plan must authorize only base_fit')
    if plan.get('training_status') != 'approved_verified_graph_and_base_fit_trainer':
        raise ValueError('mu-star plan has not approved graph data and training')
    if plan.get('data_sha256') != _sha256(data_path):
        raise ValueError('data.pt does not match the approved plan')
    if plan.get('mapping_sha256') != _sha256(mapping_path):
        raise ValueError('mapping.pt does not match the approved plan')
    if plan.get('labels_sha256') != _sha256(labels_path):
        raise ValueError('labels.pt does not match the approved plan')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = _jsonl(partition_path)
    data = _load_graph(data_path)
    labels = torch.load(labels_path, map_location='cpu', weights_only=True)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError('labels.pt must be a one-dimensional tensor')
    if int(labels.shape[0]) != int(data.num_nodes):
        raise ValueError('labels.pt length must equal graph node count')
    mapping_raw = torch.load(mapping_path, map_location='cpu', weights_only=False)
    if not isinstance(mapping_raw, Mapping):
        raise ValueError('graph mapping must be a domain-to-node mapping')
    _validate_mapping(rows, mapping_raw, int(data.num_nodes))
    train_nodes, validation_nodes = _internal_base_split(rows, seed)
    labeled_nodes = train_nodes + validation_nodes
    if bool((labels[torch.as_tensor(labeled_nodes)] < 0).any()):
        raise ValueError('base_fit contains an unlabeled external label')

    from credipred.gnn.model import Model

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(
        model_name='GAT',
        normalization='BatchNorm',
        in_channels=int(data.num_features),
        hidden_channels=256,
        out_channels=128,
        num_layers=3,
        dropout=0.1,
        binary=False,
        prediction_dim=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2.36e-5)
    train_loader = _loader(
        data, train_nodes, shuffle=True, batch_size=batch_size,
        neighbor_k=neighbor_k, num_workers=num_workers,
    )
    validation_loader = _loader(
        data, validation_nodes, shuffle=False, batch_size=batch_size,
        neighbor_k=neighbor_k, num_workers=num_workers,
    )
    best_loss = float('inf')
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch.x, batch.edge_index)[: batch.batch_size, 0]
            seed_nodes = batch.n_id[: batch.batch_size].detach().cpu()
            targets = labels[seed_nodes].to(device)
            loss = F.l1_loss(prediction, targets)
            loss.backward()
            optimizer.step()
        validation_loss = _validation_mae(model, validation_loader, labels, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError('mu-star training did not produce a checkpoint')

    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / 'mu_star_checkpoint.pt'
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    model.eval()
    prediction = torch.empty(int(data.num_nodes), dtype=torch.float32)
    inference_loader = _loader(
        data, torch.arange(int(data.num_nodes)), shuffle=False, batch_size=batch_size,
        neighbor_k=neighbor_k, num_workers=num_workers,
    )
    with torch.no_grad():
        for batch in inference_loader:
            seed_nodes = batch.n_id[: batch.batch_size]
            batch = batch.to(device)
            values = model(batch.x, batch.edge_index)[: batch.batch_size, 0]
            prediction[seed_nodes] = values.detach().cpu().clamp(0.0, 1.0)
    prediction_path = output_dir / 'mu_star_midpoint.pt'
    torch.save(prediction, prediction_path)

    checkpoint_sha256 = _sha256(checkpoint_path)
    lineage = {
        'schema_version': 'mu-star-base-fit-v1',
        'scientific_eligibility': 'formal',
        'trained_partition': 'base_fit',
        'base_fit_only': True,
        'group_disjoint': True,
        'mu_star_id': checkpoint_sha256,
        'official_group_registry_sha256': plan.get('official_group_registry_sha256'),
        'quality_partition_registry_sha256': _sha256(partition_path),
        'base_fit_group_sha256': _base_fit_groups_digest(rows),
        'checkpoint_sha256': checkpoint_sha256,
        'prediction_sha256': _sha256(prediction_path),
        'data_sha256': _sha256(data_path),
        'mapping_sha256': _sha256(mapping_path),
        'labels_sha256': _sha256(labels_path),
        'base_fit_train_nodes': len(train_nodes),
        'base_fit_validation_nodes': len(validation_nodes),
        'best_base_fit_validation_mae': best_loss,
        'head_labels_used': False,
    }
    (output_dir / 'mu_star_lineage.json').write_text(
        json.dumps(lineage, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    return lineage


def main() -> None:
    """Run base-fit-only training and full-graph midpoint export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-pt', type=Path, required=True)
    parser.add_argument('--mapping-pt', type=Path, required=True)
    parser.add_argument('--partition-jsonl', type=Path, required=True)
    parser.add_argument('--plan-json', type=Path, required=True)
    parser.add_argument('--labels-pt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--neighbor-k', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    lineage = train_mu_star(
        args.data_pt,
        args.mapping_pt,
        args.partition_jsonl,
        args.plan_json,
        args.labels_pt,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        neighbor_k=args.neighbor_k,
        num_workers=args.num_workers,
    )
    print(json.dumps(lineage, sort_keys=True))


if __name__ == '__main__':
    main()
