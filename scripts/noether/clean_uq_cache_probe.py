#!/usr/bin/env python3
"""Measure real-graph GAT parent-cache throughput without writing a partial cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from credipred.experiments.gnn_experiments.clean_uq_train import (
    _base_checkpoint_path,
    _base_predict,
    _build_base_model,
    _device,
    _load_checkpoint,
    _load_graph,
    _loader,
)


def _projection_report(
    *,
    seed: int,
    processed: int,
    elapsed: float,
    graph_nodes: int,
    batch_size: int,
    neighbor_k: int,
    max_projected_hours: float,
) -> dict[str, float | int | bool | str]:
    seeds_per_second = processed / elapsed
    projected_hours = graph_nodes / seeds_per_second / 3600.0
    return {
        'stage': 'parent-cache-throughput-probe',
        'seed': seed,
        'sampled_seed_nodes': processed,
        'elapsed_seconds': elapsed,
        'seed_nodes_per_second': seeds_per_second,
        'graph_nodes': graph_nodes,
        'projected_full_cache_hours': projected_hours,
        'max_projected_hours': max_projected_hours,
        'within_budget': projected_hours <= max_projected_hours,
        'batch_size': batch_size,
        'neighbor_k': neighbor_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data-pt', type=Path, required=True)
    parser.add_argument('--checkpoint-pt', type=Path)
    parser.add_argument('--sample-nodes', type=int, default=32768)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--neighbor-k', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-projected-hours', type=float, default=20.0)
    parser.add_argument('--device')
    args = parser.parse_args()

    data = _load_graph(args.data_pt)
    checkpoint_path = args.checkpoint_pt or _base_checkpoint_path(
        args.run_dir, 'GAT', args.seed
    )
    checkpoint = _load_checkpoint(checkpoint_path, expected_kind='base')
    if checkpoint.get('arm') != 'GAT' or checkpoint.get('seed') != args.seed:
        raise ValueError('cache probe requires the matching frozen GAT checkpoint')
    device = _device(args.device)
    model = _build_base_model(checkpoint['config'], device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    sample_count = min(args.sample_nodes, int(data.num_nodes))
    loader = _loader(
        data,
        torch.arange(sample_count),
        shuffle=False,
        batch_size=args.batch_size,
        neighbor_k=args.neighbor_k,
        num_layers=checkpoint['config']['num_layers'],
        num_workers=args.num_workers,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _base_predict(model, batch)[: batch.batch_size]
            processed += int(batch.batch_size)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = _projection_report(
        seed=args.seed,
        processed=processed,
        elapsed=elapsed,
        graph_nodes=int(data.num_nodes),
        batch_size=args.batch_size,
        neighbor_k=args.neighbor_k,
        max_projected_hours=args.max_projected_hours,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report['within_budget']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
