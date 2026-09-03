#!/usr/bin/env python3
"""Check the frozen seed identity before clean-UQ pretest or sealed-test submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ARMS = ('FF', 'GCN', 'SAGE', 'GAT', 'GAT-mlp', 'GAT-topology')


def _read_state(path: Path) -> dict[str, object]:
    state = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(state, dict):
        raise ValueError(f'invalid calibration state: {path}')
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pretest', 'sealed-test'))
    parser.add_argument('--protocol-json', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol_json.read_text(encoding='utf-8'))
    if protocol.get('model_seeds') != args.seeds:
        raise ValueError('submitted seeds differ from the frozen protocol model_seeds')
    if args.phase == 'pretest':
        print(json.dumps({'phase': args.phase, 'seeds': args.seeds}, sort_keys=True))
        return

    for seed in args.seeds:
        for arm in ARMS:
            state_path = (
                args.run_root / 'audit' / str(seed) / f'{arm}_calibration.json'
            )
            state = _read_state(state_path)
            if str(state.get('seed')) != str(seed) or state.get('arm') != arm:
                raise ValueError(f'calibration state identity mismatch: {state_path}')
        for arm in ARMS:
            checkpoint_path = (
                args.run_root / 'checkpoints' / arm / f'seed-{seed}.pt'
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
        cache_path = args.run_root / 'cache' / 'GAT' / f'seed-{seed}.pt'
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)

    ensemble_path = (
        args.run_root / 'predictions' / 'GAT-ensemble' / 'calibrate.pt'
    )
    ensemble = torch.load(ensemble_path, map_location='cpu', weights_only=True)
    if ensemble.get('member_seeds') != args.seeds:
        raise ValueError('calibration ensemble members differ from frozen model seeds')
    ensemble_state = _read_state(
        args.run_root / 'audit' / 'ensemble' / 'GAT-ensemble_calibration.json'
    )
    if ensemble_state.get('member_seeds') != args.seeds:
        raise ValueError('ensemble calibration state has different member seeds')
    print(json.dumps({'phase': args.phase, 'seeds': args.seeds}, sort_keys=True))


if __name__ == '__main__':
    main()
