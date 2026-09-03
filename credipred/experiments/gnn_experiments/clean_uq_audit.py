"""Independent calibration and sealed-test audit for clean graph UQ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from credipred.conformal_regression.cqr import compute_cqr_scores, compute_qhat


PRIMARY_ALPHA = 0.1


def _load_predictions(path: Path, expected_role: str) -> dict[str, Any]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 'clean-uq-predictions-v1':
        raise ValueError('not a clean-UQ prediction artifact')
    if payload.get('role') != expected_role:
        raise ValueError(f'audit expected {expected_role!r} predictions')
    predictions = payload.get('predictions')
    labels = payload.get('labels')
    node_ids = payload.get('node_ids')
    if (
        not isinstance(predictions, torch.Tensor)
        or predictions.ndim != 2
        or predictions.shape[1] not in (1, 3)
    ):
        raise ValueError('predictions must have shape [N, 1] or [N, 3]')
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError('labels must have shape [N]')
    if not isinstance(node_ids, torch.Tensor) or node_ids.ndim != 1:
        raise ValueError('node_ids must have shape [N]')
    if predictions.shape[0] != labels.numel() or labels.numel() != node_ids.numel():
        raise ValueError('prediction artifact columns must have equal row counts')
    if labels.numel() == 0:
        raise ValueError(f'{expected_role} prediction artifact is empty')
    prediction_kind = (
        'point-only' if predictions.shape[1] == 1 else 'point-with-endpoints'
    )
    methods = ['simple'] if predictions.shape[1] == 1 else ['simple', 'cqr']
    if 'uq_methods' in payload and payload['uq_methods'] != methods:
        raise ValueError('prediction artifact UQ methods disagree with its columns')
    payload['_prediction_kind'] = prediction_kind
    payload['_methods'] = methods
    return payload


def _ordered_endpoints(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.minimum(predictions[:, 1], predictions[:, 2]), torch.maximum(
        predictions[:, 1], predictions[:, 2]
    )


def write_prediction_ensemble(
    prediction_paths: list[Path],
    output_path: Path,
    *,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Derive a GAT ensemble from aligned seed artifacts without training."""
    if not prediction_paths:
        raise ValueError('ensemble requires at least one prediction artifact')
    first_raw = torch.load(prediction_paths[0], map_location='cpu', weights_only=True)
    if not isinstance(first_raw, dict) or not isinstance(first_raw.get('role'), str):
        raise ValueError('not a clean-UQ prediction artifact')
    role = first_raw['role']
    if expected_role is not None and role != expected_role:
        raise ValueError('ensemble payload role does not match the requested role')
    payloads = [_load_predictions(path, role) for path in prediction_paths]
    reference = payloads[0]
    if any(payload.get('arm') != 'GAT' for payload in payloads):
        raise ValueError('only GAT seed predictions can form the GAT ensemble')
    if any(payload['_prediction_kind'] != 'point-with-endpoints' for payload in payloads):
        raise ValueError('GAT ensemble members require quantile endpoints')
    for payload in payloads[1:]:
        if not torch.equal(payload['node_ids'], reference['node_ids']):
            raise ValueError('ensemble prediction node order differs across seeds')
        if not torch.equal(payload['labels'], reference['labels']):
            raise ValueError('ensemble labels differ across seeds')
    ensemble: dict[str, Any] = {
        'schema_version': 'clean-uq-predictions-v1',
        'role': role,
        'seed': 'ensemble',
        'arm': 'GAT-ensemble',
        'node_ids': reference['node_ids'],
        'labels': reference['labels'],
        'predictions': torch.stack(
            [payload['predictions'].float() for payload in payloads]
        ).mean(0),
        'member_seeds': [payload.get('seed') for payload in payloads],
        'uq_methods': ['simple', 'cqr'],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite ensemble: {output_path}')
    torch.save(ensemble, output_path)
    return ensemble


def write_calibration_state(
    predictions_path: Path,
    state_path: Path,
    *,
    alpha: float = PRIMARY_ALPHA,
    run_id: str,
    arm: str,
) -> dict[str, Any]:
    """Read calibration predictions and persist every legally supported method."""
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie strictly between zero and one')
    payload = _load_predictions(predictions_path, 'calibrate')
    if str(payload.get('seed')) != run_id or payload.get('arm') != arm:
        raise ValueError('calibration payload identity does not match the requested run')
    if state_path.exists():
        raise FileExistsError(f'refusing to overwrite calibration state: {state_path}')
    predictions = payload['predictions'].float()
    labels = payload['labels'].float()
    simple_scores = torch.abs(predictions[:, 0] - labels)
    state: dict[str, Any] = {
        'schema_version': 'clean-uq-calibration-v1',
        'seed': payload.get('seed'),
        'arm': payload.get('arm'),
        'alpha': alpha,
        'calibration_count': int(labels.numel()),
        'simple_qhat': compute_qhat(simple_scores, alpha),
        'prediction_kind': payload['_prediction_kind'],
        'methods': payload['_methods'],
    }
    if payload['_prediction_kind'] == 'point-with-endpoints':
        lower, upper = _ordered_endpoints(predictions)
        cqr_scores = compute_cqr_scores(lower, upper, labels)
        raw_cqr_qhat = compute_qhat(cqr_scores, alpha)
        state.update(
            {
                'raw_cqr_qhat': raw_cqr_qhat,
                'cqr_qhat': max(raw_cqr_qhat, 0.0),
                'cqr_interval_policy': 'sort_raw_nonnegative_qhat_adjust_clip_0_1',
            }
        )
    if 'member_seeds' in payload:
        state['member_seeds'] = payload['member_seeds']
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    return state


def _average_ranks(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_rank = _average_ranks(left.detach().cpu().numpy())
    right_rank = _average_ranks(right.detach().cpu().numpy())
    if len(left_rank) < 2 or left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _metric_row(
    *,
    payload: dict[str, Any],
    method: str,
    alpha: float,
    qhat: float,
    raw_crossing_rate: float | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, Any]:
    predictions = payload['predictions'].float()
    labels = payload['labels'].float()
    midpoint = predictions[:, 0]
    width = upper - lower
    below = torch.clamp(lower - labels, min=0.0)
    above = torch.clamp(labels - upper, min=0.0)
    interval_score = width + (2.0 / alpha) * (below + above)
    residual = midpoint - labels
    return {
        'schema_version': 'clean-uq-audit-row-v1',
        'seed': payload.get('seed'),
        'arm': payload.get('arm'),
        'method': method,
        'alpha': alpha,
        'qhat': qhat,
        'test_count': int(labels.numel()),
        'raw_crossing_rate': raw_crossing_rate,
        'coverage': float(((labels >= lower) & (labels <= upper)).float().mean()),
        'mean_width': float(width.mean()),
        'median_width': float(torch.quantile(width, 0.5)),
        'interval_score': float(interval_score.mean()),
        'midpoint_mae': float(residual.abs().mean()),
        'midpoint_rmse': float(torch.sqrt(torch.mean(residual.square()))),
        'midpoint_spearman': _spearman(midpoint, labels),
    }


def write_test_audit(
    predictions_path: Path,
    state_path: Path,
    json_path: Path,
    csv_path: Path,
    *,
    run_id: str,
    arm: str,
) -> list[dict[str, Any]]:
    """Apply frozen calibration state to sealed-test predictions and write rows."""
    payload = _load_predictions(predictions_path, 'test')
    state_raw = json.loads(state_path.read_text(encoding='utf-8'))
    if not isinstance(state_raw, dict) or state_raw.get('schema_version') != 'clean-uq-calibration-v1':
        raise ValueError('not a clean-UQ calibration state')
    if str(payload.get('seed')) != run_id or payload.get('arm') != arm:
        raise ValueError('test payload identity does not match the requested run')
    if str(state_raw.get('seed')) != run_id or state_raw.get('arm') != arm:
        raise ValueError('calibration state identity does not match the requested run')
    if payload.get('seed') != state_raw.get('seed') or payload.get('arm') != state_raw.get('arm'):
        raise ValueError('calibration state and test predictions identify different runs')
    if payload.get('member_seeds') != state_raw.get('member_seeds'):
        raise ValueError('test ensemble members differ from frozen calibration members')
    state_kind = state_raw.get('prediction_kind')
    state_methods = state_raw.get('methods')
    expected_methods = (
        ['simple'] if state_kind == 'point-only' else ['simple', 'cqr']
        if state_kind == 'point-with-endpoints'
        else None
    )
    if expected_methods is None or state_methods != expected_methods:
        raise ValueError('calibration state has an invalid prediction kind or methods')
    if payload['_prediction_kind'] != state_kind:
        raise ValueError('test prediction kind differs from frozen calibration')
    alpha = float(state_raw['alpha'])
    predictions = payload['predictions'].float()
    raw_crossing_rate = (
        float((predictions[:, 1] > predictions[:, 2]).float().mean())
        if state_kind == 'point-with-endpoints'
        else None
    )

    simple_qhat = float(state_raw['simple_qhat'])
    simple_lower = torch.clamp(predictions[:, 0] - simple_qhat, 0.0, 1.0)
    simple_upper = torch.clamp(predictions[:, 0] + simple_qhat, 0.0, 1.0)
    rows = [
        _metric_row(
            payload=payload,
            method='simple',
            alpha=alpha,
            qhat=simple_qhat,
            raw_crossing_rate=raw_crossing_rate,
            lower=simple_lower,
            upper=simple_upper,
        )
    ]
    if state_kind == 'point-with-endpoints':
        ordered_lower, ordered_upper = _ordered_endpoints(predictions)
        cqr_qhat = float(state_raw['cqr_qhat'])
        cqr_lower = torch.clamp(ordered_lower - cqr_qhat, 0.0, 1.0)
        cqr_upper = torch.clamp(ordered_upper + cqr_qhat, 0.0, 1.0)
        rows.append(
            _metric_row(
                payload=payload,
                method='cqr',
                alpha=alpha,
                qhat=cqr_qhat,
                raw_crossing_rate=raw_crossing_rate,
                lower=cqr_lower,
                upper=cqr_upper,
            )
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rows, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _default_audit_path(run_dir: Path, run_id: str, arm: str, suffix: str) -> Path:
    return run_dir / 'audit' / run_id / f'{arm}_{suffix}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='stage', required=True)
    calibrate = subparsers.add_parser('calibrate')
    calibrate.add_argument('--run-dir', type=Path, required=True)
    calibrate.add_argument('--seed', required=True)
    calibrate.add_argument('--arm', required=True)
    calibrate.add_argument('--predictions-pt', type=Path, required=True)
    calibrate.add_argument('--alpha', type=float, default=PRIMARY_ALPHA)
    test = subparsers.add_parser('test')
    test.add_argument('--run-dir', type=Path, required=True)
    test.add_argument('--seed', required=True)
    test.add_argument('--arm', required=True)
    test.add_argument('--predictions-pt', type=Path, required=True)
    ensemble = subparsers.add_parser('ensemble')
    ensemble.add_argument('--run-dir', type=Path, required=True)
    ensemble.add_argument('--role', choices=('calibrate', 'test'), required=True)
    ensemble.add_argument('--predictions-pt', type=Path, action='append', required=True)
    ensemble.add_argument('--output-pt', type=Path)
    args = parser.parse_args()

    if args.stage == 'ensemble':
        output_path = args.output_pt or (
            args.run_dir / 'predictions' / 'GAT-ensemble' / f'{args.role}.pt'
        )
        ensemble_result = write_prediction_ensemble(
            args.predictions_pt,
            output_path,
            expected_role=args.role,
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in ensemble_result.items()
                    if not isinstance(value, torch.Tensor)
                },
                sort_keys=True,
            )
        )
        return

    state_path = _default_audit_path(args.run_dir, args.seed, args.arm, 'calibration.json')
    if args.stage == 'calibrate':
        result: dict[str, Any] | list[dict[str, Any]] = write_calibration_state(
            args.predictions_pt,
            state_path,
            alpha=args.alpha,
            run_id=args.seed,
            arm=args.arm,
        )
    else:
        result = write_test_audit(
            args.predictions_pt,
            state_path,
            _default_audit_path(args.run_dir, args.seed, args.arm, 'test.json'),
            _default_audit_path(args.run_dir, args.seed, args.arm, 'test.csv'),
            run_id=args.seed,
            arm=args.arm,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
