"""Torch-only tests for the sealed clean-UQ audit boundary."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from credipred.conformal_regression.cqr import compute_qhat
from credipred.experiments.gnn_experiments.clean_uq_audit import (
    main,
    write_prediction_ensemble,
    write_calibration_state,
    write_test_audit,
)


def _write_predictions(
    path: Path,
    *,
    role: str,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    seed: int | str = 42,
    arm: str = 'GAT',
    member_seeds: list[int] | None = None,
) -> None:
    payload = {
        'schema_version': 'clean-uq-predictions-v1',
        'role': role,
        'seed': seed,
        'arm': arm,
        'node_ids': torch.arange(labels.numel()),
        'predictions': predictions,
        'labels': labels,
    }
    if member_seeds is not None:
        payload['member_seeds'] = member_seeds
    torch.save(payload, path)


class CleanUQAuditTest(unittest.TestCase):
    def test_qhat_uses_the_finite_sample_order_statistic(self) -> None:
        """Catches torch.quantile's n-minus-one interpolation semantics."""
        scores = torch.arange(1.0, 20.0)

        qhat = compute_qhat(scores, alpha=0.1)

        self.assertEqual(qhat, 18.0)

    def test_gat_ensemble_is_derived_by_averaging_seed_predictions(self) -> None:
        """Catches retraining or misaligning nodes when deriving the ensemble arm."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / 'seed-42.pt'
            second = root / 'seed-43.pt'
            output = root / 'ensemble.pt'
            _write_predictions(
                first,
                role='calibrate',
                predictions=torch.tensor([[0.2, 0.1, 0.3], [0.8, 0.7, 0.9]]),
                labels=torch.tensor([0.0, 1.0]),
            )
            _write_predictions(
                second,
                role='calibrate',
                predictions=torch.tensor([[0.4, 0.3, 0.5], [0.6, 0.5, 0.7]]),
                labels=torch.tensor([0.0, 1.0]),
            )

            payload = write_prediction_ensemble([first, second], output)

            torch.testing.assert_close(
                payload['predictions'],
                torch.tensor([[0.3, 0.2, 0.4], [0.7, 0.6, 0.8]]),
            )
            self.assertEqual(payload['seed'], 'ensemble')
            self.assertEqual(payload['arm'], 'GAT-ensemble')
            torch.testing.assert_close(torch.load(output, weights_only=True)['labels'], torch.tensor([0.0, 1.0]))

            with self.assertRaises(FileExistsError):
                write_prediction_ensemble([first, second], output)

            with self.assertRaisesRegex(ValueError, 'role'):
                write_prediction_ensemble(
                    [first, second],
                    root / 'wrong-role.pt',
                    expected_role='test',
                )

    def test_ensemble_cli_calibrates_and_tests_under_its_own_identity(self) -> None:
        """Catches coercing the ensemble into an anchor seed's audit identity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_predictions = root / 'ensemble-calibrate.pt'
            test_predictions = root / 'ensemble-test.pt'
            for path, role in (
                (calibration_predictions, 'calibrate'),
                (test_predictions, 'test'),
            ):
                _write_predictions(
                    path,
                    role=role,
                    predictions=torch.tensor(
                        [[0.3, 0.1, 0.5], [0.7, 0.5, 0.9]]
                    ),
                    labels=torch.tensor([0.2, 0.8]),
                    seed='ensemble',
                    arm='GAT-ensemble',
                    member_seeds=[42, 43, 44],
                )

            for stage, path in (
                ('calibrate', calibration_predictions),
                ('test', test_predictions),
            ):
                argv = [
                    'clean_uq_audit.py',
                    stage,
                    '--run-dir',
                    str(root),
                    '--seed',
                    'ensemble',
                    '--arm',
                    'GAT-ensemble',
                    '--predictions-pt',
                    str(path),
                ]
                with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    main()

            audit_dir = root / 'audit' / 'ensemble'
            state = json.loads(
                (audit_dir / 'GAT-ensemble_calibration.json').read_text()
            )
            rows = json.loads((audit_dir / 'GAT-ensemble_test.json').read_text())
            self.assertEqual(state['seed'], 'ensemble')
            self.assertEqual(state['arm'], 'GAT-ensemble')
            self.assertEqual(state['member_seeds'], [42, 43, 44])
            self.assertEqual({row['seed'] for row in rows}, {'ensemble'})

    def test_ensemble_test_members_must_match_frozen_calibration_members(self) -> None:
        """Catches auditing a test ensemble built from a different seed collection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_predictions = root / 'calibrate.pt'
            test_predictions = root / 'test.pt'
            state_path = root / 'state.json'
            values = torch.tensor([[0.5, 0.2, 0.8]])
            labels = torch.tensor([0.5])
            _write_predictions(
                calibration_predictions,
                role='calibrate',
                predictions=values,
                labels=labels,
                seed='ensemble',
                arm='GAT-ensemble',
                member_seeds=[42, 43, 44],
            )
            _write_predictions(
                test_predictions,
                role='test',
                predictions=values,
                labels=labels,
                seed='ensemble',
                arm='GAT-ensemble',
                member_seeds=[42, 43, 45],
            )
            write_calibration_state(
                calibration_predictions,
                state_path,
                alpha=0.1,
                run_id='ensemble',
                arm='GAT-ensemble',
            )

            with self.assertRaisesRegex(ValueError, 'member'):
                write_test_audit(
                    test_predictions,
                    state_path,
                    root / 'test.json',
                    root / 'test.csv',
                    run_id='ensemble',
                    arm='GAT-ensemble',
                )

    def test_calibration_rejects_cli_identity_mismatch(self) -> None:
        """Catches writing a payload beneath another seed or arm's audit path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / 'calibrate.pt'
            _write_predictions(
                predictions_path,
                role='calibrate',
                predictions=torch.zeros(1, 3),
                labels=torch.zeros(1),
            )

            with self.assertRaisesRegex(ValueError, 'identity'):
                write_calibration_state(
                    predictions_path,
                    root / 'state.json',
                    alpha=0.1,
                    run_id='42',
                    arm='FF',
                )

    def test_calibration_state_refuses_overwrite(self) -> None:
        """Catches silently recalibrating an already frozen run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / 'calibrate.pt'
            state_path = root / 'state.json'
            _write_predictions(
                predictions_path,
                role='calibrate',
                predictions=torch.zeros(1, 3),
                labels=torch.zeros(1),
            )
            write_calibration_state(
                predictions_path,
                state_path,
                alpha=0.1,
                run_id='42',
                arm='GAT',
            )

            with self.assertRaises(FileExistsError):
                write_calibration_state(
                    predictions_path,
                    state_path,
                    alpha=0.1,
                    run_id='42',
                    arm='GAT',
                )

    def test_calibration_uses_finite_sample_qhat_for_both_methods(self) -> None:
        """Catches ordinary interpolation or unsorted raw endpoints in CQR."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / 'calibrate.pt'
            state_path = root / 'calibration.json'
            _write_predictions(
                predictions_path,
                role='calibrate',
                predictions=torch.tensor(
                    [
                        [0.2, 0.4, 0.0],
                        [0.8, 1.1, 0.6],
                    ]
                ),
                labels=torch.tensor([0.1, 0.9]),
            )

            state = write_calibration_state(
                predictions_path,
                state_path,
                alpha=0.1,
                run_id='42',
                arm='GAT',
            )

            self.assertAlmostEqual(state['simple_qhat'], 0.1, places=6)
            self.assertAlmostEqual(state['raw_cqr_qhat'], -0.1, places=6)
            self.assertEqual(state['cqr_qhat'], 0.0)
            self.assertEqual(state['calibration_count'], 2)
            self.assertEqual(json.loads(state_path.read_text()), state)

    def test_test_audit_reports_raw_crossing_and_clipped_interval_rows(self) -> None:
        """Catches test-time recalibration and conformalizing crossed endpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_predictions = root / 'calibrate.pt'
            test_predictions = root / 'test.pt'
            state_path = root / 'calibration.json'
            json_path = root / 'test.json'
            csv_path = root / 'test.csv'
            _write_predictions(
                calibration_predictions,
                role='calibrate',
                predictions=torch.tensor(
                    [
                        [0.2, 0.4, 0.0],
                        [0.8, 1.1, 0.6],
                    ]
                ),
                labels=torch.tensor([0.1, 0.9]),
            )
            _write_predictions(
                test_predictions,
                role='test',
                predictions=torch.tensor(
                    [
                        [-0.2, 0.3, -0.1],
                        [1.2, 1.2, 0.8],
                    ]
                ),
                labels=torch.tensor([0.0, 1.0]),
            )
            write_calibration_state(
                calibration_predictions,
                state_path,
                alpha=0.1,
                run_id='42',
                arm='GAT',
            )

            rows = write_test_audit(
                test_predictions,
                state_path,
                json_path,
                csv_path,
                run_id='42',
                arm='GAT',
            )

            self.assertEqual([row['method'] for row in rows], ['simple', 'cqr'])
            self.assertTrue(all(row['raw_crossing_rate'] == 1.0 for row in rows))
            self.assertTrue(all(0.0 <= row['coverage'] <= 1.0 for row in rows))
            self.assertTrue(all(0.0 <= row['mean_width'] <= 1.0 for row in rows))
            self.assertTrue(all(0.0 <= row['median_width'] <= 1.0 for row in rows))
            self.assertAlmostEqual(rows[1]['median_width'], 0.25, places=6)
            self.assertEqual(json.loads(json_path.read_text()), rows)
            with csv_path.open(newline='') as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([row['method'] for row in csv_rows], ['simple', 'cqr'])

    def test_negative_cqr_qhat_produces_sorted_nonnegative_final_width(self) -> None:
        """Catches negative qhat reversing adjusted CQR interval endpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_predictions = root / 'calibrate.pt'
            test_predictions = root / 'test.pt'
            state_path = root / 'state.json'
            json_path = root / 'test.json'
            csv_path = root / 'test.csv'
            _write_predictions(
                calibration_predictions,
                role='calibrate',
                predictions=torch.tensor([[0.5, 0.0, 1.0]]),
                labels=torch.tensor([0.5]),
            )
            _write_predictions(
                test_predictions,
                role='test',
                predictions=torch.tensor([[0.5, 0.4, 0.6]]),
                labels=torch.tensor([0.5]),
            )
            state = write_calibration_state(
                calibration_predictions,
                state_path,
                alpha=0.1,
                run_id='42',
                arm='GAT',
            )

            rows = write_test_audit(
                test_predictions,
                state_path,
                json_path,
                csv_path,
                run_id='42',
                arm='GAT',
            )

            self.assertEqual(state['raw_cqr_qhat'], -0.5)
            self.assertEqual(state['cqr_qhat'], 0.0)
            self.assertEqual(
                state['cqr_interval_policy'],
                'sort_raw_nonnegative_qhat_adjust_clip_0_1',
            )
            self.assertAlmostEqual(rows[1]['mean_width'], 0.2, places=6)
            self.assertGreaterEqual(rows[1]['median_width'], 0.0)
            self.assertIsNone(rows[0]['midpoint_spearman'])
            self.assertIsNone(json.loads(json_path.read_text())[0]['midpoint_spearman'])

    def test_point_only_baseline_calibrates_and_audits_only_simple_conformal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_predictions = root / 'calibrate.pt'
            test_predictions = root / 'test.pt'
            state_path = root / 'state.json'
            _write_predictions(
                calibration_predictions,
                role='calibrate',
                predictions=torch.tensor([[0.2], [0.8]]),
                labels=torch.tensor([0.1, 0.9]),
                seed='deterministic',
                arm='GlobalMedian',
            )
            _write_predictions(
                test_predictions,
                role='test',
                predictions=torch.tensor([[0.3], [0.7]]),
                labels=torch.tensor([0.2, 0.8]),
                seed='deterministic',
                arm='GlobalMedian',
            )

            state = write_calibration_state(
                calibration_predictions,
                state_path,
                alpha=0.1,
                run_id='deterministic',
                arm='GlobalMedian',
            )
            rows = write_test_audit(
                test_predictions,
                state_path,
                root / 'test.json',
                root / 'test.csv',
                run_id='deterministic',
                arm='GlobalMedian',
            )

            self.assertEqual(state['prediction_kind'], 'point-only')
            self.assertEqual(state['methods'], ['simple'])
            self.assertEqual([row['method'] for row in rows], ['simple'])
            self.assertIsNone(rows[0]['raw_crossing_rate'])

    def test_test_prediction_kind_must_match_frozen_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_predictions = root / 'calibrate.pt'
            test_predictions = root / 'test.pt'
            state_path = root / 'state.json'
            _write_predictions(
                calibration_predictions,
                role='calibrate',
                predictions=torch.tensor([[0.5]]),
                labels=torch.tensor([0.5]),
            )
            _write_predictions(
                test_predictions,
                role='test',
                predictions=torch.tensor([[0.5, 0.2, 0.8]]),
                labels=torch.tensor([0.5]),
            )
            write_calibration_state(
                calibration_predictions,
                state_path,
                alpha=0.1,
                run_id='42',
                arm='GAT',
            )

            with self.assertRaisesRegex(ValueError, 'prediction kind'):
                write_test_audit(
                    test_predictions,
                    state_path,
                    root / 'test.json',
                    root / 'test.csv',
                    run_id='42',
                    arm='GAT',
                )

    def test_calibration_rejects_a_test_role_artifact(self) -> None:
        """Catches accidental opening of sealed test labels during calibration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            test_predictions = root / 'test.pt'
            _write_predictions(
                test_predictions,
                role='test',
                predictions=torch.zeros(1, 3),
                labels=torch.zeros(1),
            )

            with self.assertRaisesRegex(ValueError, 'calibrate'):
                write_calibration_state(
                    test_predictions,
                    root / 'calibration.json',
                    alpha=0.1,
                    run_id='42',
                    arm='GAT',
                )


if __name__ == '__main__':
    unittest.main()
