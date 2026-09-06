"""Submission-boundary tests for the fixed clean-UQ matrix."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).parents[1]
SUBMIT_SCRIPT = REPOSITORY / 'scripts' / 'noether' / 'clean_uq_submit.sh'


class CleanUqSubmitTest(unittest.TestCase):
    def test_sbatch_failure_aborts_without_reporting_a_submitted_dag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source'
            scripts = source / 'scripts' / 'noether'
            scripts.mkdir(parents=True)
            for name in ('clean_uq_stage.sbatch', 'clean_uq_cpu_stage.sbatch'):
                (scripts / name).touch()
            python = root / 'python'
            python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
            python.chmod(0o755)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            sbatch = bin_dir / 'sbatch'
            sbatch.write_text('#!/usr/bin/env bash\nexit 19\n', encoding='utf-8')
            sbatch.chmod(0o755)
            data = root / 'data.pt'
            run_root = root / 'run'
            target_dir = run_root / 'protocol' / 'targets'
            target_dir.mkdir(parents=True)
            representative = run_root / 'checkpoints' / 'GAT' / 'seed-42.pt'
            representative.parent.mkdir(parents=True)
            representative.touch()
            fit_select = target_dir / 'fit_select.pt'
            calibrate = target_dir / 'calibrate.pt'
            partitions = root / 'partitions.jsonl'
            protocol = root / 'protocol.json'
            target_ledger = root / 'target-ledger.json'
            for path in (
                data,
                fit_select,
                calibrate,
                partitions,
                protocol,
                target_ledger,
            ):
                path.touch()

            completed = subprocess.run(
                [
                    'bash', str(SUBMIT_SCRIPT), 'pretest',
                    '--source-dir', str(source), '--run-root', str(run_root),
                    '--python-bin', str(python), '--data-pt', str(data),
                    '--fit-select-targets-pt', str(fit_select),
                    '--calibrate-targets-pt', str(calibrate),
                    '--partition-jsonl', str(partitions),
                    '--protocol-json', str(protocol),
                    '--target-ledger', str(target_ledger), '--num-nodes', '6',
                    '--representative-job-id', '12345',
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                },
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn('pretest DAG submitted', completed.stdout)

    def test_pretest_plan_never_requires_or_mentions_sealed_test_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source'
            scripts = source / 'scripts' / 'noether'
            scripts.mkdir(parents=True)
            for name in ('clean_uq_stage.sbatch', 'clean_uq_cpu_stage.sbatch'):
                (scripts / name).touch()
            python = root / 'python'
            command_log = root / 'commands.log'
            python.write_text(
                '#!/usr/bin/env bash\nprintf "python %s\\n" "$*" >> "$COMMAND_LOG"\n',
                encoding='utf-8',
            )
            python.chmod(0o755)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            sbatch = bin_dir / 'sbatch'
            sbatch.write_text(
                '#!/usr/bin/env bash\nprintf "sbatch %s\\n" "$*" >> "$COMMAND_LOG"\nprintf "999\\n"\n',
                encoding='utf-8',
            )
            sbatch.chmod(0o755)
            inputs = root / 'inputs'
            inputs.mkdir()
            data = inputs / 'data.pt'
            run_root = root / 'run'
            target_dir = run_root / 'protocol' / 'targets'
            target_dir.mkdir(parents=True)
            representative = run_root / 'checkpoints' / 'GAT' / 'seed-42.pt'
            representative.parent.mkdir(parents=True)
            representative.touch()
            fit_select = target_dir / 'fit_select.pt'
            calibrate = target_dir / 'calibrate.pt'
            partitions = inputs / 'partitions.jsonl'
            protocol = inputs / 'protocol.json'
            target_ledger = inputs / 'target-ledger.json'
            for path in (data, fit_select, calibrate, partitions, protocol, target_ledger):
                path.touch()
            sealed = inputs / 'SEALED_TEST_TARGETS_MUST_NOT_APPEAR.pt'
            environment = {
                **os.environ,
                'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                'COMMAND_LOG': str(command_log),
            }

            completed = subprocess.run(
                [
                    'bash',
                    str(SUBMIT_SCRIPT),
                    'pretest',
                    '--source-dir', str(source),
                    '--run-root', str(run_root),
                    '--python-bin', str(python),
                    '--data-pt', str(data),
                    '--fit-select-targets-pt', str(fit_select),
                    '--calibrate-targets-pt', str(calibrate),
                    '--partition-jsonl', str(partitions),
                    '--protocol-json', str(protocol),
                    '--target-ledger', str(target_ledger),
                    '--num-nodes', '6',
                    '--representative-job-id', '12345',
                    '--seeds', '42,43,44,45,46',
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            logged = command_log.read_text(encoding='utf-8')
            self.assertNotIn(str(sealed), logged)
            self.assertNotIn('sealed-test-targets-pt', logged)
            self.assertNotIn('afterok:12345', logged)
            self.assertIn(str(fit_select), logged)
            self.assertIn(str(calibrate), logged)

    def test_sealed_representative_requires_the_separate_test_target(self) -> None:
        source_text = SUBMIT_SCRIPT.read_text(encoding='utf-8')
        self.assertIn('sealed-test-representative', source_text)
        self.assertIn('SEALED_TEST_TARGETS_PT', source_text)

    def test_pretest_rejects_a_supplied_sealed_test_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source'
            scripts = source / 'scripts' / 'noether'
            scripts.mkdir(parents=True)
            for name in ('clean_uq_stage.sbatch', 'clean_uq_cpu_stage.sbatch'):
                (scripts / name).touch()
            python = root / 'python'
            python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
            python.chmod(0o755)
            inputs = root / 'inputs'
            inputs.mkdir()
            paths = [inputs / name for name in (
                'data.pt', 'fit.pt', 'calibrate.pt', 'test.pt',
                'partitions.jsonl', 'protocol.json', 'ledger.json',
            )]
            for path in paths:
                path.touch()
            completed = subprocess.run(
                [
                    'bash', str(SUBMIT_SCRIPT), 'pretest',
                    '--source-dir', str(source), '--run-root', str(root / 'run'),
                    '--python-bin', str(python), '--data-pt', str(paths[0]),
                    '--fit-select-targets-pt', str(paths[1]),
                    '--calibrate-targets-pt', str(paths[2]),
                    '--sealed-test-targets-pt', str(paths[3]),
                    '--partition-jsonl', str(paths[4]),
                    '--protocol-json', str(paths[5]),
                    '--target-ledger', str(paths[6]), '--num-nodes', '7',
                    '--representative-job-id', '12345',
                    '--seeds', '42,43,44,45,46',
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 64)
            self.assertIn('reject --sealed-test-targets-pt', completed.stderr)


if __name__ == '__main__':
    unittest.main()
