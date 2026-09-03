"""Exposure-boundary tests for clean-UQ phase admission."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'noether' / 'clean_uq_phase_check.py'
)
SPEC = importlib.util.spec_from_file_location('clean_uq_phase_check', SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
phase_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase_check)


class CleanUqPhaseCheckTest(unittest.TestCase):
    def test_pretest_never_hashes_or_opens_the_sealed_test_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protocol_dir = root / 'protocol'
            target_dir = protocol_dir / 'targets'
            sealed_target = root / 'sealed' / 'targets' / 'test.pt'
            target_dir.mkdir(parents=True)
            partition = protocol_dir / 'clean_uq_partitions.jsonl'
            protocol = protocol_dir / 'clean_uq_protocol.json'
            fit_select = target_dir / 'fit_select.pt'
            calibrate = target_dir / 'calibrate.pt'
            for path in (partition, protocol, fit_select, calibrate):
                path.touch()
            ledger_path = target_dir / 'clean_uq_target_ledger.json'
            ledger_path.write_text(
                json.dumps(
                    {
                        'schema_version': 'clean-uq-target-ledger-v1',
                        'source_labels_sha256': phase_check.FROZEN_DQR_LABELS_SHA256,
                        'partition_sha256': phase_check.FROZEN_PARTITION_SHA256,
                        'num_nodes': phase_check.FULL_GRAPH_NODE_COUNT,
                        'artifacts': {
                            'fit_select': {
                                'path': str(fit_select.resolve()),
                                'authorized_roles': ['fit', 'select'],
                                'sha256': 'fit-sha',
                            },
                            'calibrate': {
                                'path': str(calibrate.resolve()),
                                'authorized_roles': ['calibrate'],
                                'sha256': 'calibrate-sha',
                            },
                            'test': {
                                'path': str(sealed_target.resolve()),
                                'authorized_roles': ['test'],
                                'sha256': 'sealed-sha',
                            },
                        },
                    }
                )
                + '\n',
                encoding='utf-8',
            )
            digests = {
                protocol: phase_check.FROZEN_PROTOCOL_SHA256,
                partition: phase_check.FROZEN_PARTITION_SHA256,
                fit_select: 'fit-sha',
                calibrate: 'calibrate-sha',
            }
            hashed: list[Path] = []

            def recording_hash(path: Path) -> str:
                hashed.append(path)
                return digests[path]

            with mock.patch.object(
                phase_check, 'sha256_file', side_effect=recording_hash
            ):
                phase_check._validate_target_ledger(
                    ledger_path,
                    phase='pretest',
                    protocol_json=protocol,
                    run_root=root,
                )

            self.assertEqual(
                hashed, [protocol, partition, fit_select, calibrate]
            )
            self.assertNotIn(sealed_target, hashed)
            self.assertFalse(sealed_target.exists())


if __name__ == '__main__':
    unittest.main()
