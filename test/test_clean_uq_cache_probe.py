"""Focused threshold contract for the Noether parent-cache throughput probe."""

import unittest

from scripts.noether.clean_uq_cache_probe import _projection_report


class CleanUQCacheProbeTest(unittest.TestCase):
    def test_projection_report_drives_the_frozen_runtime_gate(self) -> None:
        report = _projection_report(
            seed=42,
            processed=100,
            elapsed=1.0,
            graph_nodes=360_000,
            batch_size=50,
            neighbor_k=30,
            max_projected_hours=0.5,
        )

        self.assertEqual(report['projected_full_cache_hours'], 1.0)
        self.assertEqual(report['max_projected_hours'], 0.5)
        self.assertFalse(report['within_budget'])


if __name__ == '__main__':
    unittest.main()
