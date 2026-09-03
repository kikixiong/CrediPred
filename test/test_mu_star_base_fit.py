"""Behavior tests for the honest mu-star training split preflight."""

import json
from pathlib import Path

from credipred.experiments.gnn_experiments.mu_star_base_fit import (
    assign_mu_star_partitions,
    write_mu_star_plan,
)


def _registry(count: int) -> list[dict[str, object]]:
    return [
        {
            "canonical_domain": f"domain-{index}.example",
            "group_id": f"group-{index}",
            "node_id": index,
        }
        for index in range(count)
    ]


def test_mu_star_partition_is_stable_disjoint_and_exactly_40_35_10_15() -> None:
    first = assign_mu_star_partitions(_registry(20), seed=42)
    second = assign_mu_star_partitions(list(reversed(_registry(20))), seed=42)

    assert first == second
    assert {row["quality_split"] for row in first} == {
        "base_fit",
        "head_fit",
        "head_tune",
        "audit",
    }
    assert {partition: sum(row["quality_split"] == partition for row in first) for partition in {
        "base_fit",
        "head_fit",
        "head_tune",
        "audit",
    }} == {"base_fit": 8, "head_fit": 7, "head_tune": 2, "audit": 3}

    grouped = _registry(10) + [
        {"canonical_domain": "alias.example", "group_id": "group-0", "node_id": 10}
    ]
    grouped_assignments = assign_mu_star_partitions(grouped, seed=42)
    assert len(
        {
            row["quality_split"]
            for row in grouped_assignments
            if row["group_id"] == "group-0"
        }
    ) == 1


def test_mu_star_plan_records_only_base_fit_as_training_eligible(tmp_path: Path) -> None:
    registry_path = tmp_path / "official-registry.jsonl"
    registry_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in _registry(10)) + "\n",
        encoding="utf-8",
    )

    plan = write_mu_star_plan(registry_path, tmp_path / "plan", seed=42)

    assert plan["training_partitions"] == ["base_fit"]
    assert plan["excluded_partitions"] == ["head_fit", "head_tune", "audit"]
    assert plan["training_status"].startswith("blocked_")
    assert plan["official_group_registry_sha256"] != plan["quality_partition_registry_sha256"]
    assert (tmp_path / "plan" / "mu_star_partitions.jsonl").is_file()
