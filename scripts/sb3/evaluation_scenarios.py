"""Deterministic scenario traces for bookshelf policy evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCENARIO_VECTOR_FIELDS = [
    "reset_count",
    "scenario_bank_index",
    "row_wide_mask",
    *[f"joint_noise_{index}" for index in range(1, 8)],
    "grasp_jitter_x",
    "grasp_jitter_y",
    "grasp_jitter_z",
    "grasp_jitter_yaw",
    *[f"single_book_slot_{index}" for index in range(9)],
    *[f"wide_book_start_slot_{index}" for index in range(4)],
    "initial_book_x",
    "initial_book_y",
    "initial_book_z",
    "initial_book_qw",
    "initial_book_qx",
    "initial_book_qy",
    "initial_book_qz",
    "initial_tool_x",
    "initial_tool_y",
    "initial_tool_z",
    "initial_tool_qw",
    "initial_tool_qx",
    "initial_tool_qy",
    "initial_tool_qz",
]

SCENARIO_FIELDS = [
    "env_id",
    "slot_center_y",
    "slot_clearance",
    "missing_book_index",
    *SCENARIO_VECTOR_FIELDS,
]

SCENARIO_HASH_FIELDS = [
    "env_id",
    "slot_center_y",
    "slot_clearance",
    "missing_book_index",
    "reset_count",
    "row_wide_mask",
    *[f"joint_noise_{index}" for index in range(1, 8)],
    "grasp_jitter_x",
    "grasp_jitter_y",
    "grasp_jitter_z",
    "grasp_jitter_yaw",
    *[f"single_book_slot_{index}" for index in range(9)],
    *[f"wide_book_start_slot_{index}" for index in range(4)],
]

RESULT_FIELDS = [
    "episode_index",
    "outcome",
    "failure_code",
    "episode_reward",
    "episode_length",
]

TRACE_FIELDS = RESULT_FIELDS + SCENARIO_FIELDS


def apply_evaluation_seed_after_agent_load(agent: Any, seed: int) -> int | None:
    """Restore the requested evaluation seed after SB3 loads checkpoint state.

    Stable-Baselines3 calls ``set_random_seed`` while loading PPO and therefore
    seeds the attached environment from the checkpoint's training seed.  That
    silently overrides a seed applied before ``PPO.load``.  Reapply the CLI
    seed through the agent so Python, NumPy, Torch, the action space, and the
    wrapped Isaac Lab environment remain synchronized.
    """
    checkpoint_seed = getattr(agent, "seed", None)
    agent.set_random_seed(int(seed))
    return int(checkpoint_seed) if checkpoint_seed is not None else None


def sha256_file(path: str | Path | None) -> str | None:
    """Return a file hash, or None when the optional file does not exist."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository: str | Path) -> dict[str, Any]:
    """Read commit and branch without invoking Git LFS filters."""
    root = Path(repository)

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    return {
        "repository": str(root.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
    }


def _canonical_value(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Float32 simulator values do not need more precision than this. Fixed
        # rounding also keeps hashes portable across CSV/JSON round trips.
        return round(value, 10)
    return value


def canonical_scenario_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return sampled reset inputs in reset order for stable hashing."""
    scenarios = [
        {field: _canonical_value(row.get(field)) for field in SCENARIO_HASH_FIELDS}
        for row in rows
    ]
    return sorted(
        scenarios,
        key=lambda row: (
            int(row["env_id"]) if row["env_id"] is not None else -1,
            int(row["reset_count"]) if row["reset_count"] is not None else -1,
        ),
    )


def scenario_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(canonical_scenario_rows(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvaluationScenarioTrace:
    """Collect episode outcomes and emit a reproducible scenario manifest."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append({field: row.get(field) for field in TRACE_FIELDS})

    def write(self) -> Path:
        csv_path = self.output_dir / "episodes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        missing = Counter()
        for row in self.rows:
            for field in SCENARIO_FIELDS:
                if row.get(field) is None or row.get(field) == "":
                    missing[field] += 1

        outcomes = Counter(str(row.get("outcome", "unknown")) for row in self.rows)
        bank_metadata = self.metadata.get("frozen_scenario_bank")
        bank_coverage = None
        if isinstance(bank_metadata, dict):
            expected_count = int(bank_metadata["scenario_count"])
            indices = [int(float(row["scenario_bank_index"])) for row in self.rows]
            counts = Counter(indices)
            bank_coverage = {
                "expected_count": expected_count,
                "observed_count": len(indices),
                "missing_indices": sorted(set(range(expected_count)) - set(indices)),
                "duplicate_indices": sorted(index for index, count in counts.items() if count > 1),
                "unexpected_indices": sorted(index for index in counts if index not in range(expected_count)),
            }
            bank_coverage["complete"] = not any(
                bank_coverage[key]
                for key in ("missing_indices", "duplicate_indices", "unexpected_indices")
            ) and len(indices) == expected_count
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "episode_count": len(self.rows),
            "scenario_sha256": scenario_sha256(self.rows),
            "scenario_hash_fields": SCENARIO_HASH_FIELDS,
            "scenario_trace_complete": not missing,
            "missing_scenario_fields": dict(sorted(missing.items())),
            "outcomes": dict(sorted(outcomes.items())),
            "frozen_scenario_bank_coverage": bank_coverage,
            "metadata": self.metadata,
            "episodes_csv": str(csv_path.resolve()),
        }
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary_path
