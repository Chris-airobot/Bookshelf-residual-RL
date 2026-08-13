"""Pure helpers for exporting and replaying frozen bookshelf reset scenarios."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SINGLE_BOOK_SLOT_COUNT = 9
WIDE_BOOK_SLOT_COUNT = 4

FLOAT_SCENARIO_FIELDS = [
    "slot_center_y",
    "slot_clearance",
    *[f"joint_noise_{index}" for index in range(1, 8)],
    "grasp_jitter_x",
    "grasp_jitter_y",
    "grasp_jitter_z",
    "grasp_jitter_yaw",
]
INTEGER_SCENARIO_FIELDS = [
    "missing_book_index",
    "row_wide_mask",
    *[f"single_book_slot_{index}" for index in range(SINGLE_BOOK_SLOT_COUNT)],
    *[f"wide_book_start_slot_{index}" for index in range(WIDE_BOOK_SLOT_COUNT)],
]
FROZEN_SCENARIO_FIELDS = FLOAT_SCENARIO_FIELDS + INTEGER_SCENARIO_FIELDS


def _canonical_number(value: Any) -> int | float:
    number = float(value)
    if not number == number or abs(number) == float("inf"):
        raise ValueError(f"Scenario value is not finite: {value!r}")
    return round(number, 10)


def canonical_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    result = {"scenario_id": int(scenario["scenario_id"])}
    for field in FLOAT_SCENARIO_FIELDS:
        result[field] = _canonical_number(scenario[field])
    for field in INTEGER_SCENARIO_FIELDS:
        result[field] = int(float(scenario[field]))
    missing = result["missing_book_index"]
    if missing not in range(10):
        raise ValueError(f"missing_book_index must be in [0, 9], got {missing}")
    single_slots = [result[f"single_book_slot_{index}"] for index in range(SINGLE_BOOK_SLOT_COUNT)]
    wide_starts = [result[f"wide_book_start_slot_{index}"] for index in range(WIDE_BOOK_SLOT_COUNT)]
    if any(slot < -1 or slot > 9 for slot in single_slots):
        raise ValueError("Single-book assignments must be -1 or a row slot in [0, 9].")
    if any(start < -1 or start > 8 for start in wide_starts):
        raise ValueError("Wide-book assignments must be -1 or a start slot in [0, 8].")
    occupied = [slot for slot in single_slots if slot >= 0]
    for start in wide_starts:
        if start >= 0:
            occupied.extend((start, start + 1))
    expected = [slot for slot in range(10) if slot != missing]
    if sorted(occupied) != expected:
        raise ValueError(
            f"Book assignments must cover every slot except {missing} exactly once; got {sorted(occupied)}"
        )
    expected_mask = sum(1 << start for start in wide_starts if start >= 0)
    if result["row_wide_mask"] != expected_mask:
        raise ValueError(
            f"row_wide_mask={result['row_wide_mask']} does not match wide-book starts ({expected_mask})."
        )
    return result


def frozen_scenarios_sha256(scenarios: Iterable[dict[str, Any]]) -> str:
    canonical = [canonical_scenario(scenario) for scenario in scenarios]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_frozen_scenario_bank(trace_summary_path: str | Path) -> dict[str, Any]:
    """Build a bank from a complete scenario trace without copying outcomes."""
    summary_path = Path(trace_summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("scenario_trace_complete", False):
        raise ValueError(f"Scenario trace is incomplete: {summary_path}")

    csv_path = Path(summary["episodes_csv"])
    if not csv_path.is_absolute():
        csv_path = summary_path.parent / csv_path
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Scenario trace contains no episodes: {csv_path}")

    missing_columns = [field for field in FROZEN_SCENARIO_FIELDS if field not in rows[0]]
    if missing_columns:
        raise ValueError(
            "Trace predates complete frozen-bank capture; missing columns: "
            + ", ".join(missing_columns)
        )

    scenarios = []
    for scenario_id, row in enumerate(rows):
        missing_values = [field for field in FROZEN_SCENARIO_FIELDS if row.get(field, "") == ""]
        if missing_values:
            raise ValueError(
                f"Trace row {scenario_id} has missing frozen scenario values: "
                + ", ".join(missing_values)
            )
        scenario = {"scenario_id": scenario_id}
        scenario.update({field: row[field] for field in FROZEN_SCENARIO_FIELDS})
        scenarios.append(canonical_scenario(scenario))

    bank_hash = frozen_scenarios_sha256(scenarios)
    return {
        "schema_version": 1,
        "kind": "bookshelf_frozen_evaluation_scenario_bank",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenarios),
        "scenario_sha256": bank_hash,
        "scenario_fields": FROZEN_SCENARIO_FIELDS,
        "source": {
            "trace_summary": str(summary_path),
            "trace_summary_sha256": file_sha256(summary_path),
            "episodes_csv": str(csv_path.resolve()),
            "episodes_csv_sha256": file_sha256(csv_path),
            "trace_scenario_sha256": summary.get("scenario_sha256"),
            "metadata": summary.get("metadata", {}),
        },
        "scenarios": scenarios,
    }


def write_frozen_scenario_bank(trace_summary_path: str | Path, output_path: str | Path) -> Path:
    bank = build_frozen_scenario_bank(trace_summary_path)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def generate_frozen_scenario_bank(
    *,
    scenario_count: int,
    seed: int,
    slot_clearance_min: float,
    slot_clearance_max: float,
    slot_pitch: float,
    row_book_count: int,
    side_book_merge_probability: float,
    arm_joint_noise: float,
    grasp_x_jitter: float,
    grasp_y_jitter: float,
    grasp_z_jitter: float,
    grasp_yaw_jitter: float,
) -> dict[str, Any]:
    """Generate a policy-independent bank from the evaluation distributions."""
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")
    if row_book_count != 10:
        raise ValueError("The current frozen-bank schema requires row_book_count=10.")
    if slot_clearance_min > slot_clearance_max:
        raise ValueError("slot_clearance_min cannot exceed slot_clearance_max.")
    if not 0.0 <= side_book_merge_probability <= 1.0:
        raise ValueError("side_book_merge_probability must be in [0, 1].")

    rng = random.Random(int(seed))
    row_center = 0.5 * (row_book_count - 1)
    scenarios = []
    for scenario_id in range(scenario_count):
        missing = rng.randrange(row_book_count)
        clearance = rng.uniform(slot_clearance_min, slot_clearance_max)
        single_assignments = [-1] * SINGLE_BOOK_SLOT_COUNT
        wide_assignments = [-1] * WIDE_BOOK_SLOT_COUNT
        single_order = list(range(SINGLE_BOOK_SLOT_COUNT))
        wide_order = list(range(WIDE_BOOK_SLOT_COUNT))
        rng.shuffle(single_order)
        rng.shuffle(wide_order)
        single_cursor = 0
        wide_cursor = 0
        visible_slots = [slot for slot in range(row_book_count) if slot != missing]
        for side_slots in (
            [slot for slot in visible_slots if slot < missing],
            [slot for slot in visible_slots if slot > missing],
        ):
            cursor = 0
            while cursor < len(side_slots):
                can_merge = (
                    cursor + 1 < len(side_slots)
                    and side_slots[cursor + 1] == side_slots[cursor] + 1
                )
                merge = (
                    can_merge
                    and wide_cursor < WIDE_BOOK_SLOT_COUNT
                    and rng.random() < side_book_merge_probability
                )
                if merge:
                    wide_assignments[wide_order[wide_cursor]] = side_slots[cursor]
                    wide_cursor += 1
                    cursor += 2
                else:
                    single_assignments[single_order[single_cursor]] = side_slots[cursor]
                    single_cursor += 1
                    cursor += 1

        scenario = {
            "scenario_id": scenario_id,
            "slot_center_y": (missing - row_center) * slot_pitch,
            "slot_clearance": clearance,
            "missing_book_index": missing,
            "row_wide_mask": sum(1 << start for start in wide_assignments if start >= 0),
            **{
                f"joint_noise_{index}": rng.uniform(-arm_joint_noise, arm_joint_noise)
                for index in range(1, 8)
            },
            "grasp_jitter_x": rng.uniform(-grasp_x_jitter, grasp_x_jitter),
            "grasp_jitter_y": rng.uniform(-grasp_y_jitter, grasp_y_jitter),
            "grasp_jitter_z": rng.uniform(-grasp_z_jitter, grasp_z_jitter),
            "grasp_jitter_yaw": rng.uniform(-grasp_yaw_jitter, grasp_yaw_jitter),
            **{
                f"single_book_slot_{index}": value
                for index, value in enumerate(single_assignments)
            },
            **{
                f"wide_book_start_slot_{index}": value
                for index, value in enumerate(wide_assignments)
            },
        }
        scenarios.append(canonical_scenario(scenario))

    generation = {
        "generator": "python_random_v1",
        "seed": int(seed),
        "scenario_count": int(scenario_count),
        "slot_clearance_min": float(slot_clearance_min),
        "slot_clearance_max": float(slot_clearance_max),
        "slot_pitch": float(slot_pitch),
        "row_book_count": int(row_book_count),
        "side_book_merge_probability": float(side_book_merge_probability),
        "arm_joint_noise": float(arm_joint_noise),
        "grasp_x_jitter": float(grasp_x_jitter),
        "grasp_y_jitter": float(grasp_y_jitter),
        "grasp_z_jitter": float(grasp_z_jitter),
        "grasp_yaw_jitter": float(grasp_yaw_jitter),
    }
    return {
        "schema_version": 1,
        "kind": "bookshelf_frozen_evaluation_scenario_bank",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenarios),
        "scenario_sha256": frozen_scenarios_sha256(scenarios),
        "scenario_fields": FROZEN_SCENARIO_FIELDS,
        "source": {"generation": generation},
        "scenarios": scenarios,
    }


def write_generated_frozen_scenario_bank(output_path: str | Path, **kwargs: Any) -> Path:
    bank = generate_frozen_scenario_bank(**kwargs)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_frozen_scenario_bank(path: str | Path) -> dict[str, Any]:
    bank_path = Path(path).resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != 1:
        raise ValueError(f"Unsupported frozen scenario bank schema: {bank.get('schema_version')!r}")
    if bank.get("kind") != "bookshelf_frozen_evaluation_scenario_bank":
        raise ValueError(f"Not a bookshelf frozen scenario bank: {bank_path}")
    scenarios = bank.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Frozen scenario bank is empty: {bank_path}")
    canonical = [canonical_scenario(scenario) for scenario in scenarios]
    expected_ids = list(range(len(canonical)))
    actual_ids = [scenario["scenario_id"] for scenario in canonical]
    if actual_ids != expected_ids:
        raise ValueError("Frozen scenario IDs must be contiguous and ordered from zero.")
    expected_hash = frozen_scenarios_sha256(canonical)
    if bank.get("scenario_sha256") != expected_hash:
        raise ValueError(
            f"Frozen scenario bank hash mismatch: stored={bank.get('scenario_sha256')} computed={expected_hash}"
        )
    if int(bank.get("scenario_count", -1)) != len(canonical):
        raise ValueError("Frozen scenario bank count does not match its scenario list.")
    bank["scenarios"] = canonical
    bank["path"] = str(bank_path)
    return bank


class FrozenScenarioAllocator:
    """Assign every scenario once using policy-independent per-env queues."""

    def __init__(self, scenarios: Iterable[dict[str, Any]], num_envs: int):
        self.scenarios = [canonical_scenario(scenario) for scenario in scenarios]
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        self.next_index_by_env = list(range(self.num_envs))

    def allocate(self, env_ids: Iterable[int]) -> dict[int, dict[str, Any] | None]:
        assignments: dict[int, dict[str, Any] | None] = {}
        for env_id in sorted(int(value) for value in env_ids):
            if env_id < 0 or env_id >= self.num_envs:
                raise ValueError(f"Environment ID {env_id} is outside [0, {self.num_envs - 1}].")
            scenario_index = self.next_index_by_env[env_id]
            if scenario_index < len(self.scenarios):
                assignments[env_id] = self.scenarios[scenario_index]
                self.next_index_by_env[env_id] += self.num_envs
            else:
                assignments[env_id] = None
        return assignments

    @property
    def exhausted(self) -> bool:
        return all(index >= len(self.scenarios) for index in self.next_index_by_env)
