import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source/bookshelf/bookshelf/tasks/direct/bookshelf/frozen_scenario_bank.py"
)
SPEC = importlib.util.spec_from_file_location("bookshelf_frozen_scenario_bank_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BANK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BANK)


def _scenario(scenario_id=0):
    scenario = {
        "scenario_id": scenario_id,
        "slot_center_y": -0.017,
        "slot_clearance": 0.003,
        "missing_book_index": 4,
        "row_wide_mask": 0,
        **{f"joint_noise_{index}": 0.001 * index for index in range(1, 8)},
        "grasp_jitter_x": 0.001,
        "grasp_jitter_y": -0.001,
        "grasp_jitter_z": 0.0005,
        "grasp_jitter_yaw": 0.01,
        **{f"single_book_slot_{index}": slot for index, slot in enumerate([0, 1, 2, 3, 5, 6, 7, 8, 9])},
        **{f"wide_book_start_slot_{index}": -1 for index in range(4)},
    }
    return scenario


def test_allocator_never_reuses_scenarios():
    allocator = BANK.FrozenScenarioAllocator(
        [_scenario(0), _scenario(1), _scenario(2)], num_envs=2
    )
    first = allocator.allocate([1, 0])
    second = allocator.allocate([1, 0])
    assert first[0]["scenario_id"] == 0
    assert first[1]["scenario_id"] == 1
    assert second[0]["scenario_id"] == 2
    assert second[1] is None
    assert allocator.exhausted


def test_allocator_mapping_does_not_depend_on_completion_order():
    scenarios = [_scenario(index) for index in range(6)]
    fast_env_zero = BANK.FrozenScenarioAllocator(scenarios, num_envs=2)
    delayed_env_zero = BANK.FrozenScenarioAllocator(scenarios, num_envs=2)

    first_order = [
        fast_env_zero.allocate([0, 1]),
        fast_env_zero.allocate([0]),
        fast_env_zero.allocate([0, 1]),
    ]
    second_order = [
        delayed_env_zero.allocate([0, 1]),
        delayed_env_zero.allocate([1]),
        delayed_env_zero.allocate([0, 1]),
    ]

    def ids_by_env(assignments):
        result = {0: [], 1: []}
        for assignment in assignments:
            for env_id, scenario in assignment.items():
                if scenario is not None:
                    result[env_id].append(scenario["scenario_id"])
        return result

    assert ids_by_env(first_order) == {0: [0, 2, 4], 1: [1, 3]}
    assert ids_by_env(second_order) == {0: [0, 2], 1: [1, 3, 5]}


def test_bank_rejects_tampering(tmp_path):
    scenarios = [_scenario(0), _scenario(1)]
    bank = {
        "schema_version": 1,
        "kind": "bookshelf_frozen_evaluation_scenario_bank",
        "scenario_count": 2,
        "scenario_sha256": BANK.frozen_scenarios_sha256(scenarios),
        "scenarios": scenarios,
    }
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    assert BANK.load_frozen_scenario_bank(path)["scenario_count"] == 2

    bank["scenarios"][1]["grasp_jitter_x"] = 0.002
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="hash mismatch"):
        BANK.load_frozen_scenario_bank(path)


def test_bank_rejects_incomplete_row_geometry():
    scenario = _scenario()
    scenario["single_book_slot_0"] = 1
    with pytest.raises(ValueError, match="cover every slot"):
        BANK.canonical_scenario(scenario)


def test_export_omits_outcomes_and_preserves_scenarios(tmp_path):
    csv_path = tmp_path / "episodes.csv"
    scenario = _scenario()
    row = dict(scenario)
    row.update({"episode_index": 0, "outcome": "success", "episode_reward": 99.0})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scenario_trace_complete": True,
                "scenario_sha256": "source-trace-hash",
                "episodes_csv": str(csv_path),
                "metadata": {"seed": 123},
            }
        )
    )

    bank = BANK.build_frozen_scenario_bank(summary_path)
    assert bank["scenario_count"] == 1
    assert bank["scenarios"][0]["scenario_id"] == 0
    assert "outcome" not in bank["scenarios"][0]
    assert "episode_reward" not in bank["scenarios"][0]


def test_policy_independent_generation_is_seeded_and_valid():
    kwargs = {
        "scenario_count": 20,
        "slot_clearance_min": 0.003,
        "slot_clearance_max": 0.003,
        "slot_pitch": 0.034,
        "row_book_count": 10,
        "side_book_merge_probability": 0.35,
        "arm_joint_noise": 0.02,
        "grasp_x_jitter": 0.003,
        "grasp_y_jitter": 0.003,
        "grasp_z_jitter": 0.0015,
        "grasp_yaw_jitter": 0.05,
    }
    first = BANK.generate_frozen_scenario_bank(seed=42, **kwargs)
    repeated = BANK.generate_frozen_scenario_bank(seed=42, **kwargs)
    different = BANK.generate_frozen_scenario_bank(seed=123, **kwargs)
    assert first["scenario_sha256"] == repeated["scenario_sha256"]
    assert first["scenario_sha256"] != different["scenario_sha256"]
    assert all(scenario["slot_clearance"] == 0.003 for scenario in first["scenarios"])


def test_policy_independent_generation_can_cycle_requested_slots():
    bank = BANK.generate_frozen_scenario_bank(
        scenario_count=10,
        seed=42,
        slot_clearance_min=0.002,
        slot_clearance_max=0.004,
        slot_pitch=0.034,
        row_book_count=10,
        side_book_merge_probability=0.35,
        arm_joint_noise=0.02,
        grasp_x_jitter=0.003,
        grasp_y_jitter=0.003,
        grasp_z_jitter=0.005,
        grasp_yaw_jitter=0.05,
        missing_book_indices=range(10),
    )
    assert [scenario["missing_book_index"] for scenario in bank["scenarios"]] == list(range(10))
    assert bank["source"]["generation"]["missing_book_indices"] == list(range(10))


def test_policy_independent_generation_rejects_wrong_slot_sequence_length():
    with pytest.raises(ValueError, match="exactly scenario_count"):
        BANK.generate_frozen_scenario_bank(
            scenario_count=10,
            seed=42,
            slot_clearance_min=0.002,
            slot_clearance_max=0.004,
            slot_pitch=0.034,
            row_book_count=10,
            side_book_merge_probability=0.35,
            arm_joint_noise=0.02,
            grasp_x_jitter=0.003,
            grasp_y_jitter=0.003,
            grasp_z_jitter=0.005,
            grasp_yaw_jitter=0.05,
            missing_book_indices=[0, 1],
        )
