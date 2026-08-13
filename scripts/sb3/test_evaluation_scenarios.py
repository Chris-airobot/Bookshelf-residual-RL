import json

from evaluation_scenarios import (
    EvaluationScenarioTrace,
    SCENARIO_FIELDS,
    apply_evaluation_seed_after_agent_load,
    scenario_sha256,
)


def _row(env_id: int, reset_count: int, offset: float = 0.0) -> dict:
    row = {field: 0 for field in SCENARIO_FIELDS}
    row.update(
        {
            "episode_index": reset_count,
            "env_id": env_id,
            "reset_count": reset_count,
            "slot_center_y": offset,
            "outcome": "success",
            "failure_code": 1,
            "episode_reward": 100.0,
            "episode_length": 50,
        }
    )
    return row


def test_scenario_hash_ignores_completion_order():
    rows = [_row(2, 1), _row(0, 4), _row(1, 3)]
    assert scenario_sha256(rows) == scenario_sha256(list(reversed(rows)))


def test_scenario_hash_changes_with_reset_condition():
    rows = [_row(0, 0), _row(1, 0)]
    changed = [_row(0, 0), _row(1, 0, offset=0.001)]
    assert scenario_sha256(rows) != scenario_sha256(changed)


def test_scenario_hash_ignores_realized_physics_pose():
    rows = [_row(0, 0), _row(1, 0)]
    changed = [_row(0, 0), _row(1, 0)]
    changed[1]["initial_book_x"] = 0.00001
    changed[1]["initial_tool_qw"] = 0.99999
    assert scenario_sha256(rows) == scenario_sha256(changed)


def test_evaluation_seed_is_reapplied_after_checkpoint_load():
    class FakeAgent:
        seed = 42

        def __init__(self):
            self.applied_seeds = []

        def set_random_seed(self, seed):
            self.applied_seeds.append(seed)

    agent = FakeAgent()
    checkpoint_seed = apply_evaluation_seed_after_agent_load(agent, 123)
    assert checkpoint_seed == 42
    assert agent.applied_seeds == [123]


def test_trace_writes_complete_summary(tmp_path):
    trace = EvaluationScenarioTrace(tmp_path / "trace", {"seed": 42})
    trace.append(_row(0, 0))
    summary_path = trace.write()
    summary = json.loads(summary_path.read_text())
    assert summary["episode_count"] == 1
    assert summary["scenario_trace_complete"] is True
    assert summary["outcomes"] == {"success": 1}
    assert len(summary["scenario_sha256"]) == 64


def test_trace_reports_complete_frozen_bank_coverage(tmp_path):
    trace = EvaluationScenarioTrace(
        tmp_path / "trace",
        {"frozen_scenario_bank": {"scenario_count": 2, "scenario_sha256": "bank"}},
    )
    first = _row(0, 0)
    first["scenario_bank_index"] = 0
    second = _row(1, 0)
    second["scenario_bank_index"] = 1
    trace.append(first)
    trace.append(second)
    summary = json.loads(trace.write().read_text())
    assert summary["frozen_scenario_bank_coverage"] == {
        "complete": True,
        "duplicate_indices": [],
        "expected_count": 2,
        "missing_indices": [],
        "observed_count": 2,
        "unexpected_indices": [],
    }
