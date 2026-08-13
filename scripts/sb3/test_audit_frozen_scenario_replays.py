import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/audit_frozen_scenario_replays.py"
SPEC = importlib.util.spec_from_file_location("audit_frozen_scenario_replays_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_audit_requires_matching_bank_and_complete_coverage(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"scenario_sha256": "abc", "scenario_count": 2}))
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "episode_count": 2,
                "scenario_trace_complete": True,
                "metadata": {
                    "frozen_scenario_bank": {"scenario_sha256": "abc", "scenario_count": 2}
                },
                "frozen_scenario_bank_coverage": {"complete": True},
            }
        )
    )
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "episode_count": 1,
                "scenario_trace_complete": True,
                "metadata": {
                    "frozen_scenario_bank": {"scenario_sha256": "wrong", "scenario_count": 2}
                },
                "frozen_scenario_bank_coverage": {"complete": False},
            }
        )
    )
    assert AUDIT.audit_replays(bank, [good])["passed"] is True
    report = AUDIT.audit_replays(bank, [good, bad])
    assert report["passed"] is False
    assert report["checks"][1]["reasons"] == [
        "bank hash mismatch",
        "bank coverage incomplete",
        "episode count mismatch",
    ]
