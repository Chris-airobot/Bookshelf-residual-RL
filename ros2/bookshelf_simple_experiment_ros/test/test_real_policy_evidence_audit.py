import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[3] / "scripts" / "sb3" / "real_policy_evidence_audit.py"
SPEC = importlib.util.spec_from_file_location("real_policy_evidence_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_classification_separates_complete_partial_and_debug():
    complete = AUDIT.classify_log(["calculated", "calculated", "release_requested", "push_started", "episode_complete"], 2)
    partial = AUDIT.classify_log(["calculated", "calculated"], 2)
    debug = AUDIT.classify_log(["calculated", "complete"], 1)
    assert complete["classification"] == "complete_episode"
    assert complete["release"] and complete["push"] and complete["episode_complete"]
    assert partial["classification"] == "partial_insert"
    assert debug["classification"] == "debug_one_step"


def test_contamination_is_conservative_with_missing_evidence():
    assert AUDIT.contamination_label(False, False, True, False) == "UNKNOWN"
    assert AUDIT.contamination_label(True, True, False, False) == "UNKNOWN"
    assert AUDIT.contamination_label(True, True, True, True) == "BOTH"
    assert AUDIT.contamination_label(True, True, True, False) == "SLOT_STALE"
    assert AUDIT.contamination_label(True, False, True, True) == "MARKER_STALE"
    assert AUDIT.contamination_label(True, False, True, False) == "CLEAN"


def test_transform_rotation_error():
    identity = np.eye(4)
    rotated = np.eye(4)
    angle = np.deg2rad(3.0)
    rotated[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    assert abs(AUDIT.rotation_error_degrees(identity, rotated) - 3.0) < 1.0e-9
