import numpy as np
import pytest

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.policy_tool_frame_audit import (
    candidate_frame_names,
    evaluate_policy_tool_candidate,
    midpoint_transform,
    summarize_candidates,
)


def test_candidate_discovery_includes_link_tcp():
    candidates = candidate_frame_names(
        ["link_eef"],
        ["link_base", "link_tcp", "camera_color_optical_frame"],
    )

    assert candidates == ["link_eef", "link_tcp"]


def test_link_eef_identity_is_outside_training_tool_distance():
    transform_eef_book = make_transform([0.0, 0.0, 0.125])
    candidate = evaluate_policy_tool_candidate(
        "link_eef",
        np.eye(4),
        transform_eef_book,
        source="tf_frame",
    )

    assert candidate["tool_to_book_norm_m"] == pytest.approx(0.125)
    assert not candidate["within_conservative_training_norm_range"]
    assert not candidate["selection_authorized"]


def test_policy_equivalent_candidate_matches_training_reference():
    transform_eef_book = make_transform([0.01, -0.02, 0.12])
    transform_book_tool = make_transform([0.032, 0.0, 0.0])
    transform_eef_tool = transform_eef_book @ transform_book_tool
    candidate = evaluate_policy_tool_candidate(
        "virtual_policy_tool",
        transform_eef_tool,
        transform_eef_book,
        source="configured_virtual",
    )

    np.testing.assert_allclose(
        candidate["tool_to_book_translation_book_m"],
        [0.032, 0.0, 0.0],
        atol=1.0e-12,
    )
    assert candidate["within_conservative_training_norm_range"]


def test_midpoint_uses_average_finger_position():
    left = make_transform([0.10, 0.02, 0.03])
    right = make_transform([0.10, -0.02, 0.03])
    midpoint = midpoint_transform(left, right)
    np.testing.assert_allclose(midpoint[:3, 3], [0.10, 0.0, 0.03])


def test_summary_ranks_but_never_selects_candidate():
    transform_eef_book = np.eye(4)
    far = evaluate_policy_tool_candidate(
        "far", make_transform([0.12, 0.0, 0.0]), transform_eef_book, source="test"
    )
    near = evaluate_policy_tool_candidate(
        "near", make_transform([0.033, 0.0, 0.0]), transform_eef_book, source="test"
    )
    result = summarize_candidates([far, near])

    assert result["ranked_candidate_names"][0] == "near"
    assert result["plausible_candidate_names"] == ["near"]
    assert result["selection_required"]
    assert not result["selection_authorized"]
