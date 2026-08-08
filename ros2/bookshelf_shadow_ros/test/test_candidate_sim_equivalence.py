import numpy as np

from bookshelf_shadow_ros.candidate_sim_equivalence import (
    compare_candidate_to_simulator_preinsert,
)


def test_matching_candidate_passes_simulator_equivalence():
    observation = np.zeros((4, 12), dtype=np.float64)
    observation[:, 1] = -1.0
    observation[:, 2] = 1.0
    normalized = observation * 2.0
    actor = np.tile([2.0, -2.0, 3.0, -3.0, 4.0, -4.0], (4, 1))
    action = np.clip(actor, -1.0, 1.0)

    result = compare_candidate_to_simulator_preinsert(
        observation,
        normalized,
        actor,
        action,
        observation[0],
        normalized[0],
        actor[0],
        action[0],
    )

    assert result["equivalence_passed"]
    assert result["observation"]["outside_envelope_labels"] == []
    assert result["action"]["mismatching_labels"] == []


def test_observation_frame_mismatch_fails_equivalence():
    observation = np.zeros((3, 12), dtype=np.float64)
    normalized = np.zeros((3, 12), dtype=np.float64)
    actor = np.zeros((3, 6), dtype=np.float64)
    action = np.zeros((3, 6), dtype=np.float64)
    candidate = np.zeros(12, dtype=np.float64)
    candidate[3] = 0.25

    result = compare_candidate_to_simulator_preinsert(
        observation,
        normalized,
        actor,
        action,
        candidate,
        candidate,
        actor[0],
        action[0],
    )

    assert not result["equivalence_passed"]
    assert result["observation"]["outside_envelope_labels"] == ["lat_err"]
