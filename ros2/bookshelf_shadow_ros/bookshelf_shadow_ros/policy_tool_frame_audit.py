"""Pure geometry for auditing real-robot policy-tool frame candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrated_preinsert_target_math import transform_to_dict
from .policy_observation_math import invert_transform


TOOL_FRAME_KEYWORDS = ("finger", "gripper", "knuckle", "eef", "tool", "tcp")


@dataclass(frozen=True)
class TrainingToolReference:
    """Training-time tool-to-book distance derived from the simulator grasp."""

    hand_to_tool_m: float = 0.107
    hand_to_book_m: float = 0.075
    minimum_norm_m: float = 0.020
    maximum_norm_m: float = 0.050

    @property
    def nominal_norm_m(self) -> float:
        return abs(float(self.hand_to_tool_m) - float(self.hand_to_book_m))


def evaluate_policy_tool_candidate(
    name: str,
    transform_eef_tool,
    transform_eef_book,
    *,
    source: str,
    reference=TrainingToolReference(),
) -> dict:
    """Describe one candidate tool point relative to the calibrated book."""

    transform_eef_tool = _validated_transform(transform_eef_tool)
    transform_eef_book = _validated_transform(transform_eef_book)
    transform_book_tool = invert_transform(transform_eef_book) @ transform_eef_tool
    translation = transform_book_tool[:3, 3]
    distance = float(np.linalg.norm(translation))
    within_range = bool(reference.minimum_norm_m <= distance <= reference.maximum_norm_m)
    return {
        "name": str(name),
        "source": str(source),
        "available": True,
        "transform_eef_tool": transform_to_dict(transform_eef_tool),
        "transform_book_tool": transform_to_dict(transform_book_tool),
        "tool_to_book_translation_book_m": [float(value) for value in translation],
        "tool_to_book_norm_m": distance,
        "training_nominal_norm_m": reference.nominal_norm_m,
        "training_norm_error_m": abs(distance - reference.nominal_norm_m),
        "within_conservative_training_norm_range": within_range,
        "selection_authorized": False,
    }


def unavailable_candidate(name: str, source: str, reason: str) -> dict:
    return {
        "name": str(name),
        "source": str(source),
        "available": False,
        "reason": str(reason),
        "selection_authorized": False,
    }


def candidate_frame_names(configured_frames, known_frames, *, discover=True):
    """Combine configured frames with semantically relevant TF frame names."""

    candidates = [str(value) for value in configured_frames]
    if discover:
        candidates.extend(
            str(frame)
            for frame in known_frames
            if any(keyword in str(frame).lower() for keyword in TOOL_FRAME_KEYWORDS)
            and "camera" not in str(frame).lower()
        )
    return list(dict.fromkeys(candidates))


def midpoint_transform(transform_eef_left, transform_eef_right) -> np.ndarray:
    """Return a position-only virtual frame midway between two finger frames."""

    left = _validated_transform(transform_eef_left)
    right = _validated_transform(transform_eef_right)
    midpoint = np.eye(4, dtype=np.float64)
    midpoint[:3, 3] = 0.5 * (left[:3, 3] + right[:3, 3])
    return midpoint


def summarize_candidates(candidates, reference=TrainingToolReference()) -> dict:
    """Rank candidates without automatically selecting or authorizing one."""

    candidates = list(candidates)
    available = [value for value in candidates if value.get("available", False)]
    ranked = sorted(
        available,
        key=lambda value: float(value["training_norm_error_m"]),
    )
    plausible = [
        value["name"]
        for value in ranked
        if value["within_conservative_training_norm_range"]
    ]
    return {
        "training_reference": {
            "simulator_hand_to_tool_m": float(reference.hand_to_tool_m),
            "simulator_hand_to_book_m": float(reference.hand_to_book_m),
            "nominal_tool_to_book_norm_m": reference.nominal_norm_m,
            "conservative_norm_range_m": [
                float(reference.minimum_norm_m),
                float(reference.maximum_norm_m),
            ],
            "derivation": "abs(0.107 m IK tool offset - 0.075 m reset book offset)",
        },
        "candidate_count": len(candidates),
        "available_count": len(available),
        "plausible_candidate_names": plausible,
        "ranked_candidate_names": [value["name"] for value in ranked],
        "selection_required": True,
        "selection_authorized": False,
        "warning": (
            "Distance agreement alone cannot prove frame semantics. Review the "
            "candidate TF and gripper geometry before configuring the policy tool."
        ),
        "candidates": candidates,
    }


def _validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    return transform
