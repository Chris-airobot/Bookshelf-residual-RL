"""Pure configuration and reporting helpers for the xArm reset preflight."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PreflightProfile:
    """One reproducible grasp and scene-randomization tier."""

    name: str
    grasp_translation_abs_m: tuple[float, float, float]
    grasp_yaw_abs_rad: float
    arm_joint_noise_abs_rad: float
    slot_clearance_range_m: tuple[float, float]

    def validate(self) -> None:
        if not self.name:
            raise ValueError("profile name cannot be empty")
        if len(self.grasp_translation_abs_m) != 3:
            raise ValueError("grasp translation limits must contain three values")
        if any(not math.isfinite(value) or value < 0.0 for value in self.grasp_translation_abs_m):
            raise ValueError("grasp translation limits must be finite and non-negative")
        if not math.isfinite(self.grasp_yaw_abs_rad) or self.grasp_yaw_abs_rad < 0.0:
            raise ValueError("grasp yaw limit must be finite and non-negative")
        if not math.isfinite(self.arm_joint_noise_abs_rad) or self.arm_joint_noise_abs_rad < 0.0:
            raise ValueError("arm joint-noise limit must be finite and non-negative")
        clearance_min, clearance_max = self.slot_clearance_range_m
        if (
            not math.isfinite(clearance_min)
            or not math.isfinite(clearance_max)
            or clearance_min < 0.0
            or clearance_min > clearance_max
        ):
            raise ValueError("slot-clearance limits must be finite, non-negative, and ordered")

    def document(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["grasp_translation_abs_mm"] = [
            1000.0 * value for value in self.grasp_translation_abs_m
        ]
        result["grasp_yaw_abs_deg"] = math.degrees(self.grasp_yaw_abs_rad)
        result["arm_joint_noise_abs_deg"] = math.degrees(
            self.arm_joint_noise_abs_rad
        )
        result["slot_clearance_range_mm"] = [
            1000.0 * value for value in self.slot_clearance_range_m
        ]
        return result


DEFAULT_PROFILES = (
    PreflightProfile(
        name="current",
        grasp_translation_abs_m=(0.003, 0.003, 0.005),
        grasp_yaw_abs_rad=math.radians(3.0),
        arm_joint_noise_abs_rad=math.radians(2.0),
        slot_clearance_range_m=(0.002, 0.004),
    ),
    PreflightProfile(
        name="medium",
        grasp_translation_abs_m=(0.004, 0.004, 0.008),
        grasp_yaw_abs_rad=math.radians(5.0),
        arm_joint_noise_abs_rad=math.radians(2.5),
        slot_clearance_range_m=(0.001, 0.005),
    ),
    PreflightProfile(
        name="hard",
        grasp_translation_abs_m=(0.005, 0.005, 0.010),
        grasp_yaw_abs_rad=math.radians(7.0),
        arm_joint_noise_abs_rad=math.radians(3.0),
        slot_clearance_range_m=(0.0, 0.006),
    ),
)


def profile_documents(
    profiles: Iterable[PreflightProfile] = DEFAULT_PROFILES,
) -> list[dict[str, Any]]:
    documents = [profile.document() for profile in profiles]
    names = [document["name"] for document in documents]
    if len(names) != len(set(names)):
        raise ValueError("preflight profile names must be unique")
    return documents


def summarize_preflight_rows(
    rows: list[dict[str, Any]],
    *,
    profile_order: Iterable[str],
    minimum_pass_rate: float,
) -> dict[str, Any]:
    """Aggregate sample results without importing Isaac Lab."""

    if not 0.0 <= minimum_pass_rate <= 1.0:
        raise ValueError("minimum_pass_rate must be in [0, 1]")
    order = list(profile_order)
    if not order or len(order) != len(set(order)):
        raise ValueError("profile_order must contain unique profile names")

    profile_summaries: dict[str, Any] = {}
    for profile_name in order:
        profile_rows = [row for row in rows if row.get("profile") == profile_name]
        slot_summaries = {}
        for slot in range(10):
            slot_rows = [
                row for row in profile_rows if int(row["missing_book_index"]) == slot
            ]
            passed = sum(bool(row.get("passed")) for row in slot_rows)
            slot_summaries[str(slot)] = {
                "samples": len(slot_rows),
                "passed": passed,
                "pass_rate": passed / len(slot_rows) if slot_rows else None,
            }

        passed = sum(bool(row.get("passed")) for row in profile_rows)
        reasons: dict[str, int] = {}
        for row in profile_rows:
            for reason in str(row.get("failure_reasons", "")).split(";"):
                reason = reason.strip()
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
        profile_summaries[profile_name] = {
            "samples": len(profile_rows),
            "passed": passed,
            "pass_rate": passed / len(profile_rows) if profile_rows else None,
            "meets_minimum_pass_rate": bool(profile_rows)
            and passed / len(profile_rows) >= minimum_pass_rate,
            "failure_reasons": dict(sorted(reasons.items())),
            "slots": slot_summaries,
        }

    recommended = None
    for profile_name in order:
        if profile_summaries[profile_name]["meets_minimum_pass_rate"]:
            recommended = profile_name
        else:
            break

    return {
        "sample_count": len(rows),
        "minimum_pass_rate": minimum_pass_rate,
        "recommended_profile": recommended,
        "profiles": profile_summaries,
    }
