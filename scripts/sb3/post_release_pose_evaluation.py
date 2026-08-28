"""Evaluation-only post-release book-pose estimation and CSV logging."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


Pose = tuple[float, float, float, float, float, float, float]


def _pose(values: Sequence[float]) -> Pose:
    if len(values) != 7:
        raise ValueError(f"pose must contain xyz + wxyz (7 values), got {len(values)}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _quat_normalized(quat: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in quat))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be nonzero")
    return tuple(float(value) / norm for value in quat)  # type: ignore[return-value]


def _quat_multiply(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _rotate(quat: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    q = _quat_normalized(quat)
    conjugate = (q[0], -q[1], -q[2], -q[3])
    rotated = _quat_multiply(_quat_multiply(q, (0.0, *vector)), conjugate)
    return rotated[1], rotated[2], rotated[3]


def compose_pose(parent: Sequence[float], child: Sequence[float]) -> Pose:
    """Return ``T_parent_child`` for two xyz+wxyz poses."""
    parent_pose = _pose(parent)
    child_pose = _pose(child)
    offset = _rotate(parent_pose[3:7], child_pose[0:3])
    position = tuple(parent_pose[index] + offset[index] for index in range(3))
    quaternion = _quat_normalized(_quat_multiply(parent_pose[3:7], child_pose[3:7]))
    return _pose((*position, *quaternion))


def orientation_error_rad(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the shortest rotation angle between two wxyz quaternions."""
    q_first = _quat_normalized(first)
    q_second = _quat_normalized(second)
    dot = abs(sum(a * b for a, b in zip(q_first, q_second)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


@dataclass(frozen=True)
class PoseSample:
    """Read-only simulator poses sampled at one control boundary."""

    tcp_base: Pose
    tcp_to_book: Pose
    gt_book_base: Pose
    slot_from_base_quaternion: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tcp_base", _pose(self.tcp_base))
        object.__setattr__(self, "tcp_to_book", _pose(self.tcp_to_book))
        object.__setattr__(self, "gt_book_base", _pose(self.gt_book_base))
        object.__setattr__(
            self,
            "slot_from_base_quaternion",
            _quat_normalized(self.slot_from_base_quaternion),
        )


_POSE_COMPONENTS = ("x", "y", "z", "qw", "qx", "qy", "qz")
_POSE_PREFIXES = (
    "tcp_release_base",
    "estimated_book_release_base",
    "gt_book_release_base",
    "gt_book_settled_base",
)
CSV_FIELDS = [
    "episode_index",
    "env_id",
    "release_step",
    *[
        f"{prefix}_{component}"
        for prefix in _POSE_PREFIXES
        for component in _POSE_COMPONENTS
    ],
    "estimate_error_slot_dx_m",
    "estimate_error_slot_dy_m",
    "estimate_error_slot_dz_m",
    "estimate_orientation_error_rad",
    "settling_slot_dx_m",
    "settling_slot_dy_m",
    "settling_slot_dz_m",
    "true_settling_displacement_m",
]


def post_release_row(
    *,
    episode_index: int,
    env_id: int,
    release_step: int,
    release: PoseSample,
    settled_gt_book_base: Sequence[float],
) -> dict[str, float | int]:
    """Compute one evaluation row without exposing ground truth to control."""
    settled = _pose(settled_gt_book_base)
    estimated = compose_pose(release.tcp_base, release.tcp_to_book)
    estimate_error_base = tuple(
        estimated[index] - release.gt_book_base[index] for index in range(3)
    )
    estimate_error_slot = _rotate(
        release.slot_from_base_quaternion, estimate_error_base
    )
    settling_base = tuple(
        settled[index] - release.gt_book_base[index] for index in range(3)
    )
    settling_slot = _rotate(release.slot_from_base_quaternion, settling_base)

    row: dict[str, float | int] = {
        "episode_index": int(episode_index),
        "env_id": int(env_id),
        "release_step": int(release_step),
        "estimate_error_slot_dx_m": estimate_error_slot[0],
        "estimate_error_slot_dy_m": estimate_error_slot[1],
        "estimate_error_slot_dz_m": estimate_error_slot[2],
        "estimate_orientation_error_rad": orientation_error_rad(
            estimated[3:7], release.gt_book_base[3:7]
        ),
        "settling_slot_dx_m": settling_slot[0],
        "settling_slot_dy_m": settling_slot[1],
        "settling_slot_dz_m": settling_slot[2],
        "true_settling_displacement_m": math.sqrt(
            sum(component**2 for component in settling_slot)
        ),
    }
    poses = (
        release.tcp_base,
        estimated,
        release.gt_book_base,
        settled,
    )
    for prefix, pose in zip(_POSE_PREFIXES, poses):
        for component, value in zip(_POSE_COMPONENTS, pose):
            row[f"{prefix}_{component}"] = float(value)
    return row


class PostReleasePoseCsv:
    """Capture release and pre-PUSH settled poses for vectorized evaluation."""

    def __init__(self, output_path: str | Path, num_envs: int):
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._stream.flush()
        self._episode_indices = [0] * int(num_envs)
        self._release_keys: list[tuple[int, int] | None] = [None] * int(num_envs)
        self._pending: dict[int, tuple[int, PoseSample]] = {}
        self.row_count = 0

    def observe(
        self,
        *,
        env_id: int,
        release_step: int,
        push_start_step: int,
        sample: Callable[[], PoseSample],
    ) -> dict[str, float | int] | None:
        """Observe phase buffers after a step; sample only on phase boundaries."""
        episode_index = self._episode_indices[env_id]
        release_key = (episode_index, int(release_step))
        if release_step >= 0 and self._release_keys[env_id] != release_key:
            self._pending[env_id] = (int(release_step), sample())
            self._release_keys[env_id] = release_key

        if push_start_step < 0 or env_id not in self._pending:
            return None

        captured_release_step, release = self._pending.pop(env_id)
        settled = sample().gt_book_base
        row = post_release_row(
            episode_index=episode_index,
            env_id=env_id,
            release_step=captured_release_step,
            release=release,
            settled_gt_book_base=settled,
        )
        self._writer.writerow(row)
        self._stream.flush()
        self.row_count += 1
        return row

    def episode_done(self, env_id: int) -> None:
        """Discard incomplete captures when the environment resets."""
        self._pending.pop(env_id, None)
        self._release_keys[env_id] = None
        self._episode_indices[env_id] += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

