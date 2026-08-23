#!/usr/bin/env python3
"""Compare synchronized Panda and xArm release-geometry diagnostic files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "bookshelf_release_geometry_diagnostic":
        raise ValueError(f"not a release geometry diagnostic: {path}")
    if not value.get("release", {}).get("accepted", False):
        raise ValueError(f"diagnostic did not capture an accepted release: {path}")
    if value.get("source") == "ros_xarm_read_only" and not value["release"].get(
        "release_requested_diagnostic", False
    ):
        raise ValueError(
            f"ROS diagnostic is a geometry smoke snapshot, not an INSERT release: {path}"
        )
    return value


def _mm(value: float) -> str:
    return f"{1000.0 * float(value):+.3f} mm"


def _xyz_mm(values: list[float]) -> str:
    return "[" + ", ".join(f"{1000.0 * float(value):+.3f}" for value in values) + "] mm"


def _closest(value: dict) -> str:
    pair = value["physical_gripper_to_shelf"]["closest_body_obstacle_pair"]
    return f"{pair['body']} -> {pair['obstacle']}: {_mm(pair['distance_m'])}"


def _quat_difference_deg(a: list[float], b: list[float]) -> float:
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panda", type=Path)
    parser.add_argument("xarm", type=Path)
    args = parser.parse_args()

    panda = _load(args.panda)
    xarm = _load(args.xarm)
    rows = (
        (
            "Trailing edge depth from mouth",
            panda["book"]["trailing_edge_depth_from_mouth_m"],
            xarm["book"]["trailing_edge_depth_from_mouth_m"],
        ),
        (
            "Leading edge penetration",
            panda["book"]["leading_edge_penetration_from_mouth_m"],
            xarm["book"]["leading_edge_penetration_from_mouth_m"],
        ),
        (
            "Front-to-back remaining",
            panda["book"]["front_to_back_remaining_m"],
            xarm["book"]["front_to_back_remaining_m"],
        ),
    )
    print("===== RELEASE DEPTH =====")
    for label, panda_value, xarm_value in rows:
        print(
            f"{label}: Panda {_mm(panda_value)}, xArm {_mm(xarm_value)}, "
            f"xArm-Panda {_mm(float(xarm_value) - float(panda_value))}"
        )

    print("\n===== FRAME OFFSETS =====")
    for label, value in (("Panda", panda), ("xArm", xarm)):
        virtual = value["virtual_policy_tool"]
        print(f"{label} book->TCP translation: {_xyz_mm(virtual['book_to_tcp']['position_xyz_m'])}")
        print(
            f"{label} TCP->policy-tool translation: "
            f"{_xyz_mm(virtual['tcp_to_policy_tool']['position_xyz_m'])}"
        )
    print(
        "Book-relative TCP orientation difference: "
        f"{_quat_difference_deg(panda['virtual_policy_tool']['book_to_tcp']['quaternion_wxyz'], xarm['virtual_policy_tool']['book_to_tcp']['quaternion_wxyz']):.3f} deg"
    )

    print("\n===== PHYSICAL GRIPPER ENVELOPES =====")
    print(f"Panda closest conservative pair: {_closest(panda)}")
    print(f"xArm closest conservative pair: {_closest(xarm)}")
    print(
        "Method: conservative collision-shape AABB envelopes; zero means "
        "envelope overlap, not proven mesh contact."
    )

    print("\n===== BODY OPENING MARGINS =====")
    for label, value in (("Panda", panda), ("xArm", xarm)):
        print(label)
        for body in value["physical_gripper_to_shelf"]["bodies"]:
            margins = body["opening_margins"]
            print(
                f"  {body['name']}: mouth={_mm(margins['mouth_to_body_nearest_x_m'])} "
                f"left={_mm(margins['left_channel_margin_m'])} "
                f"right={_mm(margins['right_channel_margin_m'])} "
                f"deck={_mm(margins['deck_margin_m'])}"
            )


if __name__ == "__main__":
    main()
