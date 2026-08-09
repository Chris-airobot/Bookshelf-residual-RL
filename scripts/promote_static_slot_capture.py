#!/usr/bin/env python3
"""Promote a visually reviewed slot capture into one trial ROS config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "ros2" / "bookshelf_shadow_ros"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from bookshelf_shadow_ros.static_slot_capture import (  # noqa: E402
    APPROVAL_TOKEN,
    promote_capture_candidate,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a trial-specific slot configuration after manual RViz review. "
            "This does not launch ROS or command hardware."
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--template-dir",
        default=str(ROOT / "ros2" / "bookshelf_shadow_ros" / "config"),
    )
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args()

    provenance = promote_capture_candidate(
        args.candidate,
        args.template_dir,
        args.output,
        approval_token=args.approval_token,
    )
    print(f"Trial configuration: {provenance['trial_config']}")
    print(
        "Provenance: "
        + str(Path(provenance["trial_config"]).with_suffix(".provenance.json"))
    )
    print(f"Candidate SHA256: {provenance['candidate_report_sha256']}")
    print(f"Transform status: {provenance['transform_status']}")
    print("Hardware commanded: False")


if __name__ == "__main__":
    main()
