import os
from glob import glob

from setuptools import find_packages, setup


package_name = "bookshelf_shadow_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/bookshelf_shadow_ros"]),
        ("share/bookshelf_shadow_ros", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.json")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.xacro")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chris",
    maintainer_email="tianyuli19981009@gmail.com",
    description="Read-only RGB-D slot detection and policy observation adaptation.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "policy_observation_adapter = "
            "bookshelf_shadow_ros.policy_observation_adapter_node:main",
            "policy_shadow_inference = "
            "bookshelf_shadow_ros.policy_shadow_inference_node:main",
            "rgbd_slot_detector = bookshelf_shadow_ros.rgbd_slot_detector:main",
            "sim_rgb_camera_viewer = bookshelf_shadow_ros.sim_rgb_camera_viewer:main",
            "slot_detection_audit = "
            "bookshelf_shadow_ros.slot_detection_audit_node:main",
            "slot_orientation_audit = "
            "bookshelf_shadow_ros.slot_orientation_audit_node:main",
            "static_slot_environment_check = "
            "bookshelf_shadow_ros.static_slot_environment_check_node:main",
            "static_slot_capture = "
            "bookshelf_shadow_ros.static_slot_capture_node:main",
            "policy_stream_audit = "
            "bookshelf_shadow_ros.policy_stream_audit_node:main",
            "marker_book_calibrator = "
            "bookshelf_shadow_ros.marker_book_calibration_node:main",
            "calibrated_preinsert_target = "
            "bookshelf_shadow_ros.calibrated_preinsert_target_node:main",
            "policy_tool_frame_audit = "
            "bookshelf_shadow_ros.policy_tool_frame_audit_node:main",
            "experiment_logger = "
            "bookshelf_shadow_ros.experiment_logger_node:main",
            "physical_experiment_preflight = "
            "bookshelf_shadow_ros.physical_experiment_preflight:main",
            "book_calibration_candidate_check = "
            "bookshelf_shadow_ros.book_calibration_candidate_check:main",
            "supervised_book_calibration_candidate = "
            "bookshelf_shadow_ros.supervised_book_calibration_candidate:main",
            "offline_scene_visualizer = "
            "bookshelf_shadow_ros.offline_scene_visualizer_node:main",
            "eef_tcp_context_capture = "
            "bookshelf_shadow_ros.eef_tcp_context_capture_node:main",
            "ros_release_geometry = "
            "bookshelf_shadow_ros.ros_release_geometry_node:main",
            "stationary_capture_pipeline = "
            "bookshelf_shadow_ros.stationary_capture_pipeline:main",
            "stationary_shadow_replay = "
            "bookshelf_shadow_ros.stationary_shadow_replay:main",
            "stationary_shadow_replay_audit = "
            "bookshelf_shadow_ros.stationary_shadow_replay_audit_node:main",
            "promote_stationary_calibration = "
            "bookshelf_shadow_ros.stationary_calibration_promotion:main",
        ],
    },
)
