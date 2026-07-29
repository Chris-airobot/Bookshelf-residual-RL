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
            "policy_stream_audit = "
            "bookshelf_shadow_ros.policy_stream_audit_node:main",
        ],
    },
)
