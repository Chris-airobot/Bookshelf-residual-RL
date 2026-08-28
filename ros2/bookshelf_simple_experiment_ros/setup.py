import os
from glob import glob

from setuptools import find_packages, setup

package_name = "bookshelf_simple_experiment_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chris",
    maintainer_email="tianyuli19981009@gmail.com",
    description="Minimal isolated RGB-D slot-to-xArm pre-insertion workflow.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": [
        "slot_detector = bookshelf_simple_experiment_ros.slot_detector_node:main",
        "saved_slot = bookshelf_simple_experiment_ros.saved_slot_node:main",
        "simple_preinsert = bookshelf_simple_experiment_ros.preinsert_node:main",
        "simple_policy_control = bookshelf_simple_experiment_ros.simple_policy_control_node:main",
        "fake_policy_start = bookshelf_simple_experiment_ros.fake_policy_start_node:main",
        "virtual_trigger = bookshelf_simple_experiment_ros.virtual_trigger_node:main",
        "dual_aruco_book = bookshelf_simple_experiment_ros.dual_aruco_book_node:main",
        "real_experiment_operator = bookshelf_simple_experiment_ros.operator_console_node:main",
    ]},
)
