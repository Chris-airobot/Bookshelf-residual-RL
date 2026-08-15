import os
from glob import glob

from setuptools import find_packages, setup


package_name = "bookshelf_guarded_control_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chris",
    maintainer_email="tianyuli19981009@gmail.com",
    description=(
        "Fail-closed virtual-policy-tool planning and explicitly approved "
        "single-step execution for the bookshelf task."
    ),
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "bookshelf_scene_manager = "
            "bookshelf_guarded_control_ros.bookshelf_scene_manager_node:main",
            "calibrated_preinsert_plan_only = "
            "bookshelf_guarded_control_ros.calibrated_preinsert_plan_only_node:main",
            "guarded_preinsert_executor = "
            "bookshelf_guarded_control_ros.guarded_preinsert_executor_node:main",
            "policy_tool_plan_checker = "
            "bookshelf_guarded_control_ros.policy_tool_plan_checker_node:main",
            "guarded_policy_tool_executor = "
            "bookshelf_guarded_control_ros.guarded_policy_tool_executor_node:main",
        ],
    },
)
