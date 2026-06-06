from setuptools import find_packages, setup

package_name = 'bookshelf_policy_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chris',
    maintainer_email='tianyuli19981009@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_node = bookshelf_policy_ros.test_node:main',
            'move_to_joint_pose = bookshelf_policy_ros.move_to_joint_pose:main',
            'action_debug_node = bookshelf_policy_ros.action_debug_node:main',
            'action_executor_node = bookshelf_policy_ros.action_executor_node:main',
            'cartesian_action_executor_node = bookshelf_policy_ros.cartesian_action_executor_node:main',
            'observation_debug_node = bookshelf_policy_ros.observation_debug_node:main',
            'policy_dry_run_node = bookshelf_policy_ros.policy_dry_run_node:main',
            'policy_to_robot_node = bookshelf_policy_ros.policy_to_robot_node:main',
        ],
    },
)
