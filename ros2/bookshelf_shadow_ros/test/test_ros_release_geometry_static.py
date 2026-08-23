from pathlib import Path


PACKAGE = Path(__file__).parents[1]


def test_release_geometry_capture_is_subscriber_only_and_installed():
    node = (PACKAGE / "bookshelf_shadow_ros" / "ros_release_geometry_node.py").read_text(
        encoding="utf-8"
    )
    setup = (PACKAGE / "setup.py").read_text(encoding="utf-8")
    launch = (PACKAGE / "launch" / "release_geometry_capture.launch.py").read_text(
        encoding="utf-8"
    )

    assert "create_subscription(" in node
    assert "create_publisher(" not in node
    assert "create_client(" not in node
    assert "ActionClient(" not in node
    assert "ros_release_geometry_node:main" in setup
    assert 'executable="ros_release_geometry"' in launch
    assert 'default_value="release_requested"' in launch
    assert '"task_release"' in node
    assert '"task_status_topic", "/bookshelf_sim/task_status"' in node
    assert "self._task_status_callback" in node
    assert 'payload.get("phase") != "opening"' in node
    assert '"task_status": self.pending_task_status' in node
