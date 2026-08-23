from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "experiment_logging.launch.py"
LOGGER = ROOT / "bookshelf_shadow_ros" / "experiment_logger_node.py"


def test_logging_can_capture_direct_offline_replay_inputs():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"/camera/color/image_raw"',
        '"/camera/depth/image_rect_raw"',
        '"/camera/aligned_depth_to_color/image_raw"',
        '"/robot_description"',
        '"record_raw_replay_inputs"',
    )
    for token in required:
        assert token in source


def test_capture_condition_is_written_to_manifest():
    launch_source = LAUNCH.read_text(encoding="utf-8")
    logger_source = LOGGER.read_text(encoding="utf-8")
    assert '"capture_condition": LaunchConfiguration("capture_condition")' in launch_source
    assert '"capture_condition": str(' in logger_source
    assert '"raw_replay_inputs_recorded": bool(' in logger_source


def test_manifest_is_finalized_before_shutdown_graph_access():
    source = LOGGER.read_text(encoding="utf-8")
    destroy_source = source[source.index("    def destroy_node(self):") :]
    manifest_index = destroy_source.index("self._write_manifest(completed=True)")
    graph_index = destroy_source.index("self._write_graph_snapshot()")
    assert manifest_index < graph_index
    assert "except (KeyboardInterrupt, ExternalShutdownException):" in source


def test_logger_records_simulation_control_topics_to_readable_text():
    source = LOGGER.read_text(encoding="utf-8")
    for token in (
        'self.monitor_path = self.run_dir / "monitor.txt"',
        '"/bookshelf_sim/task_status"',
        '"/bookshelf_sim/task_complete"',
        '"/bookshelf_control/status"',
        '"/bookshelf_shadow/final_delta"',
        '"/servo_server/status"',
        "self._subscribe_float_array",
        "self._subscribe_int",
    ):
        assert token in source


def test_logger_records_physical_episode_status_and_completion():
    launch_source = LAUNCH.read_text(encoding="utf-8")
    logger_source = LOGGER.read_text(encoding="utf-8")
    for token in (
        '"/bookshelf_control/task_status"',
        '"/bookshelf_control/task_complete"',
    ):
        assert token in launch_source
        assert token in logger_source
    assert '"physical_task_status_topic", "physical_task_status"' in logger_source
    assert '"physical_task_complete_topic", "physical_task_complete"' in logger_source
