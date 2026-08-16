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
