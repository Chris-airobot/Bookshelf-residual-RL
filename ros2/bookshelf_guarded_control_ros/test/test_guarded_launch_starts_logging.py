from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_guarded_single_step_starts_automatic_logging_by_default():
    source = (ROOT / "launch" / "guarded_policy_tool_single_step.launch.py").read_text(
        encoding="utf-8"
    )
    assert '"enable_logging"' in source
    assert 'default_value="true"' in source
    assert '"experiment_logging.launch.py"' in source
    assert "IncludeLaunchDescription" in source
    assert '"experiment_output_root"' in source
    assert '"policy_bundle"' in source
    assert '"activation_envelope"' in source
