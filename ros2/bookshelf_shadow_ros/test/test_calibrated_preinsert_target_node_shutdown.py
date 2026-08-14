from pathlib import Path


NODE = (
    Path(__file__).parents[1]
    / "bookshelf_shadow_ros"
    / "calibrated_preinsert_target_node.py"
)


def test_main_guards_shutdown_after_launch_sigint():
    source = NODE.read_text(encoding="utf-8")
    finally_block = source[source.index("    finally:\n", source.index("def main")) :]
    assert "node.destroy_node()" in finally_block
    assert "if rclpy.ok():\n            rclpy.shutdown()" in finally_block
