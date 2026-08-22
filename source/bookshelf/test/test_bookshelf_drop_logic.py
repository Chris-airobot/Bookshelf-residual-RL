from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "source/bookshelf/bookshelf/tasks/direct/bookshelf/drop_logic.py"
SOURCE = MODULE_PATH.read_text(encoding="utf-8")


def test_drop_logic_selects_true_ground_only_during_insert():
    assert "threshold = torch.where(" in SOURCE
    assert "mode == int(insert_mode)" in SOURCE
    assert "torch.full_like(lowest_z, float(true_ground_z))" in SOURCE
    assert "torch.full_like(lowest_z, float(shelf_drop_z))" in SOURCE


def test_drop_logic_requires_crossing_the_selected_threshold_off_shelf():
    assert "return (lowest_z <= threshold) & ~on_shelf" in SOURCE
