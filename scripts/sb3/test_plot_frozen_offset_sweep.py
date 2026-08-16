import importlib.util
import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/plot_frozen_offset_sweep.py"
SPEC = importlib.util.spec_from_file_location("plot_frozen_offset_sweep_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def test_plot_series_is_numeric_and_sorted():
    rows = [
        {
            "offset_scale": "1.5",
            "method": "residual_ppo",
            "mean_success_pct": "70.0",
            "sample_stdev_success_percentage_points": "4.0",
        },
        {
            "offset_scale": "0.5",
            "method": "residual_ppo",
            "mean_success_pct": "95.0",
            "sample_stdev_success_percentage_points": "2.0",
        },
        {
            "offset_scale": "0.5",
            "method": "nominal_only",
            "mean_success_pct": "20.0",
            "sample_stdev_success_percentage_points": "0.0",
        },
    ]
    series = PLOT.build_plot_series(rows)
    assert series["residual_ppo"] == {
        "x": [0.5, 1.5],
        "mean": [95.0, 70.0],
        "sd": [2.0, 4.0],
    }


def test_plot_series_rejects_duplicate_method_scale():
    row = {
        "offset_scale": "1.0",
        "method": "residual_ppo",
        "mean_success_pct": "90.0",
    }
    with pytest.raises(ValueError, match="Duplicate offset scale"):
        PLOT.build_plot_series([row, dict(row)])


def test_renderer_writes_pdf_svg_and_png(tmp_path):
    pytest.importorskip("matplotlib")
    summary = tmp_path / "offset_method_summary.csv"
    rows = []
    for scale in (0.0, 1.0, 1.5):
        for method, mean, stdev in (
            ("nominal_only", 25.0, 0.0),
            ("ppo_only", 0.0, 0.0),
            ("residual_ppo", 90.0 - 20.0 * scale, 3.0),
        ):
            rows.append(
                {
                    "offset_scale": scale,
                    "method": method,
                    "mean_success_pct": mean,
                    "sample_stdev_success_percentage_points": stdev,
                }
            )
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    outputs = PLOT.render_figure(summary, tmp_path / "offset_robustness")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 0 for path in outputs)
