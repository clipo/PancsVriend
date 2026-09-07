"""plot_style helpers: outlier-only overlays past 200 runs, the y-axis chance
marker, and the presence of every helper the analysis scripts import."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_tools"))
import plot_style  # noqa: E402


def test_every_helper_the_scripts_import_exists():
    for name in ("overlay_run_points", "violin_box_points", "step_stats_forward_filled",
                 "steps_to_fraction_of_final", "mark_chance_on_axis", "tukey_outliers"):
        assert callable(getattr(plot_style, name)), name


def test_tukey_outliers_are_the_box_plot_fliers():
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0, 1, 1000), [8.0, -7.5]])
    out = plot_style.tukey_outliers(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    span = 1.5 * (q3 - q1)
    assert set(out) == set(vals[(vals < q1 - span) | (vals > q3 + span)])
    assert 8.0 in out and -7.5 in out
    assert len(plot_style.tukey_outliers([1.0, 2.0])) == 0     # too few to define a box


@pytest.mark.parametrize("n,expect_all", [(150, True), (201, False), (10000, False)])
def test_overlay_draws_every_point_only_below_the_limit(n, expect_all):
    rng = np.random.default_rng(1)
    vals = rng.normal(0, 1, n)
    fig, ax = plt.subplots()
    plot_style.overlay_run_points(ax, [vals], [0])
    drawn = sum(len(c.get_offsets()) for c in ax.collections)
    assert drawn == (n if expect_all else 0)
    plt.close(fig)


def test_overlay_can_be_forced_to_draw_everything():
    fig, ax = plt.subplots()
    vals = np.linspace(0, 1, 5000)
    plot_style.overlay_run_points(ax, [vals], [0], max_points=None)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 5000
    plt.close(fig)


def test_chance_marker_labels_the_axis_unless_a_major_tick_is_there():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    plot_style.mark_chance_on_axis(ax, 0.125)
    assert [t.get_text() for t in ax.get_yticklabels(minor=True)] == ["chance"]
    assert any(len(l.get_xdata()) == 1 for l in ax.lines)      # the triangle

    fig2, ax2 = plt.subplots()
    ax2.set_ylim(0.4, 1.0)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plot_style.mark_chance_on_axis(ax2, 0.5)                    # share's chance level
    assert ax2.get_yticklabels(minor=True) == [] or \
        all(t.get_text() == "" for t in ax2.get_yticklabels(minor=True))
    assert any(len(l.get_xdata()) == 1 for l in ax2.lines)     # marker still drawn

    fig3, ax3 = plt.subplots()
    plot_style.mark_chance_on_axis(ax3, None)                   # no chance level known
    assert len(ax3.lines) == 0
    plt.close("all")
