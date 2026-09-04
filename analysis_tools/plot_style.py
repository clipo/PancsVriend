"""Shared plot styling helpers for the analysis figures.

Single source of truth for house-style overlays so the same figure element is
not reimplemented (and drifted) in every plotting script. Import it either way,
since these scripts run both as `analysis_tools.<mod>` from the orchestrator and
directly with analysis_tools/ on sys.path:

    from analysis_tools.plot_style import overlay_run_points   # repo root on path
    from plot_style import overlay_run_points                  # analysis_tools/ on path
"""
import numpy as np

# Half-width of the horizontal spread. 0.055 keeps points inside the 0.18-wide
# boxes used throughout the house violin+box style.
POINT_JITTER = 0.055
# Fixed seed: these are research figures, so re-running the script must
# reproduce the identical image rather than reshuffling the points.
POINT_SEED = 0


def overlay_run_points(ax, plot_data, positions, jitter=POINT_JITTER,
                       seed=POINT_SEED, size=18, color="#222222"):
    """Scatter each individual observation on top of its box.

    The box only shows quartiles; with n as small as 5 the raw values are what
    tell you whether a wide box is a genuine spread or two tight clusters. Dark
    neutral points with a white ring stay legible against any scenario/model
    fill colour and keep overlapping observations individually countable.

    Args:
        ax: target axes (already carrying the violin/box artists).
        plot_data: sequence of 1-D value arrays, one per box.
        positions: x positions of the boxes, same order and length as plot_data.
        jitter: half-width of the horizontal spread, in x-axis data units.
        seed: RNG seed for the jitter; fixed so figures are reproducible.
        size: marker area passed to scatter.
        color: marker face colour.
    """
    rng = np.random.default_rng(seed)
    for pos, vals in zip(positions, plot_data):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            continue
        # A lone observation sits dead centre; jitter would misleadingly
        # suggest it carries an x value.
        x = (np.full(len(vals), float(pos)) if len(vals) == 1
             else pos + rng.uniform(-jitter, jitter, size=len(vals)))
        ax.scatter(x, vals, s=size, color=color, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)


def violin_box_points(ax, plot_data, positions, colors, jitter=POINT_JITTER,
                      seed=POINT_SEED, point_size=18):
    """The house distribution figure: violin + narrow box + every run as a point.

    Extracted from segregation_metrics_comparison (2026-08-28) so the per-metric
    panels can show the SAME data in the SAME way instead of a bare seaborn
    boxplot. That was the whole point of this module — one implementation, no
    drift — and two scripts drawing the same distribution differently had made
    the panels and the comparison figure look like different analyses.

    Args:
        ax: target axes.
        plot_data: sequence of 1-D value arrays, one per position.
        positions: x positions, same order and length as plot_data.
        colors: per-position face colour, same order and length.
        jitter/seed/point_size: forwarded to overlay_run_points.
    """
    parts = ax.violinplot(plot_data, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_edgecolor(col)
        pc.set_alpha(0.35)
        pc.set_linewidth(1.0)

    bp = ax.boxplot(plot_data, positions=positions,
                    widths=0.18, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(0.65)
        patch.set_linewidth(1.0)
    for med in bp['medians']:
        med.set_color('black')
        med.set_linewidth(1.2)
        med.set_zorder(4)   # above the run points so the median stays readable
    for wl in bp['whiskers']:
        wl.set_color('#777777')
        wl.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('#777777')
        cap.set_linewidth(1.0)

    overlay_run_points(ax, plot_data, positions, jitter=jitter, seed=seed,
                       size=point_size)
    return parts, bp


def step_stats_forward_filled(df, metric, step_col="step", run_col="run_id",
                              max_step=None):
    """Per-step mean, 95% CI half-width, and ACTIVE-run count, forward-filled.

    Converged runs stop writing rows, so a bare df.groupby(step)[metric].mean()
    averages only the SURVIVORS — the atypical runs still in motion — and the
    trace drifts purely from attrition. Measured on mistral/income_high_low
    (2026-08-28): 100 runs alive at step 4 giving mean DI 0.244, but 2 alive at
    step 9 giving 0.337, while every run's true final was 0.25. A converged
    run's grid is FROZEN, so carrying its last value forward is the honest
    continuation, not an approximation — the same convention
    vf_simulation_evaluation.load_batch uses.

    n_active is counted BEFORE the fill (it is what makes the strip meaningful)
    and is metric-independent: it depends only on how long each run lived, so
    one strip serves a whole multi-metric figure.

    Returns (mean, ci_halfwidth, n_active), each a Series indexed by step.
    """
    import numpy as np
    import pandas as pd  # local: plot_style is imported by non-pandas callers

    pv = df.pivot_table(index=run_col, columns=step_col, values=metric)
    lo = int(df[step_col].min())
    hi = int(max_step) if max_step is not None else int(df[step_col].max())
    pv = pv.reindex(columns=range(lo, hi + 1))
    n_active = pv.notna().sum(axis=0)
    pv = pv.ffill(axis=1)
    mean = pv.mean(axis=0)
    n = pv.count(axis=0).replace(0, np.nan)
    ci = 1.96 * pv.std(axis=0) / np.sqrt(n)
    return mean, ci, n_active
