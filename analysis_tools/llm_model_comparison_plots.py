"""Generate per-scenario LLM model comparison plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from experiment_list_for_analysis import (
    SCENARIO_LABELS,
    SCENARIOS,
    SCENARIOS_TO_PLOT,
    LLM_MODELS_TO_PLOT,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "axes.labelpad": 6,
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
})

METRICS = [
    "clusters",
    "switch_rate",
    "distance",
    "mix_deviation",
    "share",
    "ghetto_rate",
]

METRIC_LABELS = {
    "clusters": "Number of Clusters",
    "switch_rate": "Switch Rate",
    "distance": "Average Distance",
    "mix_deviation": "Mix Deviation",
    "share": "Segregation Share",
    "ghetto_rate": "Ghetto Formation Rate",
}


def generate_llm_model_comparisons(
    experiments_dir: Path,
    output_dir: Path,
) -> None:
    scenario_model_map, model_display = _collect_scenario_model_map(experiments_dir)
    scenario_model_map, model_display = _filter_models_to_plot(scenario_model_map, model_display)
    scenarios = _select_scenarios(scenario_model_map)
    if not scenarios:
        print("No scenarios found for LLM model comparisons; skipping.")
        return

    convergence_dir = output_dir / "convergence_patterns"
    metrics_dir = output_dir / "metrics_comparison"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        model_map = scenario_model_map.get(scenario, {})
        if not model_map:
            continue
        _plot_convergence_patterns(
            scenario,
            model_map,
            model_display,
            convergence_dir,
        )
        _plot_metrics_comparison(
            scenario,
            model_map,
            model_display,
            metrics_dir,
        )

    _write_scenario_coverage_report(
        scenario_model_map,
        model_display,
        output_dir,
    )
    _build_segregation_ranking_table(
        scenario_model_map,
        model_display,
        output_dir,
    )


def _select_scenarios(scenario_model_map: Dict[str, Dict[str, dict]]) -> List[str]:
    if SCENARIOS_TO_PLOT:
        return [s for s in SCENARIOS_TO_PLOT if s in scenario_model_map]
    return sorted(scenario_model_map.keys())


def _scenario_key_from_config(folder_name: str, scenario: str) -> str:
    if scenario == "baseline" and folder_name.startswith("llm_baseline"):
        return "llm_baseline"
    return scenario


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def _collect_scenario_model_map(
    experiments_dir: Path,
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, str]]:
    scenario_model_map: Dict[str, Dict[str, dict]] = {}
    model_display: Dict[str, str] = {}

    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        llm_model = payload.get("llm_model")
        scenario = payload.get("scenario")
        if not llm_model or not scenario:
            continue
        scenario_key = _scenario_key_from_config(entry.name, str(scenario))
        model_key = _normalize_model_name(str(llm_model))
        timestamp = payload.get("timestamp")

        model_display.setdefault(model_key, str(llm_model))

        scenario_map = scenario_model_map.setdefault(scenario_key, {})
        existing = scenario_map.get(model_key)
        if existing is None:
            scenario_map[model_key] = {
                "folder": entry.name,
                "timestamp": timestamp,
            }
            continue

        if _is_newer(timestamp, existing.get("timestamp")):
            scenario_map[model_key] = {
                "folder": entry.name,
                "timestamp": timestamp,
            }

    return scenario_model_map, model_display


def _filter_models_to_plot(
    scenario_model_map: Dict[str, Dict[str, dict]],
    model_display: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, str]]:
    if not LLM_MODELS_TO_PLOT:
        return scenario_model_map, model_display

    allowed = {_normalize_model_name(name) for name in LLM_MODELS_TO_PLOT}
    filtered_map: Dict[str, Dict[str, dict]] = {}
    filtered_display: Dict[str, str] = {}

    for scenario, models in scenario_model_map.items():
        kept = {}
        for model_key, payload in models.items():
            if model_key in allowed:
                kept[model_key] = payload
                filtered_display[model_key] = model_display.get(model_key, model_key)
        if kept:
            filtered_map[scenario] = kept

    return filtered_map, filtered_display


def _is_newer(candidate: Optional[str], current: Optional[str]) -> bool:
    if current is None:
        return True
    if candidate is None:
        return False
    return str(candidate) > str(current)


def _plot_convergence_patterns(
    scenario: str,
    model_map: Dict[str, dict],
    model_display: Dict[str, str],
    output_dir: Path,
) -> None:
    model_keys = sorted(model_map.keys())
    if not model_keys:
        return

    palette = sns.color_palette("tab10", n_colors=max(3, len(model_keys)))
    model_colors = {model: palette[idx % len(palette)] for idx, model in enumerate(model_keys)}

    n_cols = 3
    n_rows = math.ceil(len(METRICS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows))
    axes = np.array(axes).flatten()

    metrics_cache: Dict[str, pd.DataFrame] = {}

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for model_key in model_keys:
            folder = model_map[model_key]["folder"]
            if model_key in metrics_cache:
                df = metrics_cache[model_key]
            else:
                metrics_path = Path("experiments") / folder / "metrics_history.csv"
                if not metrics_path.exists():
                    continue
                df = pd.read_csv(metrics_path)
                metrics_cache[model_key] = df

            if df.empty or metric not in df.columns:
                continue

            grouped = df.groupby("step")[metric]
            mean_values = grouped.mean()
            std_values = grouped.std()
            count_values = grouped.count().replace(0, np.nan)
            ci = 1.96 * std_values / np.sqrt(count_values)

            max_step = min(1000, mean_values.index.max())
            steps = mean_values.index[mean_values.index <= max_step]

            ax.plot(
                steps,
                mean_values[steps],
                label=model_display.get(model_key, model_key),
                linewidth=2.0,
                alpha=0.95,
                color=model_colors[model_key],
            )
            ax.fill_between(
                steps,
                mean_values[steps] - ci[steps],
                mean_values[steps] + ci[steps],
                alpha=0.15,
                color=model_colors[model_key],
            )

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} Over Time", pad=8)
        ax.grid(False)

    for extra_ax in axes[len(METRICS):]:
        extra_ax.axis("off")

    handles, labels = [], []
    for ax in axes:
        h_i, lbls_i = ax.get_legend_handles_labels()
        handles += h_i
        labels += lbls_i

    seen = set()
    unique = [(h_i, lbl) for h_i, lbl in zip(handles, labels) if not (lbl in seen or seen.add(lbl))]
    if unique:
        uhandles, ulabels = zip(*unique)
        fig.legend(
            uhandles,
            ulabels,
            loc="lower center",
            ncol=min(4, len(ulabels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
            labelspacing=0.8,
            borderaxespad=1.0,
            columnspacing=1.6,
            handlelength=2.0,
        )

    scenario_label = SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
    plt.suptitle(
        f"Convergence Patterns Across LLM Models\n{scenario_label}",
        y=0.98,
    )
    sns.despine(fig=fig)
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.94), h_pad=2.0)
    output_path = output_dir / f"{scenario}_convergence_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_comparison(
    scenario: str,
    model_map: Dict[str, dict],
    model_display: Dict[str, str],
    output_dir: Path,
) -> None:
    model_keys = sorted(model_map.keys())
    if not model_keys:
        return

    palette = sns.color_palette("tab10", n_colors=max(3, len(model_keys)))
    model_colors = {model: palette[idx % len(palette)] for idx, model in enumerate(model_keys)}

    final_metrics_by_model: Dict[str, pd.DataFrame] = {}
    for model_key in model_keys:
        folder = model_map[model_key]["folder"]
        metrics_path = Path("experiments") / folder / "metrics_history.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        final_metrics = df.loc[df.groupby("run_id")["step"].idxmax()].copy()
        final_metrics_by_model[model_key] = final_metrics

    if not final_metrics_by_model:
        return

    n_cols = 3
    n_rows = math.ceil(len(METRICS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        plot_data = []
        plot_labels = []
        plot_keys = []

        for model_key in model_keys:
            df = final_metrics_by_model.get(model_key)
            if df is None or metric not in df.columns:
                continue
            vals = df[metric].dropna().values
            if len(vals) == 0:
                continue
            plot_data.append(vals)
            plot_labels.append(model_display.get(model_key, model_key))
            plot_keys.append(model_key)

        if not plot_data:
            ax.axis("off")
            continue

        positions = np.arange(len(plot_labels))
        parts = ax.violinplot(
            plot_data,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            col = model_colors[plot_keys[i]]
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(0.35)
            pc.set_linewidth(1.0)

        bp = ax.boxplot(
            plot_data,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            col = model_colors[plot_keys[i]]
            patch.set_facecolor(col)
            patch.set_edgecolor(col)
            patch.set_alpha(0.65)
            patch.set_linewidth(1.0)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.2)
        for wl in bp["whiskers"]:
            wl.set_color("#777777")
            wl.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("#777777")
            cap.set_linewidth(1.0)

        ax.set_xticks(positions)
        row_index = idx // n_cols
        if row_index < n_rows - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        else:
            ax.set_xticklabels(plot_labels, rotation=25, ha="right")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} by LLM Model", pad=8)
        ax.grid(True, axis="y", alpha=0.25)
        sns.despine(ax=ax)

    for extra_ax in axes[len(METRICS):]:
        extra_ax.axis("off")

    scenario_label = SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
    plt.suptitle(
        f"LLM Model Comparison for {scenario_label}",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.94), h_pad=1.5)
    output_path = output_dir / f"{scenario}_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_segregation_ranking_table(
    scenario_model_map: Dict[str, Dict[str, dict]],
    model_display: Dict[str, str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if SCENARIOS_TO_PLOT:
        allowed_scenarios = set(SCENARIOS_TO_PLOT)
    else:
        allowed_scenarios = set(SCENARIOS.keys()) if SCENARIOS else set()

    models = sorted({
        model_key
        for scenario_map in scenario_model_map.values()
        for model_key in scenario_map.keys()
    })
    if not models:
        print("No LLM model data available for segregation ranking table; skipping.")
        return

    rows: List[dict] = []
    markdown_sections: List[str] = [
        "# Segregation Ranking by Scenario (All LLM Models)",
        "",
        "Higher composite scores indicate more segregation.",
        "",
    ]

    for model_key in models:
        scenario_metrics: Dict[str, pd.DataFrame] = {}
        for scenario, scenario_map in scenario_model_map.items():
            if allowed_scenarios and scenario not in allowed_scenarios:
                continue
            model_payload = scenario_map.get(model_key)
            if not model_payload:
                continue
            metrics_path = Path("experiments") / model_payload["folder"] / "metrics_history.csv"
            if not metrics_path.exists():
                continue
            df = pd.read_csv(metrics_path)
            if df.empty or "run_id" not in df.columns or "step" not in df.columns:
                continue
            final_metrics = df.loc[df.groupby("run_id")["step"].idxmax()].copy()
            scenario_metrics[scenario] = final_metrics

        if not scenario_metrics:
            continue

        metrics = list(METRICS)
        has_dissimilarity = any(
            "dissimilarity_index" in df.columns for df in scenario_metrics.values()
        )
        if has_dissimilarity:
            metrics.append("dissimilarity_index")

        combined = pd.concat(
            [df.reindex(columns=metrics) for df in scenario_metrics.values() if not df.empty],
            ignore_index=True,
        )
        if combined.empty:
            continue

        metric_bounds = {}
        for metric in metrics:
            metric_series = combined[metric].dropna()
            if metric_series.empty:
                continue
            metric_min = metric_series.min()
            metric_max = metric_series.max()
            metric_bounds[metric] = (metric_min, metric_max)

        scenario_scores: Dict[str, np.ndarray] = {}
        for scenario, df in scenario_metrics.items():
            per_run_scores: List[float] = []
            for _, row in df.iterrows():
                normalized_values = []
                for metric, (metric_min, metric_max) in metric_bounds.items():
                    if metric not in row or pd.isna(row[metric]):
                        continue
                    denom = metric_max - metric_min
                    if denom == 0 or np.isnan(denom):
                        normalized = 0.0
                    else:
                        normalized = (row[metric] - metric_min) / denom
                    normalized_values.append(normalized)
                if normalized_values:
                    per_run_scores.append(float(np.mean(normalized_values)))
            if per_run_scores:
                scenario_scores[scenario] = np.array(per_run_scores, dtype=float)

        if not scenario_scores:
            continue

        ranking = sorted(
            scenario_scores.items(),
            key=lambda item: float(np.mean(item[1])),
            reverse=True,
        )

        model_label = model_display.get(model_key, model_key)
        markdown_sections.extend([
            f"## {model_label}",
            "",
            "| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |",
            "|---:|---|---:|---:|---:|:---:|",
        ])

        for idx, (scenario, scores) in enumerate(ranking):
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            n_runs = int(len(scores))

            sig_marker = ""
            p_value = None
            if idx + 1 < len(ranking):
                next_scores = ranking[idx + 1][1]
                if len(scores) >= 2 and len(next_scores) >= 2:
                    try:
                        _, p_value = stats.mannwhitneyu(
                            scores,
                            next_scores,
                            alternative="two-sided",
                        )
                        if p_value < 0.05:
                            sig_marker = "*"
                    except Exception:
                        p_value = None

            scenario_label = SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())
            rows.append({
                "llm_model": model_key,
                "llm_model_display": model_label,
                "scenario": scenario,
                "scenario_label": scenario_label,
                "rank": idx + 1,
                "mean_score": round(mean_score, 6),
                "std_score": round(std_score, 6),
                "n_runs": n_runs,
                "significant_vs_next": sig_marker,
                "p_value_vs_next": None if p_value is None else round(float(p_value), 6),
            })

            markdown_sections.append(
                f"| {idx + 1} | {scenario_label} | {mean_score:.4f} | {std_score:.4f} | {n_runs} | {sig_marker} |"
            )

        markdown_sections.append("")

    if not rows:
        print("No segregation ranking data could be generated; skipping table output.")
        return

    output_csv = output_dir / "segregation_scenario_rankings_all_llm_models.csv"
    output_md = output_dir / "segregation_scenario_rankings_all_llm_models.md"

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_md.write_text("\n".join(markdown_sections), encoding="utf-8")

    print("\nSegregation ranking table saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_md}")


def _write_scenario_coverage_report(
    scenario_model_map: Dict[str, Dict[str, dict]],
    model_display: Dict[str, str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if SCENARIOS_TO_PLOT:
        allowed_scenarios = set(SCENARIOS_TO_PLOT)
    else:
        allowed_scenarios = set(SCENARIOS.keys()) if SCENARIOS else set()

    models = sorted({
        model_key
        for scenario_map in scenario_model_map.values()
        for model_key in scenario_map.keys()
    })
    if not models:
        print("No LLM model data available for scenario coverage report; skipping.")
        return

    rows: List[dict] = []
    markdown_lines: List[str] = [
        "# LLM Scenario Coverage Report",
        "",
        "Scenarios are filtered using SCENARIOS_TO_PLOT when available; otherwise SCENARIOS keys.",
        "",
        "| LLM model | Found | Used | Missing | Extra |",
        "|---|---|---|---|---|",
    ]

    for model_key in models:
        scenarios_found = sorted(
            scenario
            for scenario, scenario_map in scenario_model_map.items()
            if model_key in scenario_map
        )

        if allowed_scenarios:
            scenarios_used = [s for s in scenarios_found if s in allowed_scenarios]
            scenarios_missing = sorted(s for s in allowed_scenarios if s not in scenarios_found)
            scenarios_extra = [s for s in scenarios_found if s not in allowed_scenarios]
        else:
            scenarios_used = list(scenarios_found)
            scenarios_missing = []
            scenarios_extra = []

        model_label = model_display.get(model_key, model_key)
        rows.append({
            "llm_model": model_key,
            "llm_model_display": model_label,
            "scenarios_found": " | ".join(scenarios_found),
            "scenarios_used": " | ".join(scenarios_used),
            "scenarios_missing": " | ".join(scenarios_missing),
            "scenarios_extra": " | ".join(scenarios_extra),
            "n_found": len(scenarios_found),
            "n_used": len(scenarios_used),
            "n_missing": len(scenarios_missing),
            "n_extra": len(scenarios_extra),
        })

        markdown_lines.append(
            " | ".join([
                model_label,
                ", ".join(scenarios_found) if scenarios_found else "-",
                ", ".join(scenarios_used) if scenarios_used else "-",
                ", ".join(scenarios_missing) if scenarios_missing else "-",
                ", ".join(scenarios_extra) if scenarios_extra else "-",
            ])
        )

    output_csv = output_dir / "llm_scenario_coverage.csv"
    output_md = output_dir / "llm_scenario_coverage.md"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_md.write_text("\n".join(markdown_lines), encoding="utf-8")

    print("\nScenario coverage report saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_md}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM model comparison plots and segregation rankings.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing experiment folders (default: experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports_all_llm_models"),
        help="Directory to write output artifacts (default: reports_all_llm_models)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_llm_model_comparisons(
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
