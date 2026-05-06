"""Compare branching-derived MOVE/STAY probabilities to empirical sampling.

Pairs with ``branching_probability_estimator.py``. For a debug-sized slice of
(scenario, agent_role, arrangement) tuples, runs both:

  1. The branching estimator (single pass, deterministic).
  2. Empirical sampling (N stochastic samples at the same temperature),
     reusing the same ``parse_decision`` parser.

Emits:
  * ``comparison_<model>_<temp>_<timestamp>.csv`` — per-context rows with
    branching probability, sample probability, sample 95% Wilson CI, deltas,
    in-CI flags, and wall-time per method.
  * ``comparison_<model>_<temp>_<timestamp>_summary.json`` — aggregate
    calibration statistics.
  * ``comparison_<model>_<temp>_<timestamp>_scatter.png`` — branch vs sample
    scatter with sample-CI error bars and a y=x reference line.
  * ``comparison_<model>_<temp>_<timestamp>_histogram.png`` — delta_MOVE
    distribution.

Example (the default slice is small enough for a single command)::

    python branching_vs_sampling_comparison.py \\
        --model-path C:/Users/Sriki/PancsVriend/llms/gemma-3-4b-it-q4_0.gguf \\
        --model-name gemma-3-4b-it-q4_0 \\
        --temperature 0.3 \\
        --n-samples 500 \\
        --debug-subset 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for candidate in (_THIS_DIR, _REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from context_scenarios import CONTEXT_SCENARIOS  # noqa: E402
from llm_token_probabilities import _sanitize_model_for_path_component  # noqa: E402
from significance_utils import binomial_two_sided_p  # noqa: E402

from branching_probability_estimator import (  # noqa: E402
    BranchingConfig,
    BranchingEngine,
    GenerationConfig,
    enumerate_arrangement_tasks,
    temp_slug,
)

try:
    from llama_cpp import Llama  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"llama-cpp-python is required: {exc}")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:
    plt = None  # type: ignore


# ----------------------------------------------------------------------------
# Empirical sampling (single-process, reusing branching engine's Llama)
# ----------------------------------------------------------------------------


def sample_prompt(
    llm: Llama,
    prompt: str,
    n_samples: int,
    temperature: float,
    gen_config: GenerationConfig,
    parse_decision,
) -> dict[str, Any]:
    move = stay = unknown = 0
    decisions: list[str] = []
    t0 = time.time()
    for _ in range(int(n_samples)):
        response = llm(
            prompt,
            max_tokens=gen_config.max_tokens,
            temperature=temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            min_p=gen_config.min_p,
            repeat_penalty=gen_config.repeat_penalty,
        )
        text = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    text = str(first.get("text", "")) or str(
                        (first.get("message") or {}).get("content", "")
                    )
        decision = parse_decision(text)
        decisions.append(decision)
        if decision == "MOVE":
            move += 1
        elif decision == "STAY":
            stay += 1
        else:
            unknown += 1
    elapsed = time.time() - t0
    total = float(max(1, n_samples))
    return {
        "move_count": move,
        "stay_count": stay,
        "unknown_count": unknown,
        "n_samples": int(n_samples),
        "move_probability": move / total,
        "stay_probability": stay / total,
        "unknown_probability": unknown / total,
        "seconds_elapsed": elapsed,
        "raw_decisions": decisions,
    }


def wilson_ci(count: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    phat = count / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------
# Comparison logic
# ----------------------------------------------------------------------------


COMPARISON_COLUMNS: tuple[str, ...] = (
    "scenario", "agent_role", "arrangement_code", "context_snippet",
    "n_samples",
    "p_branch_MOVE", "p_sample_MOVE", "delta_MOVE", "abs_delta_MOVE",
    "p_sample_MOVE_ci_low", "p_sample_MOVE_ci_high", "inside_95_ci_MOVE",
    "p_branch_STAY", "p_sample_STAY", "delta_STAY", "abs_delta_STAY",
    "p_sample_STAY_ci_low", "p_sample_STAY_ci_high", "inside_95_ci_STAY",
    "p_branch_UNKNOWN", "p_sample_UNKNOWN", "delta_UNKNOWN", "abs_delta_UNKNOWN",
    "num_final_active_states", "branch_quality_flags",
    "seconds_branch", "seconds_sample",
    "sample_move_count", "sample_stay_count", "sample_unknown_count",
    "branch_vs_sample_binom_p_move", "branch_vs_sample_binom_p_stay",
)


def compare_one(
    engine: BranchingEngine,
    task,
    temperature: float,
    max_tokens: int,
    n_samples: int,
    gen_config: GenerationConfig,
) -> dict[str, Any]:
    t0 = time.time()
    branch = engine.estimate(
        prompt=task.prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        capture_trace=False,
    )
    seconds_branch = time.time() - t0

    sample = sample_prompt(
        llm=engine.llm,
        prompt=task.prompt,
        n_samples=n_samples,
        temperature=temperature,
        gen_config=gen_config,
        parse_decision=engine.parse_decision,
    )

    ci_move = wilson_ci(sample["move_count"], sample["n_samples"])
    ci_stay = wilson_ci(sample["stay_count"], sample["n_samples"])

    def _delta(a: float, b: float) -> float:
        return float(a) - float(b)

    context_snippet = " / ".join(task.context_string.splitlines())

    row: dict[str, Any] = {
        "scenario": task.scenario,
        "agent_role": task.agent_role,
        "arrangement_code": task.arrangement_code,
        "context_snippet": context_snippet,
        "n_samples": int(sample["n_samples"]),

        "p_branch_MOVE": float(branch["move_probability"]),
        "p_sample_MOVE": float(sample["move_probability"]),
        "delta_MOVE": _delta(branch["move_probability"], sample["move_probability"]),
        "abs_delta_MOVE": abs(_delta(branch["move_probability"], sample["move_probability"])),
        "p_sample_MOVE_ci_low": float(ci_move[0]),
        "p_sample_MOVE_ci_high": float(ci_move[1]),
        "inside_95_ci_MOVE": bool(ci_move[0] <= branch["move_probability"] <= ci_move[1]),

        "p_branch_STAY": float(branch["stay_probability"]),
        "p_sample_STAY": float(sample["stay_probability"]),
        "delta_STAY": _delta(branch["stay_probability"], sample["stay_probability"]),
        "abs_delta_STAY": abs(_delta(branch["stay_probability"], sample["stay_probability"])),
        "p_sample_STAY_ci_low": float(ci_stay[0]),
        "p_sample_STAY_ci_high": float(ci_stay[1]),
        "inside_95_ci_STAY": bool(ci_stay[0] <= branch["stay_probability"] <= ci_stay[1]),

        "p_branch_UNKNOWN": float(branch["unknown_probability"]),
        "p_sample_UNKNOWN": float(sample["unknown_probability"]),
        "delta_UNKNOWN": _delta(branch["unknown_probability"], sample["unknown_probability"]),
        "abs_delta_UNKNOWN": abs(_delta(branch["unknown_probability"], sample["unknown_probability"])),

        "num_final_active_states": int(branch["num_final_active_states"]),
        "branch_quality_flags": str(branch["quality_flags"]),
        "seconds_branch": float(seconds_branch),
        "seconds_sample": float(sample["seconds_elapsed"]),

        "sample_move_count": int(sample["move_count"]),
        "sample_stay_count": int(sample["stay_count"]),
        "sample_unknown_count": int(sample["unknown_count"]),
        # Branching estimate plays the deterministic-reference role here — same
        # test shape as replay_vs_sample in all_neighborhoods_summary.csv.
        "branch_vs_sample_binom_p_move": binomial_two_sided_p(
            int(sample["move_count"]), int(sample["n_samples"]), float(branch["move_probability"])
        ),
        "branch_vs_sample_binom_p_stay": binomial_two_sided_p(
            int(sample["stay_count"]), int(sample["n_samples"]), float(branch["stay_probability"])
        ),
    }
    return row


def compute_summary(rows: list[dict[str, Any]], model_name: str, temperature: float,
                    n_samples: int) -> dict[str, Any]:
    if not rows:
        return {"model": model_name, "temperature": temperature, "n_samples": n_samples, "n_contexts": 0}

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r[name]) for r in rows], dtype=float)

    branch_move = col("p_branch_MOVE")
    sample_move = col("p_sample_MOVE")
    pearson = float(np.corrcoef(branch_move, sample_move)[0, 1]) if len(rows) >= 2 else float("nan")

    return {
        "model": model_name,
        "temperature": temperature,
        "n_samples": n_samples,
        "n_contexts": len(rows),
        "mean_abs_delta_MOVE": float(np.mean(col("abs_delta_MOVE"))),
        "median_abs_delta_MOVE": float(np.median(col("abs_delta_MOVE"))),
        "max_abs_delta_MOVE": float(np.max(col("abs_delta_MOVE"))),
        "mean_abs_delta_STAY": float(np.mean(col("abs_delta_STAY"))),
        "median_abs_delta_STAY": float(np.median(col("abs_delta_STAY"))),
        "max_abs_delta_STAY": float(np.max(col("abs_delta_STAY"))),
        "mean_abs_delta_UNKNOWN": float(np.mean(col("abs_delta_UNKNOWN"))),
        "pct_inside_95_ci_MOVE": float(np.mean([1.0 if r["inside_95_ci_MOVE"] else 0.0 for r in rows])),
        "pct_inside_95_ci_STAY": float(np.mean([1.0 if r["inside_95_ci_STAY"] else 0.0 for r in rows])),
        "pearson_r_branch_vs_sample_MOVE": pearson,
        "total_seconds_branch": float(np.sum(col("seconds_branch"))),
        "total_seconds_sample": float(np.sum(col("seconds_sample"))),
        "mean_seconds_branch_per_context": float(np.mean(col("seconds_branch"))),
        "mean_seconds_sample_per_context": float(np.mean(col("seconds_sample"))),
    }


def plot_scatter(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    if plt is None:
        return
    branch = [r["p_branch_MOVE"] for r in rows]
    sample = [r["p_sample_MOVE"] for r in rows]
    ci_low = [r["p_sample_MOVE_ci_low"] for r in rows]
    ci_high = [r["p_sample_MOVE_ci_high"] for r in rows]
    err = [
        [s - lo for s, lo in zip(sample, ci_low)],
        [hi - s for s, hi in zip(sample, ci_high)],
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(sample, branch, xerr=err, fmt="o", capsize=3, alpha=0.7,
                label="context")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y = x (perfect)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("P(MOVE) — empirical sampling")
    ax.set_ylabel("P(MOVE) — branching estimator")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_histogram(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    if plt is None:
        return
    deltas = [r["delta_MOVE"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deltas, bins=20, edgecolor="black", alpha=0.75)
    ax.axvline(0.0, color="red", linestyle="--", alpha=0.7, label="zero delta")
    ax.set_xlabel("delta = p_branch_MOVE - p_sample_MOVE")
    ax.set_ylabel("contexts")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Branching vs empirical sampling comparison.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--n-ctx", type=int, default=512)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--n-gpu-layers", type=int, default=99)

    p.add_argument("--scenarios", nargs="+", default=["baseline"])
    p.add_argument("--agent-roles", nargs="+", default=["type_a"], choices=["type_a", "type_b"])

    p.add_argument("--n-samples", type=int, default=1000,
                   help="Number of empirical samples per context (default: 1000)")
    p.add_argument("--debug-subset", type=int, default=50,
                   help="Only compare N arrangement tasks; sampled randomly with --sample-seed")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="Seed used when randomly sampling the debug subset (default: 0)")

    p.add_argument("--beam-width", type=int, default=16)
    p.add_argument("--candidate-top-n", type=int, default=1)
    p.add_argument("--candidate-top-n-cap", type=int, default=2048)
    p.add_argument("--min-step-retained-mass", type=float, default=0.999)
    p.add_argument("--early-stop-move-stay-mass", type=float, default=0.999)

    p.add_argument("--output-root", default=str(_REPO_ROOT / "llm_log_probs"),
                   help="Root directory for outputs (mirrors branching estimator layout)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if len(args.scenarios) == 1 and args.scenarios[0].lower() == "all":
        scenarios = sorted(CONTEXT_SCENARIOS.keys())
    else:
        scenarios = list(args.scenarios)
        for scen in scenarios:
            if scen not in CONTEXT_SCENARIOS:
                raise SystemExit(
                    f"Unknown scenario '{scen}'. Available: "
                    f"{sorted(CONTEXT_SCENARIOS.keys())} or 'all'"
                )

    gen_config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
    )
    branch_config = BranchingConfig(
        beam_width=args.beam_width,
        candidate_top_n=args.candidate_top_n,
        candidate_top_n_cap=args.candidate_top_n_cap,
        min_step_retained_mass=args.min_step_retained_mass,
        early_stop_move_stay_mass=args.early_stop_move_stay_mass,
    )
    engine = BranchingEngine(
        model_path=args.model_path,
        gen_config=gen_config,
        branch_config=branch_config,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    output_root = Path(args.output_root).resolve()
    model_slug = _sanitize_model_for_path_component(args.model_name)
    ts = temp_slug(args.temperature)
    out_dir = output_root / model_slug / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"comparison_{model_slug}_{ts}_{timestamp}.csv"
    summary_path = out_dir / f"comparison_{model_slug}_{ts}_{timestamp}_summary.json"
    scatter_path = out_dir / f"comparison_{model_slug}_{ts}_{timestamp}_scatter.png"
    hist_path = out_dir / f"comparison_{model_slug}_{ts}_{timestamp}_histogram.png"

    print(f"[comparison] model={args.model_name} temp={args.temperature} "
          f"scenarios={scenarios} roles={args.agent_roles} debug_subset={args.debug_subset} "
          f"n_samples={args.n_samples}", flush=True)
    print(f"[comparison] writing to {csv_path}", flush=True)

    all_tasks = []
    for scenario in scenarios:
        tasks = enumerate_arrangement_tasks(scenario, list(args.agent_roles))
        if args.debug_subset is not None and args.debug_subset > 0:
            sample_count = min(int(args.debug_subset), len(tasks))
            tasks = random.Random(int(args.sample_seed)).sample(tasks, sample_count)
        all_tasks.extend(tasks)

    print(f"[comparison] {len(all_tasks)} contexts to compare", flush=True)

    rows: list[dict[str, Any]] = []
    header_written = False
    for idx, task in enumerate(all_tasks, start=1):
        row = compare_one(
            engine=engine,
            task=task,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n_samples=args.n_samples,
            gen_config=gen_config,
        )
        rows.append(row)

        mode = "a" if header_written else "w"
        with csv_path.open(mode, encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COMPARISON_COLUMNS), extrasaction="ignore")
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)

        inside_move = "YES" if row["inside_95_ci_MOVE"] else "NO"
        print(f"[{idx}/{len(all_tasks)}] {task.scenario}/{task.agent_role}/{task.arrangement_code} "
              f"p_branch_MOVE={row['p_branch_MOVE']:.3f} p_sample_MOVE={row['p_sample_MOVE']:.3f} "
              f"(CI [{row['p_sample_MOVE_ci_low']:.3f}, {row['p_sample_MOVE_ci_high']:.3f}]) "
              f"inside_CI={inside_move} delta={row['delta_MOVE']:+.3f} "
              f"t_branch={row['seconds_branch']:.2f}s t_sample={row['seconds_sample']:.2f}s",
              flush=True)

    summary = compute_summary(rows, args.model_name, args.temperature, args.n_samples)
    summary["csv_path"] = str(csv_path)
    summary["scatter_path"] = str(scatter_path) if plt is not None else None
    summary["histogram_path"] = str(hist_path) if plt is not None else None
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    plot_scatter(rows, scatter_path, title=f"{args.model_name} @ T={args.temperature}")
    plot_delta_histogram(rows, hist_path, title=f"{args.model_name} @ T={args.temperature} — delta_MOVE")

    print("\n[comparison] summary:")
    for key in (
        "n_contexts", "mean_abs_delta_MOVE", "median_abs_delta_MOVE", "max_abs_delta_MOVE",
        "pct_inside_95_ci_MOVE", "pct_inside_95_ci_STAY", "pearson_r_branch_vs_sample_MOVE",
        "mean_seconds_branch_per_context", "mean_seconds_sample_per_context",
    ):
        if key in summary:
            val = summary[key]
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
    print(f"\n[comparison] wrote:\n  csv={csv_path}\n  summary={summary_path}\n"
          f"  scatter={scatter_path}\n  histogram={hist_path}")


if __name__ == "__main__":
    main()
