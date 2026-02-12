"""Utility approximation study for LLM movement decisions.

This script generates synthetic 3x3 neighborhood contexts for each configured
scenario, queries the LLM for MOVE/STAY decisions across a uniform range of
"like" neighbor counts, and aggregates the responses to approximate the
resulting decision distribution. For each neighbor-count value, all unique
permutations of similar/opposite neighbor placements are enumerated and the LLM
is queried `repeats` times per arrangement. The aggregated results are saved as
CSV files and visualized in a multi-panel plot where each panel shows the MOVE
rate as a function of the ratio of similar neighbors.

Example usage:

    python llm_utility_approximation.py --repeats 10 --output-dir outputs/llm
    python llm_utility_approximation.py --scenarios baseline gender_man_woman \
        --repeats 5 --seed 123
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests import Response
from tqdm import tqdm

import config as cfg
from context_scenarios import CONTEXT_SCENARIOS


TOTAL_NEIGHBORS = 8  # Number of positions surrounding the focal agent in a 3x3 grid


@dataclass
class TrialResult:
	"""Container for a single LLM trial outcome."""

	scenario: str
	agent_role: str
	agent_label: str
	opposite_label: str
	trial_index: int
	num_similar: int
	ratio_similar: float
	raw_response: str
	decision: str


def generate_neighbor_context(neighbors: List[str]) -> str:
	"""Create a 3x3 neighborhood context string from a neighbor sequence."""

	if len(neighbors) != TOTAL_NEIGHBORS:
		raise ValueError(f"Expected {TOTAL_NEIGHBORS} neighbors, got {len(neighbors)}")

	grid_rows: List[List[str]] = []
	idx = 0
	for r in range(3):
		row = []
		for c in range(3):
			if r == 1 and c == 1:
				row.append("X")
			else:
				row.append(neighbors[idx])
				idx += 1
		grid_rows.append(row)

	return "\n".join(" ".join(row) for row in grid_rows)


def generate_neighbor_arrangements(num_similar: int) -> List[List[str]]:
	"""Enumerate all unique neighbor arrangements for a given similarity count."""

	if not 0 <= num_similar <= TOTAL_NEIGHBORS:
		raise ValueError("num_similar must be between 0 and 8 inclusive")

	arrangements: List[List[str]] = []
	for similar_indices in combinations(range(TOTAL_NEIGHBORS), num_similar):
		neighbors = ["O"] * TOTAL_NEIGHBORS
		for idx in similar_indices:
			neighbors[idx] = "S"
		arrangements.append(neighbors)

	return arrangements


def build_prompt(scenario_key: str, context: str, agent_label: str, opposite_label: str) -> str:
	"""Format the prompt for the LLM using the scenario template."""

	scenario = CONTEXT_SCENARIOS[scenario_key]
	return scenario["prompt_template"].format(
		agent_type=agent_label,
		opposite_type=opposite_label,
		context=context,
	)


def call_llm(
	prompt: str,
	session: requests.Session,
	max_retries: int = 5,
	retry_backoff: float = 1.5,
	request_timeout: int = 20,
) -> str:
	"""Send a prompt to the configured LLM and return the raw response content."""

	payload = {
		"model": cfg.OLLAMA_MODEL,
		"messages": [{"role": "user", "content": prompt}],
		"stream": False,
		"temperature": 0.3,
		"max_tokens": 16,
	}

	headers = {
		"Authorization": f"Bearer {cfg.OLLAMA_API_KEY}",
		"Content-Type": "application/json",
	}

	last_exception: Exception | None = None

	for attempt in range(1, max_retries + 1):
		try:
			response: Response = session.post(
				cfg.OLLAMA_URL,
				headers=headers,
				json=payload,
				timeout=request_timeout,
			)
			response.raise_for_status()
			data = response.json()
			content = data["choices"][0]["message"]["content"].strip()
			return content
		except (requests.Timeout, requests.ConnectionError) as exc:
			last_exception = exc
			if attempt == max_retries:
				break
			sleep_time = retry_backoff ** attempt
			time.sleep(sleep_time)
		except Exception as exc:  # Catch non-response-format errors
			last_exception = exc
			if attempt == max_retries:
				break
			time.sleep(retry_backoff)

	raise RuntimeError(f"LLM request failed after {max_retries} attempts") from last_exception


def parse_decision(raw_response: str) -> str:
	"""Normalize the LLM response to MOVE, STAY, or UNKNOWN."""

	text = raw_response.strip().upper()

	# Prioritize exact matches to avoid misclassification when both words appear
	tokens = text.replace(".", " ").replace(",", " ").split()
	if tokens:
		first_token = tokens[0]
		if first_token == "MOVE":
			return "MOVE"
		if first_token == "STAY":
			return "STAY"

	if "MOVE" in text and "STAY" not in text:
		return "MOVE"
	if "STAY" in text and "MOVE" not in text:
		return "STAY"
	if "MOVE" in text and text.index("MOVE") < text.index("STAY"):
		return "MOVE"
	if "STAY" in text and text.index("STAY") < text.index("MOVE"):
		return "STAY"
	return "UNKNOWN"


def run_trials(
	scenarios: Iterable[str],
	repeats: int,
	seed: int | None,
	session: requests.Session,
) -> List[TrialResult]:
	"""Execute LLM queries across scenarios and return collected outcomes."""

	rng = random.Random(seed)
	results: List[TrialResult] = []

	for scenario_key in scenarios:
		scenario_info = CONTEXT_SCENARIOS[scenario_key]
		agent_configs = [
			("type_a", scenario_info["type_a"], scenario_info["type_b"]),
			("type_b", scenario_info["type_b"], scenario_info["type_a"]),
		]

		arrangements_by_count = {
			num_similar: generate_neighbor_arrangements(num_similar)
			for num_similar in range(TOTAL_NEIGHBORS + 1)
		}
		for arrangements in arrangements_by_count.values():
			rng.shuffle(arrangements)

		total_per_agent = sum(len(arrangements) * repeats for arrangements in arrangements_by_count.values())
		total_iterations = len(agent_configs) * total_per_agent
		progress = tqdm(total=total_iterations, desc=f"Scenario: {scenario_key}", leave=False)
		scenario_trial_index = 0
		for agent_role, agent_label, opposite_label in agent_configs:
			for num_similar in range(TOTAL_NEIGHBORS + 1):
				ratio = num_similar / TOTAL_NEIGHBORS
				arrangements = arrangements_by_count[num_similar]
				for neighbors in arrangements:
					context = generate_neighbor_context(neighbors)
					for _ in range(repeats):
						prompt = build_prompt(scenario_key, context, agent_label, opposite_label)
						raw_response = call_llm(prompt, session)
						decision = parse_decision(raw_response)

						results.append(
							TrialResult(
								scenario=scenario_key,
								agent_role=agent_role,
								agent_label=agent_label,
								opposite_label=opposite_label,
								trial_index=scenario_trial_index,
								num_similar=num_similar,
								ratio_similar=ratio,
								raw_response=raw_response,
								decision=decision,
							)
						)
						progress.update(1)
						scenario_trial_index += 1
		progress.close()

	return results


def aggregate_results(results: List[TrialResult]) -> pd.DataFrame:
	"""Convert trial results into a tidy DataFrame with counts and rates."""

	if not results:
		return pd.DataFrame(
			columns=[
				"scenario",
				"agent_role",
				"agent_label",
				"num_similar",
				"ratio_similar",
				"decision",
				"count",
				"total",
				"move_rate",
			]
		)

	df = pd.DataFrame([r.__dict__ for r in results])
	counts = (
		df.groupby(
			["scenario", "agent_role", "agent_label", "num_similar", "ratio_similar", "decision"]
		)
		.size()
		.reset_index(name="count")
	)

	totals = counts.groupby(
		["scenario", "agent_role", "agent_label", "num_similar", "ratio_similar"],
		as_index=False,
	).agg(total=("count", "sum"))

	merged = counts.merge(
		totals,
		on=["scenario", "agent_role", "agent_label", "num_similar", "ratio_similar"],
	)

	move_counts = merged[merged["decision"] == "MOVE"][
		["scenario", "agent_role", "agent_label", "num_similar", "ratio_similar", "count"]
	]
	move_counts = move_counts.rename(columns={"count": "move_count"})

	summary = totals.merge(
		move_counts,
		on=["scenario", "agent_role", "agent_label", "num_similar", "ratio_similar"],
		how="left",
	)
	summary["move_count"] = summary["move_count"].fillna(0)
	summary["move_rate"] = summary["move_count"] / summary["total"].replace(0, np.nan)
	summary["move_rate"] = summary["move_rate"].fillna(0)

	return summary.sort_values(["scenario", "agent_role", "num_similar"])


def plot_move_rates(summary: pd.DataFrame, output_path: str) -> None:
	"""Create subplot grid showing MOVE rates versus similar-neighbor ratios."""

	scenarios = summary["scenario"].unique()
	num_scenarios = len(scenarios)
	if num_scenarios == 0:
		return

	cols = 3
	rows = math.ceil(num_scenarios / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)

	for idx, scenario in enumerate(scenarios):
		ax = axes[idx // cols][idx % cols]
		scenario_df = summary[summary["scenario"] == scenario]
		for agent_label, agent_df in scenario_df.groupby("agent_label"):
			agent_df = agent_df.sort_values("num_similar")
			ax.plot(
				agent_df["ratio_similar"],
				agent_df["move_rate"],
				marker="o",
				linestyle="-",
				label=agent_label,
			)
		ax.set_ylim(-0.05, 1.05)
		ax.set_xticks([i / TOTAL_NEIGHBORS for i in range(TOTAL_NEIGHBORS + 1)])
		ax.set_xlabel("Ratio of similar neighbors")
		ax.set_ylabel("P(MOVE)")
		ax.set_title(scenario)
		ax.grid(True, alpha=0.3)
		ax.legend()

	# Hide any unused subplots
	for idx in range(num_scenarios, rows * cols):
		fig.delaxes(axes[idx // cols][idx % cols])

	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def ensure_directory(path: str) -> None:
	"""Create the directory if it doesn't exist."""

	os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approximate LLM move utility curves")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(CONTEXT_SCENARIOS.keys()),
        help="Subset of scenario keys to evaluate",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of repetitions per neighbor-count configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/llm_utility",
        help="Directory for CSV and plot outputs",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="llm_move_distribution.csv",
        help="Filename for the aggregated CSV",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="llm_move_distribution.png",
        help="Filename for the MOVE rate plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing = set(args.scenarios) - set(CONTEXT_SCENARIOS.keys())
    if missing:
        available = ", ".join(sorted(CONTEXT_SCENARIOS.keys()))
        raise ValueError(f"Unknown scenario keys: {missing}. Available: {available}")

    ensure_directory(args.output_dir)

    session = requests.Session()
    try:
        results = run_trials(args.scenarios, args.repeats, args.seed, session)
    finally:
        session.close()

    # Persist raw responses for deeper analysis
    raw_df = pd.DataFrame([r.__dict__ for r in results])
    raw_csv_path = os.path.join(args.output_dir, "llm_raw_trials.csv")
    raw_df.to_csv(raw_csv_path, index=False)

    summary = aggregate_results(results)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    summary.to_csv(csv_path, index=False)

    plot_path = os.path.join(args.output_dir, args.plot_name)
    plot_move_rates(summary, plot_path)

    print(f"Saved raw trials to: {raw_csv_path}")
    print(f"Saved summary CSV to: {csv_path}")
    print(f"Saved MOVE rate plot to: {plot_path}")


if __name__ == "__main__":
    main()

