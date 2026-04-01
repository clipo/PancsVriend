"""Analyze structural disagreement patterns in OpenAI vs Ollama logprob deltas.

Reads a delta table produced by compare_openai_ollama_logprobs.py and reports:
- bin counts by absolute MOVE delta
- mean composition metrics per bin (walls/similar/opposite/empty)
- top arrangement-code motifs among high-disagreement rows

Outputs:
- *_pattern_summary.json
- *_bin_stats.csv
- *_top_high_disagreement.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Analyze structural patterns in logprob disagreement table")
	parser.add_argument(
		"--input-csv",
		type=str,
		default=None,
		help="Path to openai_vs_ollama_logprob_delta_*.csv (defaults to latest in llm_log_probs/comparisons)",
	)
	parser.add_argument(
		"--comparisons-dir",
		type=str,
		default="llm_log_probs/comparisons",
		help="Directory used when --input-csv is omitted",
	)
	parser.add_argument("--high-threshold", type=float, default=0.05, help="High disagreement threshold on abs delta MOVE")
	parser.add_argument("--top-k", type=int, default=20, help="Rows to include in top high-disagreement table")
	return parser.parse_args()


def _latest_delta_csv(comparisons_dir: Path) -> Path:
	candidates = sorted(comparisons_dir.glob("openai_vs_ollama_logprob_delta_*.csv"))
	candidates = [path for path in candidates if "_errors_" not in path.name]
	if len(candidates) == 0:
		raise FileNotFoundError(f"No delta CSV found in {comparisons_dir}")
	return candidates[-1]


def _motif_features(arrangement_code: str) -> dict[str, int]:
	code = arrangement_code or ""
	return {
		"count_hash": code.count("#"),
		"count_s": code.count("S"),
		"count_o": code.count("O"),
		"count_e": code.count("E"),
		"starts_with_wall": int(code.startswith("#")),
		"ends_with_wall": int(code.endswith("#")),
		"contains_double_wall": int("##" in code),
	}


def _disagreement_bin(value: float) -> str:
	if value < 0.01:
		return "<0.01"
	if value < 0.03:
		return "0.01-0.03"
	if value < 0.05:
		return "0.03-0.05"
	if value < 0.10:
		return "0.05-0.10"
	return ">=0.10"


def main() -> None:
	args = parse_args()
	comparisons_dir = Path(args.comparisons_dir)
	input_csv = Path(args.input_csv) if args.input_csv else _latest_delta_csv(comparisons_dir)

	df = pd.read_csv(input_csv)
	if len(df) == 0:
		raise ValueError(f"Input CSV has no rows: {input_csv}")

	required_cols = {
		"arrangement_code",
		"num_similar",
		"num_opposite",
		"num_empty",
		"num_wall",
		"delta_move_probability",
		"abs_delta_move_probability",
	}
	missing = sorted(required_cols - set(df.columns))
	if missing:
		raise ValueError(f"Input CSV missing required columns: {missing}")

	feature_rows = [_motif_features(str(code)) for code in df["arrangement_code"].tolist()]
	features_df = pd.DataFrame(feature_rows)
	work_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

	work_df["disagreement_bin"] = work_df["abs_delta_move_probability"].map(_disagreement_bin)
	bin_order = ["<0.01", "0.01-0.03", "0.03-0.05", "0.05-0.10", ">=0.10"]

	bin_stats = (
		work_df.groupby("disagreement_bin", as_index=False)
		.agg(
			num_rows=("arrangement_code", "count"),
			mean_abs_delta_move=("abs_delta_move_probability", "mean"),
			mean_delta_move=("delta_move_probability", "mean"),
			mean_num_wall=("num_wall", "mean"),
			mean_num_similar=("num_similar", "mean"),
			mean_num_opposite=("num_opposite", "mean"),
			mean_num_empty=("num_empty", "mean"),
			frac_starts_with_wall=("starts_with_wall", "mean"),
			frac_ends_with_wall=("ends_with_wall", "mean"),
			frac_contains_double_wall=("contains_double_wall", "mean"),
		)
	)
	bin_stats["disagreement_bin"] = pd.Categorical(bin_stats["disagreement_bin"], categories=bin_order, ordered=True)
	bin_stats = bin_stats.sort_values("disagreement_bin")

	high_df = work_df[work_df["abs_delta_move_probability"] >= float(args.high_threshold)].copy()
	high_df = high_df.sort_values("abs_delta_move_probability", ascending=False)

	top_cols = [
		"context_rank",
		"arrangement_code",
		"abs_delta_move_probability",
		"delta_move_probability",
		"openai_move_probability",
		"ollama_move_probability",
		"num_wall",
		"num_similar",
		"num_opposite",
		"num_empty",
	]
	top_high = high_df[top_cols].head(max(1, int(args.top_k)))

	high_counts = (
		high_df.groupby("arrangement_code", as_index=False)
		.agg(
			count=("arrangement_code", "count"),
			mean_abs_delta=("abs_delta_move_probability", "mean"),
		)
		.sort_values(["count", "mean_abs_delta"], ascending=[False, False])
	)

	base_name = input_csv.stem
	output_dir = input_csv.parent
	bin_stats_path = output_dir / f"{base_name}_bin_stats.csv"
	top_path = output_dir / f"{base_name}_top_high_disagreement.csv"
	summary_path = output_dir / f"{base_name}_pattern_summary.json"

	bin_stats.to_csv(bin_stats_path, index=False)
	top_high.to_csv(top_path, index=False)

	summary: dict[str, Any] = {
		"input_csv": str(input_csv),
		"num_rows": int(len(work_df)),
		"mean_abs_delta_move_probability": float(work_df["abs_delta_move_probability"].mean()),
		"max_abs_delta_move_probability": float(work_df["abs_delta_move_probability"].max()),
		"high_threshold": float(args.high_threshold),
		"num_high_disagreement_rows": int(len(high_df)),
		"fraction_high_disagreement_rows": float(len(high_df) / len(work_df)),
		"high_rows_mean_num_wall": float(high_df["num_wall"].mean()) if len(high_df) > 0 else 0.0,
		"all_rows_mean_num_wall": float(work_df["num_wall"].mean()),
		"high_rows_mean_num_empty": float(high_df["num_empty"].mean()) if len(high_df) > 0 else 0.0,
		"all_rows_mean_num_empty": float(work_df["num_empty"].mean()),
		"top_high_disagreement_examples": top_high.to_dict(orient="records"),
		"most_common_high_disagreement_arrangements": high_counts.head(10).to_dict(orient="records"),
		"outputs": {
			"bin_stats_csv": str(bin_stats_path),
			"top_high_csv": str(top_path),
			"summary_json": str(summary_path),
		},
	}

	with open(summary_path, "w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2)

	print("\n=== Disagreement pattern analysis complete ===")
	print(f"Input:      {input_csv}")
	print(f"Bin stats:  {bin_stats_path}")
	print(f"Top rows:   {top_path}")
	print(f"Summary:    {summary_path}")
	print(f"Rows >= {args.high_threshold:.3f} abs delta MOVE: {len(high_df)} / {len(work_df)}")

	print("\nBin overview:")
	print(bin_stats[["disagreement_bin", "num_rows", "mean_abs_delta_move", "mean_num_wall", "mean_num_empty"]].to_string(index=False))


if __name__ == "__main__":
	main()