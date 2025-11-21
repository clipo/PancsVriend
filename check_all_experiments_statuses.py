#!/usr/bin/env python3
"""
Standalone experiment status checker
"""

import os
import json

from llm_runner import _analyze_run_status, check_existing_experiment


def extract_llm_identifier(config):
    """Return a readable identifier for the LLM used in the experiment."""
    direct_keys = ("llm_model", "llm_preset", "llm_name")
    for key in direct_keys:
        value = config.get(key)
        if value:
            return str(value)

    llm_config = config.get("llm")
    if isinstance(llm_config, dict):
        for candidate in ("model", "name", "preset"):
            value = llm_config.get(candidate)
            if value:
                return str(value)

    return "-"

def check_experiment_status(experiment_name):
    """Check status of a specific experiment using run-level inspection."""
    output_dir = f"experiments/{experiment_name}"

    if not os.path.exists(output_dir):
        return False, {
            "completed": 0,
            "reached_max": 0,
            "aborted": 0,
            "missing": 0,
            "total_runs": 0,
        }, "-"

    exists, _, resolved_output_dir, existing_run_ids = check_existing_experiment(experiment_name)
    if not exists:
        return False, {
            "completed": 0,
            "reached_max": 0,
            "aborted": 0,
            "missing": 0,
            "total_runs": 0,
        }, "-"

    output_dir = resolved_output_dir

    # Load config details when available
    config_file = os.path.join(output_dir, "config.json")
    total_runs = "unknown"
    max_steps = 1000
    llm_model = "-"

    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as fh:
                config = json.load(fh)
            total_runs = config.get("n_runs", "unknown")
            llm_model = extract_llm_identifier(config)
            max_steps = config.get("max_steps", max_steps)
            try:
                max_steps = int(max_steps)
            except (TypeError, ValueError):
                max_steps = 1000
        except Exception as exc:
            total_runs = f"error: {exc}"
            llm_model = "error"

    # Determine which run IDs to inspect
    if isinstance(total_runs, int) and total_runs >= 0:
        run_ids = list(range(total_runs))
    else:
        run_ids = sorted(existing_run_ids)

    status_counts = {
        "converged": 0,
        "reached_max": 0,
        "aborted": 0,
        "missing": 0,
        "other": 0,
    }

    for run_id in run_ids:
        run_status = _analyze_run_status(output_dir, run_id, max_steps)
        state = run_status.get("status", "other")
        if state not in status_counts:
            state = "other"
        status_counts[state] += 1

    completed_total = status_counts["converged"] + status_counts["reached_max"]

    aggregated = {
        "completed": completed_total,
        "reached_max": status_counts["reached_max"],
        "converged": status_counts["converged"],
        "aborted": status_counts["aborted"],
        "missing": status_counts["missing"],
        "other": status_counts["other"],
        "total_runs": total_runs,
        "inspected_runs": len(run_ids),
    }

    return True, aggregated, llm_model

def list_all_experiments():
    """List all experiments with their status"""
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        print("No experiments directory found.")
        return
    
    print("Experiment Status Report")
    print("=" * 150)
    print(f"{'Experiment Name':<50} {'Status':<20} {'LLM Model':<30} {'Details':<30}")
    print("-" * 150)
    
    for exp_name in sorted(os.listdir(exp_dir)):
        exp_path = os.path.join(exp_dir, exp_name)
        if os.path.isdir(exp_path):
            exists, summary, llm_model = check_experiment_status(exp_name)

            if exists:
                total = summary["total_runs"]
                inspected = summary["inspected_runs"]
                completed = summary["completed"]
                aborted = summary["aborted"]
                missing = summary["missing"]

                details_col = f"runs inspected: {inspected}"

                if isinstance(total, int):
                    status = f"{completed}/{total} complete"
                else:
                    status = f"{completed}/{inspected} complete"
                    details_col += f", total: {total}"

                extras = []
                if aborted:
                    extras.append(f"{aborted} aborted")
                if missing:
                    extras.append(f"{missing} missing")
                if summary.get("other"):
                    extras.append(f"{summary['other']} other")
                if extras:
                    status = f"{status}; {'; '.join(extras)}"

                print(f"{exp_name:<50} {status:<20} {llm_model:<30} {details_col:<30}")
            else:
                print(f"{exp_name:<50} {'ERROR':<20} {'-':<30} {'Directory missing':<30}")

if __name__ == "__main__":
    list_all_experiments()
