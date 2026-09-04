#!/usr/bin/env python3
"""Regenerate the per-metric panel figures for existing run folders.

    python analysis_tools/regenerate_metric_panels.py <run_dir> [<run_dir> ...]
    python analysis_tools/regenerate_metric_panels.py --all

Re-runs ONLY the per_metric_panels step against runs that already have their
simulations and analysis on disk, so a plotting change can be applied without
re-simulating or re-running the whole analysis suite (2026-08-28: panel B's
tilted x labels were centred under their boxes rather than right-aligned to
the tick, which made each label read as belonging to its left-hand neighbour).

Two things make this awkward enough to need a script rather than a one-liner,
and both are why each run is driven in a SUBPROCESS with cwd set to the run
folder:

  * per_metric_panels resolves OUT_DIR at IMPORT time from get_reports_dir(),
    which is cwd-relative — so cwd must be correct before the import, and a
    second run in the same process would keep the first one's OUT_DIR.
  * the scenario list must come from the run's own manifest. Importing the
    module standalone would silently use the stale hardcoded 2025 list in
    experiment_list_for_analysis, produce no data, and write empty panels.

Figures are written to <run>/analysis/metric_panels and then moved to
<run>/plots/metric_panels, matching what the orchestrator does after its
analysis stage.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES = ["panels"]        # set from --figures in main()
RUN_ROOT = REPO_ROOT / "experiments_with_llama_cpp"

# Runs inside the run folder: pick scenarios from that run's manifest exactly as
# run_all_scenario_analysis does, then regenerate the panels.
CHILD = r"""
import sys, glob
from pathlib import Path
import run_all_scenario_analysis as R
man = sorted(glob.glob("manifest/*_run_manifest.json"))
if not man:
    print("NO_MANIFEST"); sys.exit(3)
sel, *_ = R._select_experiments_from_manifest(Path(man[0]), Path("experiments"))
if not sel:
    print("NO_EXPERIMENTS"); sys.exit(4)
R._update_scenarios_for_run(sel)
R._apply_scenarios_to_plot()
import importlib
import os
which = os.environ.get("REGEN_FIGURES", "panels")
if which in ("panels", "all"):
    importlib.import_module("analysis_tools.per_metric_panels").main()
if which in ("convergence", "all"):
    # module does its work at import time
    importlib.import_module("analysis_tools.convergence_patterns_and_speed")
print("PANELS_OK")
"""


def regenerate(run_dir: Path) -> bool:
    # Absolute: the child runs with cwd=run_dir, so a relative run_dir in
    # PANCSVRIEND_REPORTS_DIR would re-resolve against itself and the figures
    # would land in a nested path nobody reads.
    run_dir = run_dir.resolve()
    if not (run_dir / "manifest").is_dir():
        print(f"  {run_dir.name}: no manifest/ — skipped")
        return False
    env_path = f"{REPO_ROOT}:{REPO_ROOT / 'analysis_tools'}"
    # analysis_tools.output_paths.get_reports_dir() defaults to "reports"
    # relative to cwd; the orchestrator overrides it via --output-folder. Set
    # the same override here, or the figures land in a stray <run>/reports/
    # instead of <run>/analysis/ and nothing picks them up.
    proc = subprocess.run([sys.executable, "-c", CHILD], cwd=str(run_dir),
                          capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": env_path,
                               "PANCSVRIEND_REPORTS_DIR": str(run_dir / "analysis"),
                               "REGEN_FIGURES": FIGURES[0]})
    if "PANELS_OK" not in proc.stdout:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        print(f"  {run_dir.name}: FAILED\n      " + "\n      ".join(tail))
        return False
    # Move every regenerated PNG, preserving relative layout — the same thing
    # run_llm_probability_simulation_analysis._move_plots_to_dedicated_folder
    # does after its analysis stage. per_metric_panels writes into
    # analysis/metric_panels/, convergence_patterns_and_speed writes into
    # analysis/ itself, so a metric_panels-only move would strand the latter.
    src_root = run_dir / "analysis"
    dst_root = run_dir / "plots"
    moved = 0
    for png in src_root.rglob("*.png"):
        rel = png.relative_to(src_root)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(png), str(target))
        moved += 1
    print(f"  {run_dir.name}: OK ({moved} figure(s) -> plots/)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--figures", choices=["panels", "convergence", "all"],
                    default="panels",
                    help="which figures to regenerate (default: panels)")
    ap.add_argument("--all", action="store_true",
                    help=f"every run_* folder under {RUN_ROOT.name}/")
    args = ap.parse_args()
    FIGURES[0] = args.figures
    dirs = sorted(RUN_ROOT.glob("run_*")) if args.all else args.run_dirs
    if not dirs:
        ap.error("give one or more run dirs, or --all")
    print(f"regenerating metric panels for {len(dirs)} run(s)")
    ok = sum(regenerate(Path(d)) for d in dirs)
    print(f"done: {ok}/{len(dirs)} succeeded")
    return 0 if ok == len(dirs) else 1


if __name__ == "__main__":
    sys.exit(main())
