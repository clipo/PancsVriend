# Changelog

All notable changes to this project are documented below.

## [0.2.0] - 2025-06-05

### New Features

- **Per-step Grid State Capture**: Added consistent saving of grid states at each step across all entry points:
  - `baseline_runner.py`
  - `llm_runner.py`
  - `SchellingSim.py` (GUI simulation)
- **Standardized State Format**: Introduced a 3D NumPy array format `(T, H, W)` (dtype=int) for storing integer grid snapshots.
- **Experiment Folder Structure**: Automatic creation of timestamped experiment subfolders under `experiments/`:
  - `states/` (saved `.npz` files)
  - `metrics_history_<timestamp>.csv`
  - `config_<timestamp>.json`
  - `convergence_summary_<timestamp>.csv`
  - `final_metrics_summary_<timestamp>.pdf` (GUI)
- **SchellingSim.py results moved into experiment folder**: 
    - On "Stop & Graph" in `SchellingSim.py`, outputs all states, metrics history, config, convergence summary, and PDF into `experiments/simulation_<timestamp>/`.
    - Moved per-step segregation metrics for the SchellingSim.py(`segregation_metrics_<timestamp>.csv`) into the experiment folder.

### Improvements

- **Helper Function**: Introduced `_grid_to_int()` helper in all runners and GUI to convert object arrays to integer grids.
- **Config Documentation**: Grouped and commented constants in `config.py` for clarity on consumer modules.
- **.gitignore**: Added top-level ignore for Python artifacts, VS Code settings, and the `experiments/` & `reports/` folders.

### Bug Fixes

- **Argument Parsing**: Switched from `argparse.parse_args()` to `parse_known_args()` in `run_experiments.py` to ignore extraneous Jupyter flags.
