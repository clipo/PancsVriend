# CCS 2026 Presentation Assets

Scripts for generating tables and plots for the CCS 2026 presentation.

## Quick Start

Regenerate all tables and plots:

```bash
python regenerate_all_tables_and_plots.py --yes
```

A wrapper script is also available in the presentation folder:

```bash
cd pres-overleaf.link/pres
python regenerate-all-tables-and-plots.py --yes
```

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `regenerate_all_tables_and_plots.py` | — | Master script that runs all others |
| `generate_value_function_plots.py` | `*_value_functions.png` | Choice function 6-panel grids |
| `generate_convergence_plots.py` | `*_convergence_clean.png` | Convergence trajectory plots |
| `generate_sorted_segregation_plots.py` | `*_segregation_sorted.png` | Segregation bar charts |
| `generate_and_deploy_tables.py` | `key_results_ordering.tex` | LaTeX tables for results |
| `deploy_heatmaps.py` | `*_heatmap_*.png` | P(MOVE) heatmaps per scenario |
| `deploy_model_images.py` | — | Copies model images to `pics/` |

## Usage Flags

All scripts support:

- `--yes` — Auto-deploy to `pres-overleaf.link/pres/pics/` without prompting
- `--no` — Generate only, skip deployment

Without flags, scripts will prompt for deployment confirmation (on target machine only).

## Requirements

- Access to `experiments_with_llama_cpp/` experiment data
- Python with pandas, numpy, matplotlib

## Target Machine

Scripts check for hostname `ECON-0FM96LD-L` before prompting for deployment.
Use `--yes` to override this check and deploy from any machine.
