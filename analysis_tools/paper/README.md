# Presentation & Paper Assets

Scripts for generating tables and plots for both the CCS 2026 presentation and the paper.

## Quick Start

Regenerate all tables and plots for the **presentation** (default):

```bash
python regenerate_all_tables_and_plots.py --yes
```

Regenerate all tables and plots for the **paper**:

```bash
python regenerate_all_tables_and_plots.py --target paper --yes
```

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `regenerate_all_tables_and_plots.py` | — | Master script that runs all others |
| `generate_choice_function_plots.py` | `*_choice_functions.png` | Choice function 6-panel grids |
| `generate_convergence_plots.py` | `*_convergence_clean.png` | Convergence trajectory plots |
| `generate_sorted_segregation_plots.py` | `*_segregation_sorted.png` | Segregation bar charts |
| `generate_and_deploy_tables.py` | `key_results_ordering.tex` | LaTeX tables for results |
| `generate_and_deploy_heatmaps.py` | `*_heatmap_*.png` | Regenerate P(MOVE) heatmaps from JSONs |
| `deploy_model_images.py` | — | Copies model images to target folder |
| `deploy_heatmaps.py` | — | Copies existing heatmaps to target folder |

## Usage Flags

All scripts support:

- `--target pres` — Deploy to presentation folder (`pres-overleaf.link/pres/pics/`) [default]
- `--target paper` — Deploy to paper folder (`paper-overleaf.link/figs/`)
- `--yes` — Auto-deploy without prompting
- `--no` — Generate only, skip deployment

Without flags, scripts will prompt for deployment confirmation (on target machine only).

## Output Directories

Base path: `~/Documents/d/PM/sakoka-schelling-llm-abm/paper-overleaf/`

| Target | Images | Tables |
|--------|--------|--------|
| `pres` | `paper-overleaf/pres/pics/` | `paper-overleaf/pres/` |
| `paper` | `paper-overleaf/figs/` | `paper-overleaf/` |

## Requirements

- Access to `experiments_with_llama_cpp/` experiment data
- Python with pandas, numpy, matplotlib

## Target Machine

Scripts check for hostname `ECON-0FM96LD-L` before prompting for deployment.
Use `--yes` to override this check and deploy from any machine.
