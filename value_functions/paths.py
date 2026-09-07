"""Single home for where the value-function code and its stores live (2026-09-07).

    value_functions/
      sampling/        sampled tables: build_value_function.py (+ rulers, plots)
      logprob/         exact tables from token log-probabilities
      batch_numerics/  the batch-state artifact study that retired sampling
      comparison/      rank stability and the cross-model figures
      results/         (gitignored) the tables themselves — see below
    experiments_with_llama_cpp/cross_model/   the cross-model figures + tables

Stores under results/:
    sampled/       vf_<label>__<scenario>__<style>.json (the retired -vf-r3 tables),
                   the -half arms, multisplit_<label>_*/ rulers, sanity/ (the
                   sequential -sanity cross-check tables) and cacheon_archive/
                   (the cache-on artifacts kept as evidence)
    llm_logprob/   vf_<label>-lp__… exact tables (what the _lp run configs use),
                   vflp_<label>__… per-cell probabilities, validation_*.json,
                   seqcheck_*.csv, raw/ traces
    figures/       value-function figures, one subfolder per kind:
                     vf_plots/          vfS_* / vfH_* table plots (plot_value_functions.py)
                     ruler_scaling/     ruler_scaling.{csv,png} (vf_ruler_scaling.py)
                     scale_dependence/  vf_<label>_scale_dependence.png (vf_scale_dependence_probe.py)
                     sampling_requirements.png and vf_sweep_comparison.png at the top
    chance_null/   metric_null_*.json — the random-allocation null cached per board

Before 2026-09-07 the sampled store was prompt_refinement/results/value_functions/,
the exact store prompt_refinement/results/value_functions_logprob/ and then
llm_log_probs/value_functions_logprob/, and the cross-model figures sat in
prompt_refinement/results/figures/. value_functions/migrate_stores.py moves
existing data into this layout.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VF_ROOT = REPO_ROOT / "value_functions"
RESULTS_DIR = VF_ROOT / "results"
SAMPLED_DIR = RESULTS_DIR / "sampled"
SANITY_DIR = SAMPLED_DIR / "sanity"
LOGPROB_DIR = RESULTS_DIR / "llm_logprob"
FIGURES_DIR = RESULTS_DIR / "figures"
VF_PLOTS_DIR = FIGURES_DIR / "vf_plots"
RULER_SCALING_DIR = FIGURES_DIR / "ruler_scaling"
SCALE_DEPENDENCE_DIR = FIGURES_DIR / "scale_dependence"
CHANCE_NULL_DIR = RESULTS_DIR / "chance_null"
CROSS_MODEL_DIR = REPO_ROOT / "experiments_with_llama_cpp" / "cross_model"

PROMPT_REFINEMENT_DIR = REPO_ROOT / "prompt_refinement"   # prompt templates + sampling harness
ANALYSIS_TOOLS_DIR = REPO_ROOT / "analysis_tools"

# Repo-relative forms, for configs and messages.
SAMPLED_REL = "value_functions/results/sampled"
SANITY_REL = "value_functions/results/sampled/sanity"
LOGPROB_REL = "value_functions/results/llm_logprob"


def add_import_paths():
    """Make the repo root, prompt_refinement/, analysis_tools/ and the
    value_functions subpackages importable by bare name, the way the
    scripts here have always been run (python value_functions/<sub>/<script>.py)."""
    for p in (REPO_ROOT, PROMPT_REFINEMENT_DIR, ANALYSIS_TOOLS_DIR,
              VF_ROOT / "sampling", VF_ROOT / "logprob", VF_ROOT / "comparison"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
