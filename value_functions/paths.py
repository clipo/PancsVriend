"""Single home for where the value-function code and its stores live (2026-09-07).

    value_functions/
      sampling/        sampled tables: build_value_function.py (+ rulers, plots)
      logprob/         exact tables from token log-probabilities
      batch_numerics/  the batch-state artifact study that retired sampling
      comparison/      rank stability and the cross-model figures
      results/         (gitignored) the tables themselves — see below
    experiments_with_llama_cpp/cross_model/   the cross-model figures + tables

Stores under results/:
    sampled/       the retired -vf-r3 sampling campaign. TOP LEVEL IS EMPTY of
                   loose files since 2026-09-07; everything is a subfolder:
                     tables/            vf_<label>__<scenario>__<style>.json —
                                        the vf-1 tables (incl. the -half arms)
                     raw/               per-draw replies, one record per LLM call
                     multisplit/        multisplit_<label>_*/ rulers, scanned by
                                        the rank-stability stage
                     vf_mapping_plots/  per-artifact P(MOVE)-vs-ratio figures
                     cacheon_archive/   cache-on artifacts kept as evidence
    sampled_small/ the sequential n=100 cross-check campaign (-vf-s). Its own
                   store since 2026-09-07 — a separate small campaign, not a
                   subset of the retired -vf-r3 one. SAME arrangement:
                     tables/  raw/  vf_mapping_plots/
                   vf_mapping_plots/ holds the COMBINED all-scenario figures —
                     vfS_<label>__<style>   curves, one panel per scenario
                     vfH_<label>__<style>   surfaces, scenarios x roles
                   and by_scenario/ holds the per-scenario figures, each the
                   curve and both role surfaces SIDE BY SIDE. Both routes write
                   them automatically at the end of a mapping run.
    llm_logprob/   the exact (log-probability) extraction, SAME arrangement:
                     tables/            vf_<label>-lp__… vf-1 tables — what the
                                        _lp run configs load
                     raw/               vflp_*_states.jsonl.gz — the extraction
                                        record: a {"_meta": true, ...} header line
                                        then one line per (role, cell). This is
                                        the ONLY per-cell record; the vflp_*.json
                                        that duplicated it was dropped 2026-09-07
                     validation_data/   seqcheck_*.csv (the sequential test) and
                                        validation_*.json (its PASS/FAIL verdict)
                     seqcheck_plots/    their figures — the fresh sequential work
    figures/       value-function figures, one subfolder per kind:
                     ruler_scaling/     ruler_scaling.{csv,png} (vf_ruler_scaling.py)
                     scale_dependence/  vf_<label>_scale_dependence.png (vf_scale_dependence_probe.py)
                     sampling_requirements.png and vf_sweep_comparison.png at the top
    chance_null/   metric_null_*.json — the random-allocation null cached per board

Before 2026-09-07 the sampled store was prompt_refinement/results/value_functions/,
the exact store prompt_refinement/results/value_functions_logprob/ and then
llm_log_probs/value_functions_logprob/, and the cross-model figures sat in
prompt_refinement/results/figures/; the data was moved into this layout that day.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VF_ROOT = REPO_ROOT / "value_functions"
RESULTS_DIR = VF_ROOT / "results"
SAMPLED_DIR = RESULTS_DIR / "sampled"
SANITY_DIR = RESULTS_DIR / "sampled_small"   # the sequential n=100 cross-check campaign
# Subfolders of the sampled store (2026-09-07), so its top level holds only
# the vf_*.json tables. MULTISPLIT_DIR is the ruler store that the
# rank-stability stage scans; VF_MAPPING_PLOTS_REL is appended to whichever
# out_dir a build writes to, so the sanity arm gets its own copy.
MULTISPLIT_DIR = SAMPLED_DIR / "multisplit"
# The vf_*.json tables themselves live in a tables/ subfolder of their store
# (2026-09-07): 102 files in sampled/ and 108 in llm_logprob/ buried every
# other artifact. TABLES_REL is relative so a build's out_dir (including the
# sanity arm's) gets the same layout without another constant.
TABLES_REL = "tables"
SAMPLED_TABLES_DIR = SAMPLED_DIR / TABLES_REL
SANITY_TABLES_DIR = SANITY_DIR / TABLES_REL
# Fresh sequential draws bought to adjudicate between disagreeing exact
# extractions, read by build_consensus_table.py (adjudication_<label>.json).
# Kept apart from the sanity tables: those are a uniform n=100 census, these
# are targeted, pre-committed sample sizes on the cells that need them.
# NOTHING WRITES IT YET: the cross-session disagreement that motivated it was
# the chat template's date (LLAMA_CPP_SERVING_NOTES.md §6); with the clock
# pinned every re-extraction has reproduced, so no cell has needed adjudicating.
ADJUDICATION_DIR = SANITY_DIR / "adjudication"
VF_MAPPING_PLOTS_REL = "vf_mapping_plots"
# Combined figures (all scenarios in one file: vfS_ curves, vfH_ surfaces) sit
# directly in vf_mapping_plots/ — those are the ones normally looked at. The
# PER-SCENARIO figures (curve + both role surfaces side by side) go one level
# down so they do not bury them (user, 2026-09-07).
BY_SCENARIO_REL = "by_scenario"
LOGPROB_DIR = RESULTS_DIR / "llm_logprob"
LOGPROB_TABLES_DIR = LOGPROB_DIR / TABLES_REL        # vf_<label>-lp__… consumer tables
# The extraction trace. Since 2026-09-07 it is the ONLY per-cell record: the
# vflp_*.json that used to sit beside it held the same 90 cells plus a meta
# block, and that meta is now a leading {"_meta": true, ...} line in the gz —
# the same header convention the sampled raw files already use. 20 MB of
# duplicate JSON removed, and the two routes now share one raw format.
# Nothing in the simulation or analysis pipeline reads it; only
# `logprob_value_function.py --validate-only` does.
LOGPROB_RAW_DIR = LOGPROB_DIR / "raw"
# The exact store keeps its validation split into data and figures so the
# tables at its top level are only ever value functions (2026-09-07).
LOGPROB_VALIDATION_DIR = LOGPROB_DIR / "validation_data"   # seqcheck_*.csv + validation_*.json
SEQCHECK_PLOTS_DIR = LOGPROB_DIR / "seqcheck_plots"        # their figures
FIGURES_DIR = RESULTS_DIR / "figures"
RULER_SCALING_DIR = FIGURES_DIR / "ruler_scaling"
SCALE_DEPENDENCE_DIR = FIGURES_DIR / "scale_dependence"
CHANCE_NULL_DIR = RESULTS_DIR / "chance_null"
# Date-sensitivity studies (2026-09-12): llama.cpp bakes the server's date into
# Llama-3 / Mistral prompts, so the exact table is a function of the date. One
# folder per model: tables/<date>/ (full exact tables + traces), simulations/
# <date>/ (pipeline runs, their own run_root so the cross-model stage never sees
# them), figures/, REPORT.md. The canonical table stays in llm_logprob/tables.
DATE_SENSITIVITY_DIR = RESULTS_DIR / "date_sensitivity"
CROSS_MODEL_DIR = REPO_ROOT / "experiments_with_llama_cpp" / "cross_model"

PROMPT_REFINEMENT_DIR = REPO_ROOT / "prompt_refinement"   # prompt templates + sampling harness
ANALYSIS_TOOLS_DIR = REPO_ROOT / "analysis_tools"

# Repo-relative forms, for configs and messages.
SAMPLED_REL = "value_functions/results/sampled"
SANITY_REL = "value_functions/results/sampled_small"
LOGPROB_REL = "value_functions/results/llm_logprob"


def add_import_paths():
    """Make the repo root, prompt_refinement/, analysis_tools/ and the
    value_functions subpackages importable by bare name, the way the
    scripts here have always been run (python value_functions/<sub>/<script>.py)."""
    for p in (REPO_ROOT, PROMPT_REFINEMENT_DIR, ANALYSIS_TOOLS_DIR,
              VF_ROOT / "sampling", VF_ROOT / "logprob", VF_ROOT / "comparison"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
