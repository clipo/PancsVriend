#!/usr/bin/env python3
"""One-time migration of the value-function stores into value_functions/results/
(layout: value_functions/paths.py). Dry run by default; --apply moves.

    python value_functions/migrate_stores.py            # show what would move
    python value_functions/migrate_stores.py --apply    # do it

Moves (all gitignored data, so git sees none of it):
    prompt_refinement/results/value_functions/                 -> value_functions/results/sampled/
    prompt_refinement/results/value_functions_sanity/          -> value_functions/results/sampled/sanity/
    prompt_refinement/results/value_functions_cacheon_archive/ -> value_functions/results/sampled/cacheon_archive/
    llm_log_probs/value_functions_logprob/*                    -> value_functions/results/llm_logprob/   (llm_log_probs/ removed)
    prompt_refinement/results/figures/cross_model_*            -> experiments_with_llama_cpp/cross_model/
    prompt_refinement/results/figures_backup_8models_*         -> experiments_with_llama_cpp/cross_model/backup_8models_*/
    prompt_refinement/results/figures/{vfS_,vfH_,vf_,ruler_,sampling_}* -> value_functions/results/figures/
    prompt_refinement/results/metric_null_*.json               -> value_functions/results/chance_null/
    prompt_refinement/batch_numerics/results/* (untracked leftovers) -> value_functions/batch_numerics/results/
    prompt_refinement/batch_numerics/*.sh (gitignored runner)        -> value_functions/batch_numerics/

and rewrites the same path strings inside the gitignored runner scripts (*.sh
in the repo root and prompt_refinement/), which git cannot carry.

Refuses to run while a value-function job is alive (the sampler, the exact
extractor, the batch-numerics probe or the sanity chain): a shell script that is
executing is read lazily from disk, and a store that is being written must not
move under its writer.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from value_functions import paths as P  # noqa: E402

REPO = P.REPO_ROOT
PR_RESULTS = P.PROMPT_REFINEMENT_DIR / "results"

DIR_MOVES = [
    (PR_RESULTS / "value_functions", P.SAMPLED_DIR),
    (PR_RESULTS / "value_functions_sanity", P.SANITY_DIR),
    (PR_RESULTS / "value_functions_cacheon_archive", P.SAMPLED_DIR / "cacheon_archive"),
    (REPO / "llm_log_probs" / "value_functions_logprob", P.LOGPROB_DIR),
]
GLOB_MOVES = [
    (PR_RESULTS / "figures", "cross_model_*", P.CROSS_MODEL_DIR),
    (PR_RESULTS, "figures_backup_8models_*", P.CROSS_MODEL_DIR),
    (PR_RESULTS / "figures", "vfS_*", P.FIGURES_DIR),
    (PR_RESULTS / "figures", "vfH_*", P.FIGURES_DIR),
    (PR_RESULTS / "figures", "vf_*", P.FIGURES_DIR),
    (PR_RESULTS / "figures", "ruler_*", P.FIGURES_DIR),
    (PR_RESULTS / "figures", "sampling_requirements.*", P.FIGURES_DIR),
    (PR_RESULTS, "metric_null_*.json", P.CHANCE_NULL_DIR),
    (P.PROMPT_REFINEMENT_DIR / "batch_numerics" / "results", "*", P.VF_ROOT / "batch_numerics" / "results"),
    (P.PROMPT_REFINEMENT_DIR / "batch_numerics", "*.sh", P.VF_ROOT / "batch_numerics"),   # gitignored runner
]
# Path strings to rewrite in the gitignored runner scripts (order matters).
TEXT_SUBS = [
    (r"llm_log_probs/value_functions_logprob", P.LOGPROB_REL),
    (r"prompt_refinement/results/value_functions_sanity", P.SANITY_REL),
    (r"prompt_refinement/results/value_functions_cacheon_archive", P.SAMPLED_REL + "/cacheon_archive"),
    (r"prompt_refinement/results/value_functions(?![_a-zA-Z])", P.SAMPLED_REL),
    (r"prompt_refinement/results/figures/cross_model", "experiments_with_llama_cpp/cross_model/cross_model"),
    (r"prompt_refinement/build_value_function\.py", "value_functions/sampling/build_value_function.py"),
    (r"prompt_refinement/plot_value_functions\.py", "value_functions/sampling/plot_value_functions.py"),
    (r"prompt_refinement/logprob_value_function\.py", "value_functions/logprob/logprob_value_function.py"),
    (r"prompt_refinement/batch_numerics", "value_functions/batch_numerics"),
    (r"analysis_tools/vf_multisplit_check\.py", "value_functions/sampling/vf_multisplit_check.py"),
    (r"analysis_tools/vf_ruler_scaling\.py", "value_functions/sampling/vf_ruler_scaling.py"),
    (r"analysis_tools/vf_rank_stability\.py", "value_functions/comparison/vf_rank_stability.py"),
    (r"analysis_tools/cross_model_vf_comparison\.py", "value_functions/comparison/cross_model_vf_comparison.py"),
]
JOB_PATTERNS = ("build_value_function.py", "logprob_value_function.py", "batch_numerics_probe.py",
                "run_batch_numerics_study.sh", "run_sanity_vf_chain.sh", "run_logprob_vf_campaign.sh",
                "run_lp_dispatcher.sh", "run_vf_prod10k_queue.sh", "vf_multisplit_check.py",
                "run_llm_probability_simulation_analysis.py")


def live_jobs():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [l for l in out.splitlines()[1:]
            if any(p in l for p in JOB_PATTERNS) and not l.split()[0] == me]


def rel(p):
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def move(src, dst, apply, log):
    """Move src to dst; if dst exists as a directory, merge src's entries into it."""
    if not src.exists():
        return
    if dst.exists() and dst.is_dir() and src.is_dir():
        for child in sorted(src.iterdir()):
            move(child, dst / child.name, apply, log)
        if apply and not any(src.iterdir()):
            src.rmdir()
        return
    if dst.exists():
        log.append(f"SKIP  {rel(src)} -> {rel(dst)} (destination exists)")
        return
    log.append(f"move  {rel(src)} -> {rel(dst)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def rewrite_scripts(apply, log):
    scripts = sorted(list(REPO.glob("*.sh")) + list(P.PROMPT_REFINEMENT_DIR.glob("*.sh"))
                     + list((P.VF_ROOT / "batch_numerics").glob("*.sh"))
                     + list((P.PROMPT_REFINEMENT_DIR / "batch_numerics").glob("*.sh")))
    for s in scripts:
        text = s.read_text()
        new = text
        for pat, repl in TEXT_SUBS:
            new = re.sub(pat, repl, new)
        if new != text:
            n = sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b)
            log.append(f"edit  {rel(s)} ({n} line(s))")
            if apply:
                s.write_text(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="perform the moves (default: dry run)")
    ap.add_argument("--force", action="store_true", help="proceed even if a value-function job is running")
    args = ap.parse_args()

    jobs = live_jobs()
    if jobs and not args.force:
        print("REFUSING: value-function jobs are running; wait for them or pass --force:")
        for j in jobs:
            print("   ", j[:140])
        return 2

    log = []
    for src, dst in DIR_MOVES:
        move(src, dst, args.apply, log)
    for base, pattern, dst_dir in GLOB_MOVES:
        if base.exists():
            for src in sorted(base.glob(pattern)):
                move(src, dst_dir / src.name, args.apply, log)
    rewrite_scripts(args.apply, log)
    if args.apply:
        for empty in (REPO / "llm_log_probs", P.PROMPT_REFINEMENT_DIR / "batch_numerics" / "results",
                      P.PROMPT_REFINEMENT_DIR / "batch_numerics"):
            if empty.is_dir() and not any(empty.rglob("*")):
                shutil.rmtree(empty)
                log.append(f"rmdir {rel(empty)} (empty)")
    print("\n".join(log) if log else "nothing to do")
    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {len(log)} action(s)."
          + ("" if args.apply else " Re-run with --apply to perform them."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
