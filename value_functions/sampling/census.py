#!/usr/bin/env python3
"""The sequential sanity census, start to end, in Python (2026-09-19).

    python value_functions/sampling/census.py --models llama mistral
    python value_functions/sampling/census.py --models all --skip-sims
    python value_functions/sampling/census.py --models qwen --escalate-only
    python value_functions/sampling/census.py --models all --scenarios race_white_black --validate-exact --skip-sims

Per model, under its pinned clock (value_functions/serving.py):
  1. archive any existing census tables/raw for the label (dated folder)
  2. hash the served weights (page-cache read), start the pinned server,
     verify the rendered prompt carries the pin
  3. build the census: 540 cells x n=100, ONE request in flight
     (configs/value_function_scenarios_<key>_sanity.yaml)
  4. compare every cell with the exact table (comparison/sanity_vs_exact.py)
  5. ESCALATE the misses on the SAME server: ONE round of fresh draws per
     miss, sized for power from the census alone (floor --escalate, ceiling
     --max-n), tested on the fresh draws only (sanity_vs_exact.py docstring:
     the pooled multi-round re-test was retired 2026-11-16). It runs here so a
     census is never escalated under a different prompt than the table. A
     census can be escalated once; --escalate-only on an escalated census is
     refused by sanity_vs_exact.py (exit 4) — rebuild it instead.
  6. stop the server, hash the weights again, redraw the exact-vs-sanity
     figures
  7. the 100-run sanity simulation (configs/vf_run_<key>_sanity.yaml) — the
     server is down first: analysis and a served model must not overlap
     (121 GB box).
--scenarios redoes the census of the named scenarios only (2026-11-17, when the
race label lost its class cue and only that scenario's prompts changed). Each
scenario's census is its own table with its own draws and escalation record,
and a pinned server reproduces the regime, so the other scenarios' census
stays valid: only the named ones are archived, rebuilt and escalated, and the
model's verdict is then recomputed over all six.
--validate-exact first re-runs the exact table's own sequential check
(logprob_value_function.py --validate-only) on the same server, the previous
verdict files moving to llm_logprob/validation_data/superseded_<stamp>/.
Each stage is a subprocess of the same interpreter with the repo root on
sys.path, so a failure in one model is reported and the chain continues.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from value_functions.paths import (LOGPROB_DIR, LOGPROB_VALIDATION_DIR, SANITY_DIR,  # noqa: E402
                                   SANITY_TABLES_DIR, SEQCHECK_PLOTS_DIR)
from value_functions.serving import MODELS, PinnedServer, weights_sha256  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=str(REPO))


def say(msg: str, log: Path) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


def ntfy(title: str, body: str, tags: str = "test_tube", prio: str = "default") -> None:
    sh = REPO / "ntfy.sh"
    if sh.exists():
        subprocess.run([str(sh), title, body, tags, prio], capture_output=True)


def run(cmd: list[str], log_to: Path) -> int:
    with open(log_to, "a") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO), env=ENV).returncode


def misses(label: str) -> int | None:
    p = SEQCHECK_PLOTS_DIR / f"sanity_vs_exact_{label}.csv"
    if not p.exists():
        return None
    return sum(r["inside"].strip() != "True" for r in csv.DictReader(open(p)))


def verdict(label: str) -> dict:
    p = SEQCHECK_PLOTS_DIR / f"sanity_vs_exact_{label}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def verdict_line(label: str) -> str:
    v = verdict(label)
    if not v:
        return "no verdict"
    g, s = v["goodness_of_fit"], v["sanity_check"]
    return (f"goodness of fit (paper) {g['verdict']}, p {g['p_mc']:.3f}; per-cell sanity check "
            f"(internal) {s['verdict']}: {s['census_misses']} census misses, "
            f"{s['cells_disagree']} disagree, {s['cells_consistent']} consistent, "
            f"{s['cells_beyond_cap']} beyond cap, {s['cells_not_tested']} not tested")


def archive_existing(label: str, key: str, log: Path, scenarios: list[str] | None = None) -> None:
    """Move the label's census (tables + raw) aside; with `scenarios`, only theirs."""
    pats = [f"vf_{label}-sanity__{s}__*" for s in scenarios] if scenarios else [f"vf_{label}-sanity__*"]
    hits = [p for pat in pats for p in SANITY_TABLES_DIR.glob(pat + ".json")]
    if not hits:
        return
    arch = SANITY_DIR / f"superseded_{datetime.now():%Y%m%d_%H%M}_{key}_census"
    arch.mkdir(parents=True, exist_ok=True)
    for p in hits + [q for pat in pats for q in (SANITY_DIR / "raw").glob(pat)]:
        shutil.move(str(p), str(arch / p.name))
    say(f"{key}: archived the existing census{' of ' + ', '.join(scenarios) if scenarios else ''} "
        f"to {arch.relative_to(REPO)}", log)


def scenario_build_config(build_cfg: Path, key: str, scenarios: list[str]) -> Path:
    """The model's census config cut to `scenarios` (generated, next to the census)."""
    import yaml
    cfg = yaml.safe_load(build_cfg.read_text())
    unknown = [s for s in scenarios if s not in cfg["scenarios"]]
    if unknown:
        raise ValueError(f"{unknown} not in {build_cfg.name}'s scenarios {cfg['scenarios']}")
    cfg["scenarios"] = list(scenarios)
    out = SANITY_DIR / "generated_configs" / f"{build_cfg.stem}__{'+'.join(scenarios)}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# GENERATED by census.py --scenarios from configs/{build_cfg.name}\n"
                   + yaml.safe_dump(cfg, sort_keys=False))
    return out


def validate_exact(label: str, key: str, base_url: str, log: Path) -> str:
    """The exact table's own sequential check, redone on this server."""
    old = [p for p in LOGPROB_VALIDATION_DIR.glob(f"*_{label}*") if p.is_file()]
    if old:
        arch = LOGPROB_VALIDATION_DIR / f"superseded_{datetime.now():%Y%m%d_%H%M}"
        arch.mkdir(parents=True, exist_ok=True)
        for p in old:
            shutil.move(str(p), str(arch / p.name))
    rc = run([PY, "-u", str(REPO / "value_functions/logprob/logprob_value_function.py"), "--label", label,
              "--url", base_url, "--validate-only"], REPO / "logs" / f"census_validate_exact_{key}.log")
    vp = LOGPROB_VALIDATION_DIR / f"validation_{label}.json"
    v = json.loads(vp.read_text()) if vp.exists() else {}
    line = (f"exact-table check {'PASS' if rc == 0 else 'FAIL' if rc == 4 else f'CRASHED (exit {rc})'}: "
            f"{v.get('seq_check_inside_ci')} of {v.get('seq_check_cells')} cells inside, "
            f"{v.get('seq_check_escalated')} escalated")
    say(f"{key}: {line}", log)
    return line


def record_hash(label: str, gguf: Path, note: str, log: Path) -> str:
    h = weights_sha256(gguf)
    f = SEQCHECK_PLOTS_DIR / "launch_probe" / f"gguf_sha256_{label}.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fh:
        fh.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  page-cache read, {note}\n{h}  {gguf.relative_to(REPO)}\n")
    say(f"{label}: weights sha256 {h[:16]}… ({note})", log)
    return h


def census_one(key: str, args: argparse.Namespace, log: Path) -> str:
    m = MODELS[key]
    gguf = REPO / m.gguf
    build_cfg = REPO / "configs" / f"value_function_scenarios_{key}_sanity.yaml"
    run_cfg = REPO / "configs" / f"vf_run_{key}_sanity.yaml"
    if not build_cfg.exists() or not run_cfg.exists():
        return f"{key}: missing sanity configs (run configs/make_sanity_configs.py)"
    scens = args.scenarios
    if scens:
        build_cfg = scenario_build_config(build_cfg, key, scens)
    n_cells = 90 * len(scens) if scens else 540
    if not args.escalate_only:
        archive_existing(m.label, key, log, scens)
    exact_line = ""
    before = record_hash(m.label, gguf, f"before the {key} census under {m.pin}", log)
    t0 = time.time()
    with PinnedServer(m, port=args.port, log_tag=f"census_{key}") as srv:
        (SANITY_DIR / "raw").mkdir(parents=True, exist_ok=True)
        (SANITY_DIR / "raw" / f"vf_{m.label}-sanity__rendered_probe_prompt.txt").write_text(srv.rendered_probe)
        say(f"{key}: server up under {m.pin}; rendered prompt ok" + (f" ({m.pin_check!r} present)" if m.pin_check else ""), log)
        if args.validate_exact:
            exact_line = validate_exact(m.label, key, srv.base_url, log) + "; "
        if not args.escalate_only:
            say(f"{key}: sampling {n_cells} cells x n=100, one request in flight"
                + (f" ({', '.join(scens)} only)" if scens else ""), log)
            rc = run([PY, "-u", str(REPO / "value_functions/sampling/build_value_function.py"),
                      "--config", str(build_cfg), "--plot"], REPO / "logs" / f"census_build_{key}.log")
            if rc != 0:
                return f"{key}: census build failed (exit {rc}, logs/census_build_{key}.log)"
            say(f"{key}: census built in {(time.time()-t0)/3600:.1f} h", log)
        cmp_all = [PY, "-u", str(REPO / "value_functions/comparison/sanity_vs_exact.py"), "--label", m.label]
        cmp = cmp_all + (["--scenarios", *scens] if scens else [])
        rc = run(cmp, REPO / "logs" / f"census_compare_{key}.log")
        if rc == 2:
            return f"{key}: no exact table for {m.label}; nothing to compare against"
        n0 = misses(m.label)
        say(f"{key}: census vs exact table: {n0}/{n_cells} misses before escalation", log)
        if n0 and not args.no_escalate:
            rc = run(cmp + ["--url", srv.chat_url, "--escalate", str(args.escalate),
                            "--max-n", str(args.max_n)],
                     REPO / "logs" / f"census_escalate_{key}.log")
            if rc == 4:
                say(f"{key}: escalation REFUSED or UNRESOLVED (exit 4, logs/census_escalate_{key}.log)", log)
            n1 = verdict_line(m.label)
            say(f"{key}: escalated {n0} miss cell(s) once on the same pinned server: {n1}", log)
        if scens:                    # the model's verdict over ALL its scenarios, from the recorded draws
            run(cmp_all, REPO / "logs" / f"census_compare_{key}.log")
            say(f"{key}: verdict over all scenarios: {verdict_line(m.label)}", log)
        n1 = verdict_line(m.label)
    after = record_hash(m.label, gguf, f"after the {key} census", log)
    if after != before:
        say(f"{key}: WARNING served weights hash changed during the census", log)
    run([PY, str(REPO / "value_functions/logprob/plot_exact_vs_sanity.py")], REPO / "logs" / "census_plots.log")
    if not args.skip_sims:
        say(f"{key}: sanity simulations (100 runs x 6 scenarios), server down", log)
        rc = run([PY, str(REPO / "run_llm_probability_simulation_analysis.py"), "--config-yaml", str(run_cfg),
                  "--config-profile", "production"], REPO / "logs" / f"census_sims_{key}.log")
        if rc != 0:
            return f"{key}: census misses {n0}; {n1}; sanity sims FAILED (logs/census_sims_{key}.log)"
    return f"{key}: {exact_line}census misses {n0}; {n1}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True, help="registry keys, or 'all'")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--escalate", type=int, default=300, help="floor of extra draws per miss cell")
    ap.add_argument("--max-n", type=int, default=1200, help="ceiling of fresh draws per miss cell")
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--escalate-only", action="store_true", help="skip the build; compare + escalate the existing census")
    ap.add_argument("--skip-sims", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="redo the census of these scenarios only; the others' census is kept")
    ap.add_argument("--validate-exact", action="store_true",
                    help="first redo the exact table's sequential check on the same server")
    args = ap.parse_args()
    keys = list(MODELS) if args.models == ["all"] else args.models
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        ap.error(f"unknown model key(s) {unknown}; known: {list(MODELS)}")
    (REPO / "logs").mkdir(exist_ok=True)
    log = REPO / "logs" / f"census_{datetime.now():%Y%m%d_%H%M%S}.log"
    ntfy("census STARTED", f"{', '.join(keys)}; escalate once, {args.escalate}-{args.max_n} fresh draws per miss", "rocket")
    results = []
    for key in keys:
        try:
            results.append(census_one(key, args, log))
        except Exception as e:                             # one model's failure never stops the rest
            results.append(f"{key}: ERROR {e}")
        say(results[-1], log)
    say("ALL DONE: " + " | ".join(results), log)
    ntfy("census DONE", " | ".join(results), "tada")
    return 0 if all("FAILED" not in r and "ERROR" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
