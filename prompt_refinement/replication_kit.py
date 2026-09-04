#!/usr/bin/env python3
"""Bitwise replication kit (audit tier) for the value-function measurements.

    python prompt_refinement/replication_kit.py --generate   # store reference
    python prompt_refinement/replication_kit.py --verify     # PASS/FAIL vs it

Runs a fixed manifest of requests STRICTLY SERIALLY (one in flight) with the
clean protocol (cache_prompt=false + deterministic request_seed) — the one
configuration measured to be bitwise reproducible on a given server build
(20/20 identical across passes, immune to interleaving and to adversarial
cache-on priming; see LLAMA_CPP_SERVING_NOTES.md reproducibility table and
KV_CACHE_SAMPLING_ARTIFACT.md).

Manifest: per scenario × role, one SATURATED cell (0-similar-of-8) and one
TRANSITION cell (chosen at --generate time as the composition whose clean
p̂ is nearest 0.5; recorded in the reference so --verify replays the same
cells) × 25 seeds ⇒ 600 requests ≈ 15–20 min serial.

The reference file records the server fingerprint (/props build_info +
model_path). --verify refuses to compare across different builds — bitwise
identity is only promised per build/model/hardware.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sampling_common import (  # noqa: E402
    GRAMMAR,
    RESULTS_DIR,
    request_seed,
    role_keywords,
    sample_once,
    server_fingerprint,
)
from ratio_prompt_templates import RATIO_CANDIDATES  # noqa: E402
from evaluate_ratio_prompts import render_prompt  # noqa: E402

SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]
STYLE = "R3_dual_count"
LABEL = "gemma-4-31b-chat-grammar"   # default; override with --label
N_SEEDS = 25
REF_PATH = RESULTS_DIR / f"replication_reference_{LABEL}.json"
VF_DIR = RESULTS_DIR / "value_functions"


def transition_cell(scenario, role):
    """Composition with clean p̂ nearest 0.5 for (scenario, role); (0, 2) if no
    artifact is available (matches the artifact-doc's canonical probe cell)."""
    path = VF_DIR / f"vf_{LABEL}__{scenario}__{STYLE}.json"
    if not path.exists():
        return [0, 2]
    vf = json.loads(path.read_text())
    best, best_d = [0, 2], 1.0
    for c in vf["compositions"][role]:
        valid = c["n_move"] + c["n_stay"]
        if valid == 0 or c["n_occupied"] == 0:
            continue
        d = abs(c["n_move"] / valid - 0.5)
        if d < best_d:
            best, best_d = [c["n_similar"], c["n_occupied"]], d
    return best


def build_manifest():
    entries = []
    for sc in SCENARIOS:
        for role in ("red", "blue"):
            for kind, (ns, no) in (("saturated", (0, 8)),
                                   ("transition", transition_cell(sc, role))):
                entries.append({"scenario": sc, "role": role, "kind": kind,
                                "n_similar": ns, "n_occupied": no})
    return entries


def run_manifest(entries, url, model):
    """SERIAL execution — one request in flight at any moment, by construction
    (plain loop, no executor). Returns ordered {key: raw_text}."""
    tpl, fn = RATIO_CANDIDATES[STYLE]
    out = {}
    total = len(entries) * N_SEEDS
    done = 0
    for e in entries:
        kw = role_keywords(e["scenario"], "scenarios_a2.py")[e["role"]]
        prompt, _ = render_prompt(STYLE, tpl, fn, e["n_similar"],
                                  e["n_occupied"], kw)
        for i in range(N_SEEDS):
            seed = request_seed("audit", e["scenario"], STYLE, e["role"],
                                e["n_similar"], e["n_occupied"], i)
            r = sample_once(url, model, prompt, 0.3, GRAMMAR,
                            cache_prompt=False, seed=seed)
            key = (f"{e['scenario']}|{e['role']}|{e['n_similar']}"
                   f"|{e['n_occupied']}|{i}")
            out[key] = r["text"]
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total}", flush=True)
    return out


def digest(outputs):
    h = hashlib.sha256()
    for k in sorted(outputs):
        h.update(k.encode()); h.update(b"\x00")
        h.update(outputs[k].encode()); h.update(b"\x00")
    return h.hexdigest()


def main() -> int:
    # Declared up front: LABEL is READ below as the --label default, and a
    # `global` after that read is a SyntaxError ("used prior to global
    # declaration") that only surfaces at compile time, not from ast.parse.
    # That silently cost llama/qwen/deepseek their bitwise audit (2026-08-23..27).
    global LABEL, REF_PATH
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--verify", action="store_true")
    ap.add_argument("--llm-url", default="http://localhost:8085/v1/chat/completions")
    ap.add_argument("--model", default="gemma-4-31B-it-Q5_K_M")
    ap.add_argument("--label", default=LABEL,
                    help="value-function label (vf_<label>__<scenario>__<style>.json); "
                         "also names the reference file")
    args = ap.parse_args()

    LABEL = args.label
    REF_PATH = RESULTS_DIR / f"replication_reference_{LABEL}.json"

    fp = server_fingerprint(args.llm_url)
    if fp is None:
        sys.exit("server /props unreachable — is llama-server running?")
    print(f"server: {fp}")

    if args.generate:
        entries = build_manifest()
        print(f"generating reference: {len(entries)} cells x {N_SEEDS} seeds, serial")
        outputs = run_manifest(entries, args.llm_url, args.model)
        REF_PATH.write_text(json.dumps({
            "label": LABEL, "style": STYLE, "n_seeds": N_SEEDS,
            "server": fp, "manifest": entries,
            "outputs": outputs, "sha256": digest(outputs),
        }, indent=1))
        print(f"wrote {REF_PATH}  sha256={digest(outputs)[:16]}...")
        return 0

    ref = json.loads(REF_PATH.read_text())
    if ref["server"] != fp:
        sys.exit(f"REFUSING to verify: server fingerprint differs.\n"
                 f"  reference: {ref['server']}\n  current:   {fp}\n"
                 f"Bitwise identity is only promised per build/model.")
    print(f"verifying against {REF_PATH.name} (sha256={ref['sha256'][:16]}...)")
    outputs = run_manifest(ref["manifest"], args.llm_url, args.model)
    mismatches = [k for k in ref["outputs"]
                  if outputs.get(k) != ref["outputs"][k]]
    if not mismatches:
        assert digest(outputs) == ref["sha256"]
        print(f"PASS — all {len(outputs)} outputs bitwise identical to reference")
        return 0
    print(f"FAIL — {len(mismatches)}/{len(ref['outputs'])} outputs differ:")
    for k in mismatches[:10]:
        print(f"  {k}: ref={ref['outputs'][k]!r} now={outputs.get(k)!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
