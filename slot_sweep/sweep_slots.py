#!/usr/bin/env python
"""
sweep_slots.py — find the THROUGHPUT-EFFICIENT number of llama.cpp slots (-np).

WHY THIS EXISTS
---------------
"How many slots FIT in memory" and "how many slots are FAST" are different
questions, and the memory ceiling is usually far above the efficient point. This
script answers the second question by MEASURING -- it times the real workload at
each slot count -- because throughput cannot be derived from byte accounting:

  * Decode is memory-bandwidth-bound. Batching amortises the cost of streaming
    the weights, so 1 -> 8 slots is nearly free. But once bandwidth saturates,
    extra slots stop adding aggregate throughput and merely slice the same pie
    thinner -- latency per request grows for no gain.
  * llm_runner.py gives up on a request after LLM_CLIENT_TIMEOUT_S seconds and
    RAISES (no mechanical fallback), so an over-large -np does not just run slow,
    it crashes the run.

The number this prints is what you pin as `processes:` in the run YAML; the
run_*.sh launcher sets the server's -np directly from it.

WHAT IT MEASURES
----------------
For each candidate slot count S it launches a dedicated llama-server with
`-np S`, saturates it with S concurrent clients issuing the REAL simulation
prompt (imported from context_scenarios, same payload as llm_runner: raw
/v1/completions, max_tokens=5, pinned pure-temperature sampler), and records
steady-state throughput and the latency distribution.

Results land in  slot_sweep/results/<label>/  -- one folder per model:
    sweep_<label>.csv     per-slot-count stats
    sweep_<label>.png     throughput / latency / scaling-efficiency plots
    sweep_<label>.json    stats + the recommendation + run metadata

USAGE
-----
    # a model already in the MODELS registry below (legacy context_scenarios.py prompt)
    python slot_sweep/sweep_slots.py --model llama-3.3-70b-q4

    # measure the workload PRODUCTION actually sends: A3 prompts + grammar.
    # --scenario-file swaps the prompt set; --llm-style swaps the payload
    # (endpoint + grammar), both routed through llm_runner so the bytes match.
    # Use a distinct --label: prompt style changes the numbers, so results are
    # not comparable across styles.
    python slot_sweep/sweep_slots.py \
        --model-path llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
        --label llama-3.3-70b-q4-a3 \
        --scenario-file scenarios_a3.py --llm-style completions+grammar \
        --slots 1,2,4,8,12,16,24

    # any other GGUF, no registry edit needed
    python slot_sweep/sweep_slots.py \
        --model-path llms/some-model.gguf --label some-model --slots 1,4,8,16

Then read the recommendation off the plot and pin it by hand in the run YAML:

    contexts_args:
      processes: <N>        # the run_*.sh launcher sets the server's -np from this

The GPU must be otherwise IDLE -- a second llama-server competing for memory
bandwidth poisons every number here. The script refuses to start if it finds one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from context_scenarios import CONTEXT_SCENARIOS          # noqa: E402
from llm_runner import (                                 # noqa: E402  (single source of truth)
    SAMPLER_PARAMS,
    apply_scenario_file,
    build_llm_request,
    resolve_llm_style,
)

# --------------------------------------------------------------------------- #
# MODEL REGISTRY — add a model here to make it a one-flag sweep.
# `slots` is the ladder of -np values to test. Include 1 as the batch-1 baseline;
# it is what every speedup is measured against.
# --------------------------------------------------------------------------- #
MODELS = {
    "llama-3.3-70b-q4": {
        "path": REPO / "llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "slots": [1, 2, 4, 8, 12, 16, 24],
    },
    "gemma-4-31b-q5": {
        "path": REPO / "llms/gemma-4-31B-it-Q5_K_M.gguf",
        "slots": [1, 2, 4, 8, 16, 24, 32],
    },
}

# --------------------------------------------------------------------------- #
# Workload constants — MUST track llm_runner.py or the sweep measures fiction.
# --------------------------------------------------------------------------- #
MAX_TOKENS = 5              # llm_runner.py payload
TEMPERATURE = 0.3           # configs/llama_cpp_run_*.yaml
LLM_CLIENT_TIMEOUT_S = 20   # llm_runner.py: requests.post(..., timeout=20) then RAISES
CTX_PER_SLOT = 2048         # run_llama70b.sh
BENCH_TIMEOUT_S = 180       # generous: we want to MEASURE slow configs, not fail them

# --------------------------------------------------------------------------- #
# Sample size — how many requests per slot count?
#
# We stop adaptively rather than fixing n, because the right n depends on a
# variance we do not know until we start measuring.
#
# For the MEAN we want a 95% CI half-width within REL_MOE of the mean. With a
# coefficient of variation c, that needs  n >= (1.96 * c / REL_MOE)^2 :
#       c = 0.10  ->  n >=  16
#       c = 0.20  ->  n >=  62
#       c = 0.30  ->  n >= 139
# So we sample until the CI is tight enough, floored at MIN_N and capped at
# MAX_N. Low-variance configs finish in ~30 requests; noisy ones keep going.
#
# MIN_N is set at 50 rather than the ~16-30 the mean alone would need, because we
# also report p95 to check headroom against the client timeout, and a p95 from
# fewer than ~50 samples is close to meaningless. MAX_N caps the tail: at ~5s a
# request and S in flight, 150 requests is ~8 min at S=1 and well under a minute
# at S=16, so the whole ladder stays under an hour even for a 70B.
# --------------------------------------------------------------------------- #
MIN_N = 50
MAX_N = 150
REL_MOE = 0.05              # target 95% CI half-width, as a fraction of the mean
Z95 = 1.96

# Recommendation policy
LATENCY_SAFETY = 0.5        # require p95 <= 50% of the client timeout
KNEE_FRACTION = 0.95        # smallest S reaching 95% of peak throughput


# --------------------------------------------------------------------------- #
# Workload: the real prompt, on random neighbourhoods
# --------------------------------------------------------------------------- #
def build_prompts(n, scenario="baseline", seed=0):
    """Real simulation prompts over random 3x3 neighbourhoods.

    Mirrors LLMAgent.get_context_grid: S=same type, O=opposite, E=empty,
    #=out of bounds, X=self at centre. Neighbourhoods are randomised (as they are
    in a real run) so we do NOT accidentally measure llama.cpp's prompt cache
    replaying one identical prefix -- that would flatter every config.
    """
    rng = random.Random(seed)
    info = CONTEXT_SCENARIOS[scenario]
    prompts = []
    for _ in range(n):
        cells = []
        for i in range(3):
            row = []
            for j in range(3):
                if i == 1 and j == 1:
                    row.append("X")
                else:
                    # weights approximate a mid-run board: mostly occupied, some empty,
                    # occasional edge cell.
                    row.append(rng.choices(["S", "O", "E", "#"], weights=[38, 38, 20, 4])[0])
            cells.append(" ".join(row))
        prompts.append(info["prompt_template"].format(
            agent_type=info["type_a"],
            opposite_type=info["type_b"],
            context="\n".join(cells),
        ))
    return prompts


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def gpu_busy():
    """Any other llama-server already holding the GPU? Its bandwidth use would
    contaminate every measurement, so we refuse rather than report a bad number.

    `pgrep -x` matches the EXECUTABLE NAME. Do not use `pgrep -f`: that matches
    full command lines, so any shell wrapper merely mentioning "llama-server"
    (e.g. `env LLAMA_SERVER_BIN=.../llama-server ...`) matches itself and the
    sweep refuses to start against a phantom.
    """
    try:
        pids = subprocess.check_output(["pgrep", "-x", "llama-server"], text=True).split()
    except subprocess.CalledProcessError:
        return []

    busy = []
    for pid in pids:
        try:
            cmd = subprocess.check_output(["ps", "-p", pid, "-o", "args="], text=True).strip()
        except subprocess.CalledProcessError:
            continue
        if "autoslots-probe" in cmd or "slot-sweep" in cmd:   # our own short-lived servers
            continue
        busy.append(f"{pid} {cmd}")
    return busy


# Servers we own. A llama-server orphaned by a crashed/killed sweep keeps its
# weights resident (tens of GB) and silently starves the next job -- so make it
# impossible to leak one by any exit path we can actually intercept.
_OWNED: list[subprocess.Popen] = []


def _reap(*_args):
    for p in list(_OWNED):
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=20)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    _OWNED.clear()


def _install_cleanup():
    import atexit
    import signal
    atexit.register(_reap)                       # normal exit + unhandled exception
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: (_reap(), sys.exit(128 + s)))
    # SIGKILL cannot be caught. `kill -9` on the sweep, or a reboot, still orphans
    # the current server -- check with `pgrep -x llama-server` afterwards.


def launch_server(bin_path, model_path, slots, port, ngl, log_path):
    ctx = slots * CTX_PER_SLOT
    cmd = [bin_path, "-m", str(model_path), "--alias", "slot-sweep",
           "-ngl", str(ngl), "-fa", "on",
           "-np", str(slots), "-c", str(ctx),
           "--host", "127.0.0.1", "--port", str(port)]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    _OWNED.append(proc)
    return proc, log, ctx


def wait_ready(port, proc, timeout=900):
    """Poll /health until the model is loaded (a 70B takes minutes)."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died during load (exit {proc.returncode}) — see its log")
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server not ready after {timeout}s")


def stop_server(proc, log):
    proc.terminate()
    try:
        proc.communicate(timeout=30)
    except Exception:
        proc.kill()
        proc.communicate()
    log.close()
    if proc in _OWNED:
        _OWNED.remove(proc)
    # llama.cpp frees unified memory lazily; give the next launch a clean slate.
    time.sleep(5)


def verify_slots(port, expected):
    """The server is the authority on its own slot count — confirm -np took."""
    try:
        got = requests.get(f"http://127.0.0.1:{port}/props", timeout=5).json().get("total_slots")
        if got is not None and int(got) != expected:
            print(f"    ! server reports {got} slots, expected {expected}")
        return int(got) if got else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Benchmark one slot count
# --------------------------------------------------------------------------- #
def one_request(url, model, prompt, llm_style=None):
    """One timed request, built by llm_runner.build_llm_request so the measured
    workload is byte-identical to what the simulation sends for this llm_style
    (endpoint, grammar, max_tokens, sampler). llm_style=None keeps the historical
    raw /v1/completions payload, so old sweeps stay comparable."""
    if llm_style is None:
        request_url = url
        payload = {
            "model": model, "prompt": prompt, "stream": False,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            **SAMPLER_PARAMS,
        }
    else:
        request_url, payload = build_llm_request(
            url, llm_style, model, prompt, TEMPERATURE, max_tokens=MAX_TOKENS)
    t0 = time.perf_counter()
    r = requests.post(request_url, json=payload, timeout=BENCH_TIMEOUT_S)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt


def ci_halfwidth(samples):
    if len(samples) < 2:
        return float("inf")
    return Z95 * statistics.stdev(samples) / math.sqrt(len(samples))


def bench(port, model_label, slots, prompts, min_n, max_n, llm_style=None):
    """Saturate the server with `slots` concurrent clients; sample adaptively.

    Throughput is measured in steady state (post-warmup, all workers always busy)
    as completed_requests / wall_time — which is exactly the rate the simulation
    would see with `processes == slots`.
    """
    url = f"http://127.0.0.1:{port}/v1/completions"

    # Warmup: first requests pay CUDA graph capture / allocator costs. Discard.
    with ThreadPoolExecutor(max_workers=slots) as ex:
        list(ex.map(lambda p: one_request(url, model_label, p, llm_style), prompts[:slots]))

    lat, errors = [], 0
    pool = prompts[slots:]
    idx = 0
    stop = False
    t_start = time.perf_counter()

    def worker():
        nonlocal idx, errors, stop
        while not stop:
            p = pool[idx % len(pool)]
            idx += 1
            try:
                dt = one_request(url, model_label, p, llm_style)
            except Exception:
                errors += 1
                continue
            lat.append(dt)
            n = len(lat)
            if n >= min_n and (ci_halfwidth(lat) <= REL_MOE * statistics.fmean(lat) or n >= max_n):
                stop = True

    with ThreadPoolExecutor(max_workers=slots) as ex:
        futures = [ex.submit(worker) for _ in range(slots)]
        for f in futures:
            f.result()

    wall = time.perf_counter() - t_start
    n = len(lat)
    if n == 0:
        raise RuntimeError("every request failed")

    mean = statistics.fmean(lat)
    sd = statistics.stdev(lat) if n > 1 else 0.0
    srt = sorted(lat)
    return {
        "slots": slots,
        "n": n,
        "errors": errors,
        "throughput_rps": n / wall,
        "tokens_per_s": n * MAX_TOKENS / wall,
        "latency_mean_s": mean,
        "latency_sd_s": sd,
        "latency_cv": sd / mean if mean else 0.0,
        "latency_ci95_s": ci_halfwidth(lat),
        "latency_p50_s": srt[int(0.50 * (n - 1))],
        "latency_p95_s": srt[int(0.95 * (n - 1))],
        "latency_max_s": srt[-1],
        "wall_s": wall,
        "hit_max_n": n >= max_n,
    }


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def recommend(rows):
    """Highest throughput that keeps p95 latency safely under the client timeout,
    then backed off to the KNEE: the smallest slot count still within
    KNEE_FRACTION of peak throughput. Paying latency for a <5% throughput gain is
    a bad trade when the client raises on timeout."""
    budget = LATENCY_SAFETY * LLM_CLIENT_TIMEOUT_S
    safe = [r for r in rows if r["latency_p95_s"] <= budget]
    if not safe:
        best = min(rows, key=lambda r: r["latency_p95_s"])
        return best, (f"NO slot count keeps p95 under {budget:.0f}s "
                      f"({LATENCY_SAFETY:.0%} of the {LLM_CLIENT_TIMEOUT_S}s client timeout). "
                      f"Lowest-latency option shown; raise the timeout in llm_runner.py "
                      f"or accept the risk of a run-killing timeout.")
    peak = max(safe, key=lambda r: r["throughput_rps"])
    knee = min((r for r in safe
                if r["throughput_rps"] >= KNEE_FRACTION * peak["throughput_rps"]),
               key=lambda r: r["slots"])
    note = (f"peak throughput at {peak['slots']} slots "
            f"({peak['throughput_rps']:.3f} req/s); {knee['slots']} slots reaches "
            f"{knee['throughput_rps'] / peak['throughput_rps']:.0%} of it with lower latency.")
    return knee, note


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot(rows, label, pick, outfile):
    s = [r["slots"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"llama.cpp slot sweep — {label}", fontsize=14, fontweight="bold")

    # 1. Throughput — the number that decides total experiment wall-clock.
    ax = axes[0]
    ax.plot(s, [r["throughput_rps"] for r in rows], "o-", lw=2, color="#2b6cb0")
    ax.axvline(pick["slots"], ls="--", c="#38a169", lw=2,
               label=f"recommended: {pick['slots']}")
    ax.set(xlabel="slots (-np)", ylabel="throughput (requests/s)",
           title="Throughput — higher is better")
    ax.grid(alpha=.3); ax.legend()

    # 2. Latency vs the hard client timeout.
    ax = axes[1]
    means = [r["latency_mean_s"] for r in rows]
    ax.errorbar(s, means, yerr=[r["latency_ci95_s"] for r in rows],
                fmt="o-", lw=2, capsize=4, color="#2b6cb0", label="mean ±95% CI")
    ax.plot(s, [r["latency_p95_s"] for r in rows], "s--", color="#dd6b20", label="p95")
    ax.axhline(LLM_CLIENT_TIMEOUT_S, color="#c53030", ls="-", lw=2,
               label=f"client timeout ({LLM_CLIENT_TIMEOUT_S}s) — run CRASHES above")
    ax.axhline(LATENCY_SAFETY * LLM_CLIENT_TIMEOUT_S, color="#c53030", ls=":", lw=1.5,
               label=f"safety budget ({LATENCY_SAFETY:.0%})")
    ax.axvline(pick["slots"], ls="--", c="#38a169", lw=2)
    ax.set(xlabel="slots (-np)", ylabel="latency per request (s)",
           title="Latency — must stay under the timeout")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # 3. Scaling efficiency — where batching stops paying for itself.
    ax = axes[2]
    base = next((r["throughput_rps"] for r in rows if r["slots"] == 1), rows[0]["throughput_rps"])
    base_slots = next((r["slots"] for r in rows if r["slots"] == 1), rows[0]["slots"])
    eff = [(r["throughput_rps"] / base) / (r["slots"] / base_slots) * 100 for r in rows]
    ax.plot(s, eff, "o-", lw=2, color="#805ad5")
    ax.axhline(100, color="grey", ls=":", label="ideal linear scaling")
    ax.axvline(pick["slots"], ls="--", c="#38a169", lw=2)
    ax.set(xlabel="slots (-np)", ylabel="scaling efficiency (% of linear)",
           title="Diminishing returns")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Sweep llama.cpp -np to find the throughput-efficient slot count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), help="model from the MODELS registry")
    ap.add_argument("--model-path", help="path to any GGUF (instead of --model)")
    ap.add_argument("--label", help="folder name under results/ (required with --model-path)")
    ap.add_argument("--slots", help="comma-separated slot ladder, e.g. 1,2,4,8,16")
    ap.add_argument("--scenario", default="baseline",
                    help="scenario name (validated against the loaded scenario set)")
    ap.add_argument("--scenario-file", default=None,
                    help="python module defining CONTEXT_SCENARIOS (e.g. scenarios_a3.py); "
                         "measures the prompts production actually sends")
    ap.add_argument("--llm-style", default=None,
                    help="completions | completions+grammar | chat | chat+grammar; "
                         "omit for the historical raw-completions payload")
    ap.add_argument("--port", type=int, default=8090, help="port for the sweep's own server")
    ap.add_argument("--ngl", default="-1", help="GPU layers (-1 = all)")
    ap.add_argument("--llama-server-bin",
                    default=os.environ.get("LLAMA_SERVER_BIN", "llama-server"))
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--max-n", type=int, default=MAX_N)
    ap.add_argument("--quick", action="store_true",
                    help="min-n=20, max-n=50 — a rough shape, NOT a trustworthy p95")
    ap.add_argument("--force", action="store_true",
                    help="run even if another llama-server holds the GPU (results will be junk)")
    args = ap.parse_args()
    _install_cleanup()

    if args.model:
        spec = MODELS[args.model]
        label, model_path, slots = args.model, Path(spec["path"]), spec["slots"]
    elif args.model_path:
        if not args.label:
            ap.error("--label is required with --model-path")
        label, model_path = args.label, Path(args.model_path)
        slots = [1, 2, 4, 8, 16]
    else:
        ap.error("pass --model or --model-path")

    if args.slots:
        slots = [int(x) for x in args.slots.split(",") if x.strip()]
    slots = sorted(set(slots))

    if not model_path.exists():
        ap.error(f"model not found: {model_path}")

    min_n, max_n = (20, 50) if args.quick else (args.min_n, args.max_n)

    busy = gpu_busy()
    if busy and not args.force:
        print("REFUSING TO RUN — another llama-server is on the GPU:", file=sys.stderr)
        for line in busy:
            print(f"    {line}", file=sys.stderr)
        print("\nIts memory-bandwidth use would contaminate every measurement here.\n"
              "Wait for it to finish (or kill it), then re-run. --force overrides.",
              file=sys.stderr)
        return 1

    outdir = Path(__file__).parent / "results" / label
    outdir.mkdir(parents=True, exist_ok=True)

    llm_style = resolve_llm_style(args.llm_style)
    scenario_source = "context_scenarios.py (default)"
    if args.scenario_file:
        scenario_source = apply_scenario_file(args.scenario_file)
    if args.scenario not in CONTEXT_SCENARIOS:
        sys.exit(f"unknown scenario '{args.scenario}'. Available: {', '.join(sorted(CONTEXT_SCENARIOS))}")

    print(f"\n{'=' * 72}\nSLOT SWEEP — {label}\n{'=' * 72}")
    print(f"model     : {model_path}")
    print(f"slots     : {slots}")
    print(f"workload  : '{args.scenario}' prompt, style={llm_style or 'legacy-raw'}, "
          f"max_tokens={MAX_TOKENS}, temp={TEMPERATURE}")
    print(f"prompts   : {scenario_source}")
    print(f"sampling  : adaptive, n in [{min_n}, {max_n}], stop at 95% CI <= {REL_MOE:.0%} of mean")
    print(f"results   : {outdir}\n")

    prompts = build_prompts(max(max_n + max(slots) + 10, 200), args.scenario)

    rows = []
    for s in slots:
        print(f"[-np {s:>2}] launching ...", flush=True)
        proc, log, ctx = launch_server(args.llama_server_bin, model_path, s, args.port,
                                       args.ngl, outdir / f"server_np{s}.log")
        try:
            t0 = time.monotonic()
            wait_ready(args.port, proc)
            verify_slots(args.port, s)
            print(f"    ready in {time.monotonic() - t0:.0f}s (-c {ctx}); benchmarking ...",
                  flush=True)
            r = bench(args.port, label, s, prompts, min_n, max_n, llm_style)
        except Exception as e:
            print(f"    FAILED: {e}")
            stop_server(proc, log)
            continue
        stop_server(proc, log)

        rows.append(r)
        flag = "  [hit max-n: CI wider than target]" if r["hit_max_n"] else ""
        print(f"    n={r['n']:<4} {r['throughput_rps']:.3f} req/s | "
              f"mean {r['latency_mean_s']:.2f}±{r['latency_ci95_s']:.2f}s | "
              f"p95 {r['latency_p95_s']:.2f}s | errors {r['errors']}{flag}\n", flush=True)

    if not rows:
        print("no slot count completed — see the server logs in", outdir, file=sys.stderr)
        return 1

    pick, note = recommend(rows)

    # CSV
    cols = list(rows[0])
    csv_path = outdir / f"sweep_{label}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    png_path = outdir / f"sweep_{label}.png"
    plot(rows, label, pick, png_path)

    payload = {
        "label": label,
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario": args.scenario,
        "scenario_file": args.scenario_file,
        "llm_style": llm_style,
        "workload": {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                     "client_timeout_s": LLM_CLIENT_TIMEOUT_S, "ctx_per_slot": CTX_PER_SLOT},
        "sampling": {"min_n": min_n, "max_n": max_n, "rel_moe": REL_MOE},
        "recommended_processes": pick["slots"],
        "recommendation_note": note,
        "results": rows,
    }
    (outdir / f"sweep_{label}.json").write_text(json.dumps(payload, indent=2))

    base = next((r for r in rows if r["slots"] == 1), None)
    print("=" * 72)
    print(f"RECOMMENDED  processes: {pick['slots']}")
    print("=" * 72)
    print(f"  {note}")
    print(f"  throughput : {pick['throughput_rps']:.3f} req/s "
          f"({pick['tokens_per_s']:.1f} tok/s)"
          + (f"  = {pick['throughput_rps'] / base['throughput_rps']:.1f}x the 1-slot baseline"
             if base else ""))
    print(f"  latency    : mean {pick['latency_mean_s']:.2f}s, p95 {pick['latency_p95_s']:.2f}s "
          f"(timeout {LLM_CLIENT_TIMEOUT_S}s)")
    print(f"\n  Pin it in the run YAML — the run_*.sh launcher sets the server's -np from it:")
    print(f"      contexts_args:\n        processes: {pick['slots']}")
    print(f"\n  plot : {png_path}\n  data : {csv_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
