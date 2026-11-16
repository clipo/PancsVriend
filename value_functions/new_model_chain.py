#!/usr/bin/env python3
"""Bring a NEW model (or a new quantisation of one) through the whole
value-function workflow, one step after another, on one machine:

    download the gguf  ->  wait until the GPU/RAM are free  ->  llama-server up
    ->  exact tables (logprob_value_function.py, sequential, validated)
    ->  batch-numerics probe (batch_numerics_probe.py, same server)
    ->  server down  ->  simulation + analysis pipeline
        (run_llm_probability_simulation_analysis.py: contexts, scenario
         analysis, rank stability, cross-model figures)

    python value_functions/new_model_chain.py gemma-q8              # everything
    python value_functions/new_model_chain.py gemma-q8 --dry-run    # print the plan
    python value_functions/new_model_chain.py gemma-q8 --smoke      # 2-run pipeline profile
    python value_functions/new_model_chain.py gemma-q8 --skip-download --skip-probe

Every step is strictly sequential and that is load-bearing (see
run_sanity_vf_chain.sh's header): this is a 121 GB box, a served model can
take 96 GB, and on 2026-09-06 an analysis pass and a model load ran at the
same time and the OOM killer took both. The chain therefore
  * refuses to start GPU work while any other launcher / server is alive
    (OTHER_LAUNCHERS below) or less than MIN_AVAILABLE_GB of RAM is free,
    polling until they are gone (no --wait flag: waiting is the default and
    the only safe behaviour; --no-wait fails instead of waiting);
  * stops its own server (by PID, never by name) before the CPU pipeline;
  * exits on the first failure — a later step with the earlier step's
    output missing would only produce a misleading run folder.

Notifications go through ./ntfy.sh (stage start/finish, failure) and a
low-priority progress line every PROGRESS_EVERY_S while the pipeline runs
(user rule 2026-09-05: anything over an hour reports on ntfy).

Since 2026-09-07 the exact extractor needs no sampled campaign (the sampled
tables are not canonical), so a new model starts here; nothing in
value_functions/results/sampled/ is read or written.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

_THIS = Path(__file__).resolve().parent            # value_functions/
REPO_ROOT = _THIS.parent
sys.path.insert(0, str(REPO_ROOT))
from value_functions.paths import LOGPROB_VALIDATION_DIR, add_import_paths  # noqa: E402
add_import_paths()
from value_functions.batch_numerics.run_batch_numerics_study import (  # noqa: E402
    COMMON_SERVER_FLAGS, DEFAULT_SERVER_BIN, SERVER_STARTUP_TIMEOUT_S)

EXTRACTOR = REPO_ROOT / "value_functions" / "logprob" / "logprob_value_function.py"
PROBE = REPO_ROOT / "value_functions" / "batch_numerics" / "batch_numerics_probe.py"
PROBE_RESULTS = REPO_ROOT / "value_functions" / "batch_numerics" / "results"
PIPELINE = REPO_ROOT / "run_llm_probability_simulation_analysis.py"
LOG_DIR = REPO_ROOT / "logs"

# key -> everything the chain needs. `download` = (url, expected bytes);
# `probe_args` are appended to the probe command (--from-exact ranks the
# probed cells on this label's exact tables; --match-cells adds the cells
# another probe measured so two quantisations are compared cell for cell).
MODELS = {
    "gemma-q8": dict(
        label="gemma-4-31b-q8-chat-grammar",
        gguf="llms/gemma-4-31B-it-Q8_0.gguf",
        download=("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q8_0.gguf",
                  32635677632),
        server_args=["--jinja", "--reasoning", "off"],
        run_yaml="configs/vf_run_gemma_q8_lp.yaml",
        probe_args=["--from-exact", "--match-cells", "gemma-4-31b-chat-grammar"],
    ),
}

# Anything whose presence in `ps` means the GPU or the RAM is spoken for.
OTHER_LAUNCHERS = ("llama-server", "run_sanity_vf_chain.sh", "run_logprob_vf_campaign.sh",
                   "run_logprob_resume_chain.sh", "run_batch_numerics_study.py",
                   "logprob_value_function.py", "batch_numerics_probe.py",
                   "build_value_function.py", "run_llm_probability_simulation_analysis.py",
                   "run_all_contexts.py", "run_vf_prod10k_queue.sh")
MIN_AVAILABLE_GB = 60          # a Q8_0 31B model + KV cache is ~40 GB; leave headroom
POLL_S = 120
PROGRESS_EVERY_S = 3600


class Chain:
    def __init__(self, key, args):
        self.key = key
        self.m = MODELS[key]
        self.args = args
        self.label = self.m["label"]
        self.base_url = f"http://127.0.0.1:{args.port}"
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / f"new_model_chain_{key}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.server = None
        self.t0 = time.monotonic()

    # -- logging / notifications ------------------------------------------
    def say(self, msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        if not self.args.dry_run:
            with open(self.log_path, "a") as fh:
                fh.write(line + "\n")

    def ntfy(self, title, body, tags="test_tube", prio="default"):
        if self.args.dry_run or self.args.no_ntfy:
            return
        script = REPO_ROOT / "ntfy.sh"
        if script.exists():
            subprocess.run([str(script), title, body, tags, prio],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def elapsed_min(self):
        return int((time.monotonic() - self.t0) / 60)

    def run(self, cmd, log_name):
        """Run a subprocess with its output appended to logs/<log_name>; returns rc."""
        self.say(f"$ {' '.join(map(str, cmd))}  >> logs/{log_name}")
        if self.args.dry_run:
            return 0
        with open(LOG_DIR / log_name, "a") as fh:
            fh.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(map(str, cmd))}\n")
            fh.flush()
            return subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(REPO_ROOT)).returncode

    # -- 0. download --------------------------------------------------------
    def download(self):
        gguf = REPO_ROOT / self.m["gguf"]
        url, size = self.m["download"]
        if gguf.exists() and gguf.stat().st_size == size:
            self.say(f"download: {gguf.name} complete ({size:,} bytes)")
            return True
        self.say(f"download: fetching {gguf.name} ({size / 1e9:.1f} GB) from {url}")
        self.ntfy(f"{self.key}: download started", f"{gguf.name}, {size / 1e9:.1f} GB", "arrow_down", "low")
        rc = self.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-",
                       "-o", str(gguf), url], f"download_{self.key}.log")
        if self.args.dry_run:
            return True
        got = gguf.stat().st_size if gguf.exists() else 0
        if rc != 0 or got != size:
            self.say(f"download FAILED (rc {rc}, {got:,} of {size:,} bytes) — re-run to resume")
            self.ntfy(f"{self.key}: download FAILED", f"rc {rc}, {got:,}/{size:,} bytes", "rotating_light", "high")
            return False
        self.say(f"download: complete ({got:,} bytes)")
        return True

    # -- 1. wait for the machine ---------------------------------------------
    @staticmethod
    def available_gb():
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
        return 0.0

    def busy_processes(self):
        # never match ourselves or the shell that launched us (its argv can
        # quote any of these names)
        mine = {os.getpid(), os.getppid()}
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
        hits = []
        for line in out.splitlines()[1:]:
            pid, _, cmd = line.strip().partition(" ")
            if not pid.isdigit() or int(pid) in mine:
                continue
            if any(name in cmd for name in OTHER_LAUNCHERS):
                hits.append(cmd.strip()[:90])
        return hits

    def port_answers(self):
        try:
            requests.get(f"{self.base_url}/health", timeout=3)
            return True
        except requests.RequestException:
            return False

    def wait_for_machine(self):
        announced = False
        while True:
            hits = self.busy_processes()
            avail = self.available_gb()
            free = not hits and not self.port_answers() and avail >= MIN_AVAILABLE_GB
            if free or self.args.dry_run:
                if announced:
                    self.say(f"machine free ({avail:.0f} GB available)")
                return True
            if self.args.no_wait:
                self.say(f"machine busy ({len(hits)} process(es), {avail:.0f} GB available) and --no-wait")
                for h in hits:
                    self.say(f"    {h}")
                return False
            if not announced:
                self.say(f"waiting for the machine: {len(hits)} process(es) hold the GPU/RAM, "
                         f"{avail:.0f} GB available (need {MIN_AVAILABLE_GB})")
                for h in hits:
                    self.say(f"    {h}")
                self.ntfy(f"{self.key}: queued", f"waiting for {hits[0] if hits else 'RAM'} to finish; "
                          f"polling every {POLL_S} s", "hourglass", "low")
                announced = True
            time.sleep(POLL_S)

    # -- 2. server ------------------------------------------------------------
    def start_server(self):
        gguf = REPO_ROOT / self.m["gguf"]
        slog = LOG_DIR / f"server_chain_{re.sub(r'[^A-Za-z0-9._-]', '_', self.label)}.log"
        cmd = [self.args.server_bin, "-m", str(gguf), *COMMON_SERVER_FLAGS,
               "--port", str(self.args.port), *self.m["server_args"]]
        self.say(f"server: {' '.join(cmd)}")
        if self.args.dry_run:
            return True
        if self.port_answers():
            self.say(f"server: something already answers on :{self.args.port} — refusing to reuse it")
            return False
        self.server = subprocess.Popen(cmd, stdout=open(slog, "w"), stderr=subprocess.STDOUT,
                                       cwd=str(REPO_ROOT), start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < SERVER_STARTUP_TIMEOUT_S:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=3)
                if r.ok and '"ok"' in r.text:
                    self.say(f"server: up after {int(time.monotonic() - t0)} s")
                    return True
            except requests.RequestException:
                pass
            if self.server.poll() is not None:
                break
            time.sleep(5)
        self.say(f"server: did not come up (see {slog})")
        self.stop_server()
        return False

    def stop_server(self):
        if self.server is None or self.server.poll() is not None:
            self.server = None
            return
        self.server.send_signal(signal.SIGTERM)          # by PID, never by name
        for _ in range(30):
            if self.server.poll() is not None:
                break
            time.sleep(1)
        else:
            self.server.kill()
            self.server.wait(timeout=10)
        self.server = None
        self.say("server: stopped")
        time.sleep(5)

    # -- 3. exact tables ----------------------------------------------------------
    def extract(self):
        self.say(f"exact tables: extracting {self.label} (sequential)")
        self.ntfy(f"{self.key}: exact tables started", f"{self.label}; sequential path sums + seq-check",
                  "abacus", "low")
        t1 = time.monotonic()
        rc = self.run([sys.executable, EXTRACTOR, "--label", self.label, "--url", self.base_url],
                      f"logprob_vf_{self.key}.log")
        mins = int((time.monotonic() - t1 + 59) // 60)
        vpath = LOGPROB_VALIDATION_DIR / f"validation_{self.label}.json"
        verdict = ""
        if vpath.exists() and not self.args.dry_run:
            v = json.loads(vpath.read_text())
            verdict = (f"seq-check {v.get('seq_check_inside_ci')} inside CI over {v.get('seq_check_cells')} cells, "
                       f"median |diff| {v.get('seq_check_median_abs_diff')}")
        if rc == 0:
            self.say(f"exact tables: VALIDATED in {mins} min ({verdict})")
            self.ntfy(f"{self.key}: exact tables VALIDATED", f"{mins} min; {verdict}", "white_check_mark", "default")
            return True
        if rc == 4:
            self.say(f"exact tables: validation FAILED in {mins} min ({verdict}) — tables written for inspection")
            self.ntfy(f"{self.key}: exact tables validation FAILED", f"{mins} min; {verdict}", "warning", "high")
            return self.args.continue_on_invalid
        self.say(f"exact tables: extractor CRASHED (rc {rc}); see logs/logprob_vf_{self.key}.log")
        self.ntfy(f"{self.key}: extractor CRASHED", f"rc {rc}", "rotating_light", "high")
        return False

    # -- 4. batch-numerics probe -----------------------------------------------------
    def probe(self):
        self.say(f"probe: batch-numerics on {self.label}")
        self.ntfy(f"{self.key}: batch-numerics probe started", "sequential / concurrent / interleaved", "microscope", "low")
        t1 = time.monotonic()
        rc = self.run([sys.executable, PROBE, "--label", self.label, "--url", self.base_url,
                       "--seq-samples", str(self.args.seq_samples), *self.m.get("probe_args", [])],
                      f"batch_numerics_{self.key}.log")
        mins = int((time.monotonic() - t1 + 59) // 60)
        if rc != 0:
            self.say(f"probe: FAILED (rc {rc}); see logs/batch_numerics_{self.key}.log")
            self.ntfy(f"{self.key}: probe FAILED", f"rc {rc}", "warning", "high")
            return False
        body = "done"
        jpath = PROBE_RESULTS / f"probe_{self.label}.json"
        if jpath.exists() and not self.args.dry_run:
            j = json.loads(jpath.read_text())
            body = (f"sequential max range {j.get('sequential_max_range')}, concurrent span median "
                    f"{j.get('concurrent_span_median'):.3f} max {j.get('concurrent_span_max'):.3f} "
                    f"over {j.get('cells')} cells")
        self.say(f"probe: done in {mins} min — {body}")
        self.ntfy(f"{self.key}: probe done", body, "white_check_mark", "low")
        return True

    # -- 5. pipeline ----------------------------------------------------------------
    def pipeline(self):
        y = REPO_ROOT / self.m["run_yaml"]
        profile = "smoke_test" if self.args.smoke else "production"
        self.say(f"pipeline: {y.name} [{profile}] (CPU; no server up)")
        self.ntfy(f"{self.key}: simulations started", f"{y.name} [{profile}]", "rocket", "default")
        stop = threading.Event()
        watcher = threading.Thread(target=self._progress_loop, args=(stop,), daemon=True)
        if not self.args.dry_run:
            watcher.start()
        t1 = time.monotonic()
        try:
            rc = self.run([sys.executable, PIPELINE, "--config-yaml", y, "--config-profile", profile],
                          f"pipeline_{self.key}.log")
        finally:
            stop.set()
        hours = (time.monotonic() - t1) / 3600
        if rc != 0:
            self.say(f"pipeline: FAILED (rc {rc}) after {hours:.1f} h; see logs/pipeline_{self.key}.log")
            self.ntfy(f"{self.key}: pipeline FAILED", f"rc {rc} after {hours:.1f} h", "rotating_light", "high")
            return False
        verdict = self._rank_verdict()
        self.say(f"pipeline: DONE in {hours:.1f} h — rank stability: {verdict}")
        self.ntfy(f"{self.key}: pipeline DONE", f"{hours:.1f} h; DI rank stability: {verdict}; "
                  "cross-model figures refreshed", "white_check_mark", "default")
        return True

    def _run_dir(self):
        slug = self._slug()
        runs = sorted((REPO_ROOT / "experiments_with_llama_cpp").glob(f"run_*_{slug}"))
        return runs[-1] if runs else None

    def _slug(self):
        import yaml
        return yaml.safe_load(open(REPO_ROOT / self.m["run_yaml"]))["llm_model"]

    def _rank_verdict(self):
        run = self._run_dir()
        f = run / "analysis" / "rank_stability" / "rank_status.json" if run else None
        if f and f.exists():
            d = json.loads(f.read_text())
            return f"{d.get('verdict')} {d.get('decision', {}).get('counts')}"
        return "no rank_status.json"

    def _progress_loop(self, stop):
        """One low-priority ntfy line per PROGRESS_EVERY_S while the pipeline runs:
        finished runs / total, elapsed, ETA (rate = done / elapsed)."""
        started = time.time()
        target = 2 if self.args.smoke else None
        while not stop.wait(PROGRESS_EVERY_S):
            run = self._run_dir()
            if run is None:
                continue
            exps = sorted((run / "experiments").glob("llm_*"))
            if not exps:
                continue
            if target is None:
                try:
                    target = int(json.load(open(exps[0] / "config.json")).get("n_runs", 0)) or 10000
                except Exception:
                    target = 10000
            n_scen = 6 if not self.args.smoke else 1
            done = 0
            for e in exps:
                done += len(list((e / "states").glob("states_run_*.npz")))
                packed = e / "states" / "states_packed.npz"
                if packed.exists():
                    import numpy as np
                    with np.load(packed) as z:
                        done += len(z.files)
            total = n_scen * target
            el = time.time() - started
            eta = el * (total - done) / done if done else float("nan")
            body = (f"{done:,}/{total:,} runs ({100 * done / total:.0f}%), scenario {len(exps)}/{n_scen}, "
                    f"{el / 3600:.1f} h elapsed, ETA {eta / 3600:.1f} h")
            self.say(f"progress: {body}")
            self.ntfy(f"{self.key} progress ({datetime.now():%H:%M})", body, "hourglass", "low")

    # -- whole chain ----------------------------------------------------------------
    def go(self):
        a = self.args
        self.say(f"chain {self.key}: label {self.label}, gguf {self.m['gguf']}, yaml {self.m['run_yaml']}"
                 + (" [DRY RUN]" if a.dry_run else ""))
        if not a.skip_download and not self.download():
            return 1
        gpu_steps = [s for s, skip in (("extract", a.skip_extract), ("probe", a.skip_probe)) if not skip]
        if gpu_steps:
            if not self.wait_for_machine():
                return 2
            self.ntfy(f"{self.key}: GPU stage started", f"server loading {Path(self.m['gguf']).name}; "
                      f"then {' + '.join(gpu_steps)}", "rocket", "default")
            if not self.start_server():
                return 3
            try:
                if "extract" in gpu_steps and not self.extract():
                    return 4
                if "probe" in gpu_steps and not self.probe():
                    return 5
            finally:
                self.stop_server()
        if not a.skip_pipeline:
            # the CPU pipeline is 15 processes for hours: the same no-overlap rule applies
            if not self.wait_for_machine():
                return 2
            if not self.pipeline():
                return 6
        self.say(f"chain {self.key}: COMPLETE in {self.elapsed_min()} min")
        self.ntfy(f"{self.key}: chain COMPLETE", f"{self.elapsed_min()} min total; tables, probe, run folder "
                  "and cross-model figures all in place", "tada", "high")
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("key", choices=sorted(MODELS), help="model to bring through the workflow")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8085)))
    ap.add_argument("--server-bin", default=DEFAULT_SERVER_BIN)
    ap.add_argument("--seq-samples", type=int, default=300, help="probe: sequential samples per cell")
    ap.add_argument("--smoke", action="store_true", help="pipeline: smoke_test profile (2 runs x 2 steps)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--continue-on-invalid", action="store_true",
                    help="run the probe and the pipeline even if the exact tables fail their sequential check")
    ap.add_argument("--no-wait", action="store_true",
                    help="fail instead of waiting when another launcher/server holds the machine")
    ap.add_argument("--no-ntfy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if shutil.which(args.server_bin) is None and not Path(args.server_bin).exists():
        print(f"llama-server not found at {args.server_bin}")
        return 2
    return Chain(args.key, args).go()


if __name__ == "__main__":
    sys.exit(main())
