#!/usr/bin/env python3
"""Run the batch-numerics probe across the sampled models, one llama-server at a time.

    python value_functions/batch_numerics/run_batch_numerics_study.py                  # all nine
    python value_functions/batch_numerics/run_batch_numerics_study.py qwen3.6-27b-chat-grammar gemma-4-31b-chat-grammar
    python value_functions/batch_numerics/run_batch_numerics_study.py --wait           # first wait for any
                                                          # extraction chain / campaign to free the port
    python value_functions/batch_numerics/run_batch_numerics_study.py --dry-run        # print the plan only

For each model: start llama-server with the flags its sampling campaign used
(-ngl -1 -fa on -np 4 -c 8192 plus the model's chat flags), wait for /health,
run batch_numerics_probe.py against the chat endpoint (sequential vs concurrent
vs interleaved probabilities, plus sequential sampling), stop the server.
Results: value_functions/batch_numerics/results/probe_<label>{.json,.csv,_summary.csv,.png}.

This is the Python port of the one-time shell driver (run_batch_numerics_study.sh,
retired 2026-09-07). The study itself completed on 2026-09-07 for all nine
models; the driver is kept so the measurement can be reproduced end to end from
Python. It is not part of any pipeline.

Server handling: exactly one server, on --port; refuses to start if the port
already answers /health; SIGTERM then SIGKILL by PID when done, never by name.
Server output goes to logs/server_bn_<label>.log, the study log to
logs/batch_numerics_study_<timestamp>.log. ntfy pings (./ntfy.sh) are best-effort.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

_THIS = Path(__file__).resolve().parent
REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from value_functions.paths import add_import_paths  # noqa: E402
add_import_paths()

PROBE = _THIS / "batch_numerics_probe.py"
RESULTS_DIR = _THIS / "results"
LOG_DIR = REPO_ROOT / "logs"
DEFAULT_SERVER_BIN = os.environ.get(
    "LLAMA_SERVER_BIN", "/srv/shared/schelling/llama.cpp/build/bin/llama-server")
COMMON_SERVER_FLAGS = ["-ngl", "-1", "-fa", "on", "-np", "4", "-c", "8192", "--host", "127.0.0.1"]

# (label, gguf relative to the repo, extra server flags) — the sampling
# campaign's serving configuration for each model, so the probe measures the
# server the tables were sampled from.
MODELS = [
    ("qwen3.6-27b-chat-grammar", "llms/Qwen3.6-27B-Q5_K_M.gguf", ["--jinja", "--reasoning", "off"]),
    ("gemma-4-31b-chat-grammar", "llms/gemma-4-31B-it-Q5_K_M.gguf", ["--jinja", "--reasoning", "off"]),
    ("phi-4-14b-chat-grammar", "llms/phi-4-Q8_0.gguf", ["--jinja"]),
    ("granite-4.2-30b-chat-grammar", "llms/granite-4.2-30b-Q8_0.gguf", ["--jinja", "--reasoning", "off"]),
    ("olmo-2-32b-chat-grammar", "llms/OLMo-2-0325-32B-Instruct-Q8_0.gguf", ["--jinja"]),
    ("hermes-4.3-36b-chat-grammar", "llms/hermes-4_3_36b-Q8_0.gguf", ["--jinja", "--reasoning", "off"]),
    ("llama-3.3-70b-chat-grammar", "llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf", ["--jinja", "--reasoning", "off"]),
    ("deepseek-v4-flash-chat-grammar", "llms/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
     ["--jinja", "--reasoning", "off"]),
    ("mistral-small-4-119b-chat-grammar", "llms/Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf",
     ["--jinja", "--reasoning", "off"]),
]
# Other launchers that own the port; --wait blocks while any of them runs.
OTHER_LAUNCHERS = ("run_logprob_resume_chain.sh", "run_logprob_vf_campaign.sh",
                   "run_sanity_vf_chain.sh", "logprob_value_function.py")
SERVER_STARTUP_TIMEOUT_S = 2400      # the 119B model takes ~50 min to load


class Study:
    def __init__(self, args):
        self.args = args
        LOG_DIR.mkdir(exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"batch_numerics_study_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.base_url = f"http://127.0.0.1:{args.port}"

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

    # -- server ------------------------------------------------------------
    def health_ok(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.ok and '"ok"' in r.text
        except requests.RequestException:
            return False

    def port_answers(self):
        try:
            requests.get(f"{self.base_url}/health", timeout=3)
            return True
        except requests.RequestException:
            return False

    def start_server(self, label, gguf, extra):
        slog = LOG_DIR / f"server_bn_{re.sub(r'[^A-Za-z0-9._-]', '_', label)}.log"
        cmd = [self.args.server_bin, "-m", str(gguf), *COMMON_SERVER_FLAGS,
               "--port", str(self.args.port), *extra]
        self.say(f"{label}: starting server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=open(slog, "w"), stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT), start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < SERVER_STARTUP_TIMEOUT_S:
            if self.health_ok():
                self.say(f"{label}: server up after {int(time.monotonic() - t0)} s")
                return proc
            if proc.poll() is not None:
                break
            time.sleep(5)
        self.say(f"{label}: server did not come up (see {slog})")
        self.stop_server(proc)
        return None

    @staticmethod
    def stop_server(proc):
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGTERM)          # by PID, never by name
        for _ in range(30):
            if proc.poll() is not None:
                return
            time.sleep(1)
        proc.kill()
        proc.wait(timeout=10)

    # -- one model ---------------------------------------------------------
    def run_probe(self, label):
        cmd = [sys.executable, str(PROBE), "--label", label, "--url", self.base_url,
               "--seq-samples", str(self.args.seq_samples)]
        if self.args.probe_args:
            cmd.extend(self.args.probe_args)
        self.say(f"{label}: {' '.join(cmd)}")
        with open(self.log_path, "a") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode
        return rc == 0

    def run_model(self, label, gguf_rel, extra):
        gguf = REPO_ROOT / gguf_rel
        if not gguf.exists():
            self.say(f"{label}: missing {gguf_rel} — skipped")
            return False
        if self.args.dry_run:
            self.say(f"{label}: would start {gguf_rel} {' '.join(extra)} and probe it")
            return True
        proc = self.start_server(label, gguf, extra)
        if proc is None:
            return False
        t1 = time.monotonic()
        try:
            ok = self.run_probe(label)
        finally:
            self.stop_server(proc)
            time.sleep(5)
        minutes = int((time.monotonic() - t1 + 59) // 60)
        if ok:
            self.say(f"{label}: probe done in {minutes} min")
            summary = RESULTS_DIR / f"probe_{label}.json"
            body = summary.read_text()[:300].replace("\n", "") if summary.exists() else "done"
            self.ntfy(f"batch-numerics: {label} done", body, "white_check_mark", "low")
        else:
            self.say(f"{label}: probe FAILED (see {self.log_path})")
            self.ntfy(f"batch-numerics: {label} FAILED", f"see {self.log_path}", "warning", "high")
        return ok

    # -- whole study -------------------------------------------------------
    def wait_for_port(self):
        self.say("waiting for extraction chain / campaign to finish")
        while True:
            ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
            busy = any(name in ps for name in OTHER_LAUNCHERS) or self.port_answers()
            if not busy:
                return
            time.sleep(120)

    def run(self):
        wanted = set(self.args.labels)
        unknown = wanted - {m[0] for m in MODELS}
        if unknown:
            self.say(f"unknown label(s): {sorted(unknown)}; known: {[m[0] for m in MODELS]}")
            return 2
        plan = [m for m in MODELS if not wanted or m[0] in wanted]
        if self.args.wait and not self.args.dry_run:
            self.wait_for_port()
        if not self.args.dry_run and self.port_answers():
            self.say(f"port {self.args.port} busy — stop that server first")
            return 2
        self.ntfy("batch-numerics study STARTED", f"{len(plan)} model(s), chat endpoint", "rocket")
        done, failed = [], []
        for label, gguf, extra in plan:
            (done if self.run_model(label, gguf, extra) else failed).append(label)
        self.say(f"study done — done: {done or 'none'} | failed: {failed or 'none'}")
        self.ntfy("batch-numerics study FINISHED", f"done: {done}; failed: {failed}", "tada")
        return 0 if not failed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels", nargs="*", help="model labels to probe (default: all nine)")
    ap.add_argument("--wait", action="store_true",
                    help="block until no extraction chain / campaign runs and the port is free")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8085)))
    ap.add_argument("--seq-samples", type=int, default=int(os.environ.get("SEQ_SAMPLES", 300)),
                    help="sequential samples per probed cell (0 = skip)")
    ap.add_argument("--server-bin", default=DEFAULT_SERVER_BIN,
                    help="llama-server binary (env LLAMA_SERVER_BIN)")
    ap.add_argument("--probe-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this flag is passed to batch_numerics_probe.py")
    ap.add_argument("--no-ntfy", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; start nothing")
    args = ap.parse_args(argv)
    return Study(args).run()


if __name__ == "__main__":
    sys.exit(main())
