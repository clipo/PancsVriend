"""One place for HOW a model is served: its gguf, server flags, and the pinned
clock (2026-09-19).

llama.cpp injects the server's current date into every chat-template render
(`common_chat_extra_context`), and two of our templates print it — Llama-3.3
("Today Date: …", falling back to the literal "26 Jul 2024" only when no date
is supplied) and Mistral-Small-4 ("The current date is …", no fallback). So
those prompts changed every calendar day, and with them every P(MOVE). See
prompt_refinement/LLAMA_CPP_SERVING_NOTES.md §6.

The fix is to freeze the server process's realtime clock with libfaketime.
`PinnedServer` starts llama-server with LD_PRELOAD/FAKETIME set, waits for
/health, renders a probe message through /apply-template and REFUSES to hand
back a URL unless the rendered prompt carries the pinned date (for the two
templates that print one). The monotonic clock is left real so the server's
timers work. Every campaign — extraction, census, escalation — goes through
this so the exact table and the sampling that certifies it always see the
same prompt. Nothing here needs a shell wrapper.

    from value_functions.serving import MODELS, PinnedServer
    with PinnedServer("llama") as srv:
        ...  srv.chat_url, srv.base_url, srv.rendered_probe
"""
from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from value_functions.paths import REPO_ROOT

SERVER_BIN = Path(os.environ.get("LLAMA_SERVER_BIN", "/srv/shared/schelling/llama.cpp/build/bin/llama-server"))
FAKETIME_LIB = Path(os.environ.get("FAKETIME_LIB", "/srv/shared/schelling/tools/libfaketime/src/libfaketime-time64.so.1"))
COMMON_FLAGS = ["-ngl", "-1", "-fa", "on", "-np", "4", "-c", "8192", "--host", "127.0.0.1"]
LOG_DIR = REPO_ROOT / "logs"

# Canonical pins (user decisions 2026-09-12). Date-free templates get the
# llama pin for uniformity; it has no effect on them (verified bit-identical).
PIN_LLAMA = "2024-07-26 12:00:00"      # the Llama-3.3 template's own fallback, "26 Jul 2024"
PIN_MISTRAL = "2026-03-16 12:00:00"    # Mistral Small 4's release date; template has no fallback


@dataclass(frozen=True)
class Model:
    key: str
    label: str                 # the value-function label, e.g. llama-3.3-70b-chat-grammar
    gguf: str                  # repo-relative; first shard for split files
    server_args: tuple
    pin: str = PIN_LLAMA
    pin_check: str | None = None   # substring the rendered prompt MUST contain, if the template prints a date
    sim_model: str = ""        # llm_model slug used by the simulation pipeline


MODELS: dict[str, Model] = {m.key: m for m in [
    Model("qwen", "qwen3.6-27b-chat-grammar", "llms/Qwen3.6-27B-Q5_K_M.gguf", ("--jinja", "--reasoning", "off"), sim_model="qwen3.6-27b"),
    Model("gemma", "gemma-4-31b-chat-grammar", "llms/gemma-4-31B-it-Q5_K_M.gguf", ("--jinja", "--reasoning", "off"), sim_model="gemma-4-31b"),
    Model("phi4", "phi-4-14b-chat-grammar", "llms/phi-4-Q8_0.gguf", ("--jinja",), sim_model="phi-4-14b"),
    Model("granite", "granite-4.2-30b-chat-grammar", "llms/granite-4.2-30b-Q8_0.gguf", ("--jinja", "--reasoning", "off"), sim_model="granite-4.2-30b"),
    Model("olmo2", "olmo-2-32b-chat-grammar", "llms/OLMo-2-0325-32B-Instruct-Q8_0.gguf", ("--jinja",), sim_model="olmo-2-32b"),
    Model("hermes", "hermes-4.3-36b-chat-grammar", "llms/hermes-4_3_36b-Q8_0.gguf", ("--jinja", "--reasoning", "off"), sim_model="hermes-4.3-36b"),
    Model("llama", "llama-3.3-70b-chat-grammar", "llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf", ("--jinja", "--reasoning", "off"),
          pin=PIN_LLAMA, pin_check="26 Jul 2024", sim_model="llama-3.3-70b"),
    Model("deepseek", "deepseek-v4-flash-chat-grammar", "llms/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf", ("--jinja", "--reasoning", "off"), sim_model="deepseek-v4-flash"),
    Model("mistral", "mistral-small-4-119b-chat-grammar", "llms/Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf", ("--jinja", "--reasoning", "off"),
          pin=PIN_MISTRAL, pin_check="2026-03-16", sim_model="mistral-small-4-119b"),
]}


def by_label(label: str) -> Model:
    for m in MODELS.values():
        if m.label == label:
            return m
    raise KeyError(label)


def weights_sha256(gguf: Path, chunk: int = 1 << 24) -> str:
    """sha256 of the served file through the page cache — the bytes mmap hands the model."""
    h = hashlib.sha256()
    with open(gguf, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class PinnedServer:
    """Context manager: llama-server for `model` under its pinned clock.

    Raises RuntimeError if the port is already served, the server dies, or the
    rendered prompt does not carry the pin the template is expected to print.
    """

    def __init__(self, model: Model | str, port: int = 8085, log_tag: str = "", startup_timeout_s: int = 2400,
                 pin: str | None = None):
        self.m = MODELS[model] if isinstance(model, str) else model
        self.port = port
        self.pin = pin or self.m.pin
        self.log_tag = log_tag or self.m.key
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None
        self.rendered_probe = ""
        self.base_url = f"http://127.0.0.1:{port}"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.log_path = LOG_DIR / f"server_{re.sub(r'[^A-Za-z0-9._-]', '_', self.log_tag)}.log"

    def _answers(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    def __enter__(self) -> "PinnedServer":
        if not SERVER_BIN.exists():
            raise RuntimeError(f"llama-server not found at {SERVER_BIN}")
        if not FAKETIME_LIB.exists():
            raise RuntimeError(f"libfaketime not found at {FAKETIME_LIB}; build it (see LLAMA_CPP_SERVING_NOTES.md §6)")
        gguf = REPO_ROOT / self.m.gguf
        if not gguf.exists():
            raise RuntimeError(f"missing {gguf}")
        if self._answers():
            raise RuntimeError(f"something already answers on :{self.port}; refusing to reuse it")
        env = dict(os.environ, LD_PRELOAD=str(FAKETIME_LIB), FAKETIME=self.pin, FAKETIME_DONT_FAKE_MONOTONIC="1")
        cmd = [str(SERVER_BIN), "-m", str(gguf), *COMMON_FLAGS, "--port", str(self.port), *self.m.server_args]
        LOG_DIR.mkdir(exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT,
                                     cwd=str(REPO_ROOT), env=env, start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.startup_timeout_s:
            if self._answers():
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during load (see {self.log_path})")
            time.sleep(3)
        else:
            self.stop()
            raise RuntimeError(f"server did not come up in {self.startup_timeout_s} s (see {self.log_path})")
        r = requests.post(f"{self.base_url}/apply-template", json={"messages": [{"role": "user", "content": "probe"}]}, timeout=60)
        r.raise_for_status()
        self.rendered_probe = r.json().get("prompt", "")
        if self.m.pin_check and self.m.pin_check not in self.rendered_probe:
            self.stop()
            raise RuntimeError(f"{self.m.label}: rendered prompt does not carry the pinned date "
                               f"{self.m.pin_check!r}: {self.rendered_probe[:200]!r}")
        return self

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)          # by PID, never by name
            for _ in range(30):
                if self.proc.poll() is not None:
                    break
                time.sleep(1)
            else:
                self.proc.kill()
        self.proc = None
        for _ in range(15):                                 # port free before the next tenant
            if not self._answers():
                break
            time.sleep(2)

    def __exit__(self, *exc) -> None:
        self.stop()
