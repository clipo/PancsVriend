#!/usr/bin/env python
"""
gpu_autoslots.py — size the llama.cpp server's `-np` slot count to (a) the run
config's `processes` and (b) what actually fits in GPU VRAM, measured empirically.

WHY: the number of continuous-batching slots (`-np`) and the client concurrency
(`processes` in the run YAML) must match — extra client requests beyond the slot
count just queue, extra slots beyond the client concurrency waste KV cache. This
helper makes `processes` the single source of truth and derives `-np` from it,
then caps `-np` so the per-slot KV cache never overflows VRAM.

MECHANISM (empirical calibration — arch-proof, no metadata guessing)
  1. Read `processes` from the run YAML profile              [desired concurrency]
  2. Read free VRAM from nvidia-smi                           [before touching GPU]
  3. PROBE: launch `llama-server -np 1 -c ctx_per_slot` once and parse its own
     startup log for the EXACT allocations it reports:
        weights      = sum of "CUDAn model buffer size"
        kv_per_slot  = sum of "CUDAn KV buffer size"   (at ctx_per_slot, -np 1)
        compute      = sum of "CUDAn compute buffer size"
     then kill the probe. These are llama.cpp's real numbers, so SWA / GQA /
     odd head dims are all accounted for — no architecture formula involved.
  4. max_slots = floor((free - weights - compute - reserve) / kv_per_slot)
     NP  = clamp(min(processes, max_slots), 1, processes)
     CTX = NP * ctx_per_slot        (llama.cpp -c is TOTAL context, split evenly)

Emits shell-eval assignments on stdout, a human breakdown on stderr:

    eval "$(python gpu_autoslots.py --config configs/llama_cpp_run_gemma31b.yaml \
              --profile production --model "$MODEL_PATH" --ngl "$NGL" \
              --ctx-per-slot 2048 --llama-server-bin "$LLAMA_SERVER_BIN")"
    llama-server -m "$MODEL_PATH" -np "$NP" -c "$CTX" ...

Fail-open: if the probe can't run (no binary, timeout) it falls back to a
conservative GGUF-metadata estimate; if that also fails it uses `processes`
uncapped and warns. A launch is never silently blocked.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time

KV_BYTES_F16 = 2


def _warn(msg: str) -> None:
    print(f"[gpu_autoslots] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# 1. Desired concurrency — single source of truth (run YAML `processes`)
# --------------------------------------------------------------------------- #
def read_processes(config_path: str, profile: str | None) -> int:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _procs(block):
        if isinstance(block, dict):
            ca = block.get("contexts_args", block)
            if isinstance(ca, dict) and ca.get("processes") is not None:
                return int(ca["processes"])
        return None

    if profile:
        p = _procs((cfg.get("profiles") or {}).get(profile))
        if p is not None:
            return p
        _warn(f"profile '{profile}' has no processes; falling back to top-level.")
    p = _procs(cfg)
    if p is None:
        raise ValueError(f"no `processes` found in {config_path} (profile={profile})")
    return p


# --------------------------------------------------------------------------- #
# 2. Free VRAM (nvidia-smi CLI)
# --------------------------------------------------------------------------- #
def gpu_free_mib(gpu_index: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            stderr=subprocess.STDOUT, text=True, timeout=15,
        )
        return int(out.strip().splitlines()[0])
    except Exception as exc:
        _warn(f"could not read free VRAM via nvidia-smi ({exc}).")
        return None


# --------------------------------------------------------------------------- #
# 3a. Empirical probe — launch -np 1 and read llama.cpp's own buffer report
# --------------------------------------------------------------------------- #
_SUM_PATTERNS = {
    "weights": re.compile(r"CUDA\d+ model buffer size\s*=\s*([\d.]+)"),
    "kv": re.compile(r"CUDA\d+ KV buffer size\s*=\s*([\d.]+)"),
    "compute": re.compile(r"CUDA\d+ compute buffer size\s*=\s*([\d.]+)"),
}
_CPU_SPILL = re.compile(r"CPU_Mapped model buffer size\s*=\s*([\d.]+)")
_READY = re.compile(r"compute buffer size|server is listening|model loaded")


def probe_buffers(bin_path, model, ctx_per_slot, ngl, use_fa, port, timeout):
    """Launch a 1-slot server, capture reported CUDA buffer sizes, kill it.
    Returns dict(weights, kv, compute, cpu_spill) in MiB, or None on failure."""
    cmd = [bin_path, "-m", model, "--alias", "autoslots-probe",
           "-ngl", str(ngl), "-np", "1", "-c", str(ctx_per_slot),
           "--host", "127.0.0.1", "--port", str(port)]
    if use_fa:
        cmd += ["-fa", "on"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError:
        _warn(f"probe binary not found: {bin_path}")
        return None

    lines, saw_compute = [], False
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            lines.append(line)
            if _SUM_PATTERNS["compute"].search(line):
                saw_compute = True
            # Once the compute buffer is reported, allocations are done — the
            # KV/weights lines came before it. Give one beat, then stop early
            # (avoids waiting through warmup, which matters for big models).
            if saw_compute and ("server is listening" in line or "model loaded" in line):
                break
        else:
            _warn(f"probe timed out after {timeout}s.")
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=20)
        except Exception:
            proc.kill()

    text = "".join(lines)

    def _sum(pat):
        vals = [float(x) for x in pat.findall(text)]
        return sum(vals) if vals else None

    weights = _sum(_SUM_PATTERNS["weights"])
    kv = _sum(_SUM_PATTERNS["kv"])
    compute = _sum(_SUM_PATTERNS["compute"])
    spill = _sum(_CPU_SPILL)
    if weights is None or kv is None:
        _warn("probe ran but did not report CUDA model/KV buffer sizes "
              "(CPU-only build, or unrecognized log format).")
        return None
    return {"weights": weights, "kv": kv, "compute": compute or 0.0,
            "cpu_spill": spill or 0.0}


# --------------------------------------------------------------------------- #
# 3b. Fallback — conservative GGUF-metadata KV estimate (upper bound)
# --------------------------------------------------------------------------- #
def kv_mib_per_slot_estimate(gguf_path, ctx_per_slot, kv_bytes):
    try:
        import gguf
        r = gguf.GGUFReader(gguf_path)

        def field(key):
            f = r.fields.get(key)
            if f is None:
                return None
            v = f.contents() if hasattr(f, "contents") else None
            return max(v) if isinstance(v, (list, tuple)) else v

        arch = field("general.architecture")
        n_layer = field(f"{arch}.block_count")
        n_kv = field(f"{arch}.attention.head_count_kv")
        n_head = field(f"{arch}.attention.head_count")
        k_len = field(f"{arch}.attention.key_length")
        v_len = field(f"{arch}.attention.value_length")
        n_embd = field(f"{arch}.embedding_length")
        if k_len is None or v_len is None:
            k_len = v_len = int(n_embd) // int(n_head)
        if not (n_layer and n_kv and k_len and v_len):
            return None
        kv_tok = int(n_layer) * int(n_kv) * (int(k_len) + int(v_len)) * kv_bytes
        return kv_tok * ctx_per_slot / (1024 * 1024)
    except Exception as exc:
        _warn(f"metadata KV estimate failed ({exc}).")
        return None


# --------------------------------------------------------------------------- #
# 4. Combine
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Cap llama.cpp -np to VRAM, synced to `processes`.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="run YAML to read `processes` from")
    src.add_argument("--processes", type=int, help="use this concurrency directly")
    ap.add_argument("--profile", default="production")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx-per-slot", type=int, default=2048)
    ap.add_argument("--reserve-mib", type=int, default=512,
                    help="extra headroom on top of the measured compute buffer")
    ap.add_argument("--gpu", type=int, default=0)
    # probe controls
    ap.add_argument("--llama-server-bin", default=os.environ.get("LLAMA_SERVER_BIN", "llama-server"))
    ap.add_argument("--ngl", default="-1", help="GPU layers for the probe (match the real launch)")
    ap.add_argument("--no-fa", action="store_true", help="probe without flash-attention")
    ap.add_argument("--probe-port", type=int, default=8099)
    ap.add_argument("--probe-timeout", type=int, default=600)
    ap.add_argument("--no-probe", action="store_true", help="skip probe, use metadata estimate")
    ap.add_argument("--kv-bytes", type=int, default=KV_BYTES_F16)
    args = ap.parse_args(argv)

    processes = args.processes if args.processes is not None \
        else read_processes(args.config, args.profile)

    free_mib = gpu_free_mib(args.gpu)

    # --- measure per-slot KV + weights (empirical, then fallback) ---
    measured, weights_mib, per_slot_mib, compute_mib, source = None, None, None, 0.0, None
    if not args.no_probe:
        _warn(f"probing 1-slot server to measure real VRAM use (ctx={args.ctx_per_slot}) ...")
        measured = probe_buffers(args.llama_server_bin, args.model, args.ctx_per_slot,
                                 args.ngl, not args.no_fa, args.probe_port, args.probe_timeout)
    if measured:
        weights_mib = measured["weights"]
        per_slot_mib = measured["kv"]
        compute_mib = measured["compute"]
        source = "probe"
        if measured["cpu_spill"] > 0:
            _warn(f"WARNING: {measured['cpu_spill']:.0f} MiB of weights spilled to CPU at -ngl "
                  f"{args.ngl} — the model does not fully fit; consider lowering -ngl or the model.")
    else:
        # fallback: metadata KV estimate + file-size weights
        per_slot_mib = kv_mib_per_slot_estimate(args.model, args.ctx_per_slot, args.kv_bytes)
        try:
            weights_mib = os.path.getsize(args.model) / (1024 * 1024)
        except OSError:
            weights_mib = None
        source = "metadata-estimate"
        _warn("using fallback metadata estimate (conservative upper bound).")

    # --- decide slot count ---
    if free_mib is None or per_slot_mib is None or weights_mib is None:
        _warn(f"insufficient info to cap; using processes={processes} uncapped.")
        np_eff, max_slots = processes, None
    else:
        budget = free_mib - weights_mib - compute_mib - args.reserve_mib
        max_slots = int(math.floor(budget / per_slot_mib)) if per_slot_mib > 0 else processes
        np_eff = min(processes, max_slots)
        if np_eff < 1:
            _warn(f"VRAM budget fits <1 slot (budget={budget:.0f} MiB, "
                  f"per-slot={per_slot_mib:.0f} MiB). Forcing 1 — launch may still OOM.")
            np_eff = 1

    ctx = np_eff * args.ctx_per_slot

    _warn("--- slot sizing ---")
    _warn(f"source              : {source}")
    _warn(f"requested processes : {processes}")
    if free_mib is not None:
        _warn(f"free VRAM           : {free_mib} MiB")
    if weights_mib is not None:
        _warn(f"model weights       : {weights_mib:.0f} MiB")
    if compute_mib:
        _warn(f"compute buffer      : {compute_mib:.0f} MiB")
    if per_slot_mib is not None:
        _warn(f"KV per slot @{args.ctx_per_slot:>5}: {per_slot_mib:.0f} MiB")
    if max_slots is not None:
        _warn(f"max slots (VRAM)    : {max_slots}")
    _warn(f"=> NP={np_eff}  CTX={ctx}  (per slot {args.ctx_per_slot})")
    if max_slots is not None and np_eff < processes:
        _warn(f"NOTE: capped {processes} -> {np_eff} to fit VRAM.")

    print(f"NP={np_eff}; CTX={ctx}; CTX_PER_SLOT={args.ctx_per_slot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
