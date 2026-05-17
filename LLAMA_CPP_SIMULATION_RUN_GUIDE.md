# llama.cpp Simulation Run Guide

Serve a GGUF with `llama-cpp-python`'s server and run the Schelling pipeline
against it. Three stages: token-probs (skipped), live simulation, analysis.
Scale is chosen with `--config-profile`:

- `smoke_test` — 5 runs × 200 steps × baseline. Run this first.
- `production` — 100 runs × 1000 steps × all scenarios.

## 1. Install

```bash
git clone https://github.com/clipo/PancsVriend.git
cd PancsVriend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install "llama-cpp-python[server]"
```

For GPU speed, install a CUDA build of `llama-cpp-python` per the
[project docs](https://github.com/abetlen/llama-cpp-python#installation).
CPU-only works (slower) and is fine for `smoke_test`.

## 2. Start the server

Edit **one line** in `configs/llama_cpp_server.yaml` — set `model:` to your
GGUF's absolute path. Then start it (leave it running):

```bash
python -m llama_cpp.server --config_file configs/llama_cpp_server.yaml
```

Ready when it prints `Uvicorn running on http://0.0.0.0:8080`. Speed settings
(`n_gpu_layers: -1`, `flash_attn: true`, `n_ctx: 2048`) are pre-set in that
file. If it fails to load with a CUDA OOM, lower `n_gpu_layers` from `-1` to a
positive number (e.g. `28`) until it loads.

## 3. Run the pipeline

In `configs/llama_cpp_simulation_run.yaml`, set top-level `llm_model:` to a
label for this GGUF (e.g. `gemma-3-4b-it-q4`) — it names the output folders.
Nothing else needs editing.

```bash
# Smoke test
python run_llm_probability_simulation_analysis.py \
  --config-yaml configs/llama_cpp_simulation_run.yaml \
  --config-profile smoke_test

# Production (long — run under screen/tmux)
screen -S schelling
python run_llm_probability_simulation_analysis.py \
  --config-yaml configs/llama_cpp_simulation_run.yaml \
  --config-profile production
```

## 4. Output

A timestamped run dir under `experiments_with_llama_cpp/`:

```
experiments_with_llama_cpp/run_<ts>_<model>/
├── run_config_effective.yaml      # resolved config
├── experiments/llm_<scenario>_<ts>/
│   ├── metrics_history.csv         # one row per (run, step)
│   ├── convergence_summary.csv
│   ├── move_logs/  states/
├── analysis/                       # ANOVA, rankings, combined metrics
└── plots/                          # segregation plots (PNG)
```

Monitor a running job:

```bash
wc -l experiments_with_llama_cpp/run_*/experiments/*/metrics_history.csv
tail -f experiments_with_llama_cpp/run_*/experiments/*/convergence_summary.csv
nvidia-smi -l 5     # if on GPU
```

## Notes

- The server is **single-stream** (one request at a time). `processes` is
  pinned to `1` in both profiles — raising it only queues. For faster
  production, run several servers on different ports behind a round-robin
  proxy, point `llm_url` at the proxy, and raise `processes`.
- Default `llm_url` is `http://localhost:8080/v1/chat/completions`. Change it
  in the YAML only if you moved the server's port.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` to `localhost:8080` | Server not up yet — wait for `Uvicorn running…`, or check the port matches `llm_url` |
| Stage 2: `log probs summary not found` | `contexts_args.use_log_probs` must be `false` (it already is unless you edited it) |
| Stage 3: `Missing optional dependency 'pyarrow'` | `pip install -r requirements.txt` |
| Slow on CPU | Use a CUDA build of `llama-cpp-python` (server config already requests full GPU offload) |
| Windows: `UnicodeEncodeError … '✅'` | `set PYTHONIOENCODING=utf-8` (cmd) / `$env:PYTHONIOENCODING="utf-8"` (PowerShell) before running |
| Windows: `Failed to load shared library 'llama.dll'` | Copy `cudart64_*`, `cublas64_*`, `cublasLt64_*` from a CUDA `torch`'s `torch/lib` into `llama_cpp/lib/` (major version must match). Not applicable on Linux. |
