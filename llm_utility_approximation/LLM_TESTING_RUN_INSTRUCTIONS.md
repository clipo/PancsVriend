# Branching vs Sampling Comparison — Run Guide

## What it does

`branching_vs_sampling_comparison.py` compares two ways of estimating the LLM's
P(MOVE) and P(STAY) on Schelling-neighborhood prompts:

1. **Branching estimator** — single deterministic pass that enumerates the
   high-mass token paths and sums their probabilities.
2. **Empirical sampling** — N stochastic generations at the same temperature.

The script reports per-context deltas, Wilson 95% CIs, and aggregate
calibration stats. The branching estimator is correct iff the sample
probabilities fall inside its CI most of the time.

## Prerequisites

- **Free disk space:** ~70 GB (the two GGUFs sum to ~64 GB; leave room for outputs and the venv).
- **Python:** 3.10+ recommended.
- **Hugging Face account** with the Gemma and Llama license terms accepted on each model's page (the unsloth re-hosts inherit the gating — without acceptance, the downloads will fail silently or return an HTML login page).
- **CUDA toolkit + compatible NVIDIA driver** installed; tell me which CUDA major version you have so we use a matching `llama-cpp-python` wheel.

## Setup (Linux + CUDA GPU)

Run all commands below from the **repo root** (`PancsVriend/`). The script uses
relative paths and will not find files if you run it from inside a subdirectory.

```bash
git clone https://github.com/clipo/PancsVriend.git && cd PancsVriend

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CUDA build of llama-cpp-python (the PyPI wheel is CPU-only).
# Easiest: install a prebuilt CUDA wheel from the abetlen index. Pick the
# index that matches your CUDA major version — examples:
#   CUDA 12.4: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
#   CUDA 12.1: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip install --force-reinstall --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    llama-cpp-python==0.3.20

# Fallback (if no prebuilt wheel matches your CUDA version) — build from source.
# Needs cmake, gcc/g++, python3-dev, and matching CUDA toolkit installed:
# CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.20
```

## Models to download

Both are **dense transformers** (no MoE). Accept the license on each Hugging
Face model page first, then either log in with `huggingface-cli login` or pass
your access token via `--header` to `wget`. From the repo root:

```bash
mkdir -p ./llms

# Gemma 4 31B-it — Q5_K_M (~22 GB on disk, dense, 60 layers)
wget -O ./llms/gemma-4-31B-it-Q5_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q5_K_M.gguf

# Llama 3.3 70B Instruct — Q4_K_M (~42 GB on disk, dense, 80 layers, GQA)
wget -O ./llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf

# Verify both files are the expected size (NOT a small HTML login page).
# Gemma should be ~22 GB; Llama should be ~42 GB. If either is < 1 MB,
# the download was redirected to a login page — fix HF auth and retry.
ls -lh ./llms/
```

VRAM rough sizing for full GPU offload:

| Model                  | Quant  | Disk   | VRAM (full offload)                            |
| ---------------------- | ------ | ------ | ---------------------------------------------- |
| Gemma 4 31B-it         | Q5_K_M | ~22 GB | ~24+ GB                                        |
| Llama 3.3 70B Instruct | Q4_K_M | ~42 GB | ~48+ GB (single H100/A100 80 GB, or multi-GPU) |

If VRAM is tight, drop `--n-gpu-layers` below the model's layer count
(60 for Gemma, 80 for Llama) — the rest spills to system RAM and runs slower. 
ALTERNATIVELY, use a smaller model for testing. 

## GPU offload (important — opt-in)

The script defaults `--n-gpu-layers 99` to
offload all transformer layers to the GPU. (`99` is a sentinel; llama.cpp
clamps to the model's real layer count.) Lower the number if you hit a CUDA
out-of-memory (OOM) error at load time.

To confirm the GPU is actually being used, watch `nvidia-smi -l 2` in a
second terminal — the python process should show meaningful GPU memory and
utilization.

## Smoke test (Gemma, ~few min)

Run this first against the smaller model to verify the environment:

```bash
python llm_utility_approximation/branching_vs_sampling_comparison.py \
  --model-path ./llms/gemma-4-31B-it-Q5_K_M.gguf \
  --model-name gemma-4-31B-it-Q5_K_M \
  --debug-subset 5 \
  --n-samples 200 \
```

If 4 output files appear with no Python tracebacks, continue.

## Production runs (defaults — one per model)

```bash
# Gemma 4 31B-it
python llm_utility_approximation/branching_vs_sampling_comparison.py \
  --model-path ./llms/gemma-4-31B-it-Q5_K_M.gguf \
  --model-name gemma-4-31B-it-Q5_K_M \

# Llama 3.3 70B Instruct
python llm_utility_approximation/branching_vs_sampling_comparison.py \
  --model-path ./llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
  --model-name Llama-3.3-70B-Instruct-Q4_K_M \
```

Defaults in effect: `--scenarios baseline`, `--agent-roles type_a`,
`--temperature 0.3`, `--n-samples 1000`, `--debug-subset 50`,
`--sample-seed 0`.

The script prints per-context
progress so you can tell early if it is hung.

## Output files

Each run writes 4 files under `llm_log_probs/<model_slug>/T<temp_slug>/`:

- `comparison_<model>_<temp>_<timestamp>.csv` — one row per context
- `comparison_<model>_<temp>_<timestamp>_summary.json` — aggregate stats
- `comparison_<model>_<temp>_<timestamp>_scatter.png` — branch vs sample P(MOVE)
- `comparison_<model>_<temp>_<timestamp>_histogram.png` — delta distribution

Two model runs → two output subdirectories.

## How to evaluate

Open each `_summary.json` and check three numbers:

| Metric                              | Good    | Marginal     | Bad    |
| ----------------------------------- | ------- | ------------ | ------ |
| `pct_inside_95_ci_MOVE`           | ≥ 0.90 | 0.75 – 0.90 | < 0.75 |
| `mean_abs_delta_MOVE`             | < 0.05  | 0.05 – 0.10 | > 0.10 |
| `pearson_r_branch_vs_sample_MOVE` | > 0.95  | 0.85 – 0.95 | < 0.85 |

Sanity-check the plots:

- **Scatter**: points should hug the `y = x` line, mostly within their
  horizontal CI bars.
- **Histogram**: centered near 0, no fat tails, no bimodality.

Note: at `--n-samples 1000`, the sampling CI half-width near p = 0.5 is
~±0.031, so any `mean_abs_delta_MOVE` below ~0.03 is at the noise floor —
treat that as "perfect agreement," not a real difference to investigate.

## If something breaks

Re-run with the smallest possible scope to isolate the problem:

```bash
python llm_utility_approximation/branching_vs_sampling_comparison.py \
  --model-path ./llms/gemma-4-31B-it-Q5_K_M.gguf \
  --model-name gemma-4-31B-it-Q5_K_M \
  --debug-subset 2 --n-samples 50 \
```
