# 04 — Serving infrastructure: models, llama.cpp, grammar, clock pin, batch numerics

Code: `value_functions/serving.py` (model registry, pins, `PinnedServer`).
Deep notes: `prompt_refinement/LLAMA_CPP_SERVING_NOTES.md`,
`GRAMMAR_GBNF_NOTES.md`, `KV_CACHE_SAMPLING_ARTIFACT.md`,
`value_functions/batch_numerics/README.md`.

## 1. Models

All nine are open-weight models served from local gguf files (`llms/`). The
quantisation differs by model and is part of the model's identity for every
number in the paper.

| paper name | gguf file | quantisation | reasoning flag | pinned clock |
|---|---|---|---|---|
| Olmo 2 32B | `OLMo-2-0325-32B-Instruct-Q8_0.gguf` | Q8_0 | — | (date-free template) |
| DeepSeek V4 Flash | `DeepSeek-V4-Flash-UD-IQ3_XXS-0000{1..4}-of-00004.gguf` | IQ3_XXS | off | (date-free) |
| Llama 3.3 70B | `Llama-3.3-70B-Instruct-Q4_K_M.gguf` | Q4_K_M | off | **2024-07-26** |
| Qwen 3.6 27B | `Qwen3.6-27B-Q5_K_M.gguf` | Q5_K_M | off | (date-free) |
| Gemma 4 31B | `gemma-4-31B-it-Q5_K_M.gguf` | Q5_K_M | off | (date-free) |
| Hermes 4.3 36B | `hermes-4_3_36b-Q8_0.gguf` | Q8_0 | off | (date-free) |
| Granite 4.2 30B | `granite-4.2-30b-Q8_0.gguf` | Q8_0 | off | (date-free) |
| Mistral Small 4 119B | `Mistral-Small-4-119B-2603-UD-Q4_K_M-0000{1..3}-of-00003.gguf` | Q4_K_M | off | **2026-03-16** |
| Phi 4 14B | `phi-4-Q8_0.gguf` | Q8_0 | — | (date-free) |

DeepSeek V4 Flash is run at a ~3-bit quantisation (its gguf is 96 GB on a
121 GB machine), far more aggressive than the others. A Q8_0 Gemma 4 31B
comparison (`configs/vf_run_gemma_q8_lp.yaml`,
`value_functions/new_model_chain.py gemma-q8`) is prepared to measure how
much quantisation moves a choice function; it is not part of the nine-model
result.

## 2. Serving stack

Local llama.cpp `llama-server`, build `b1-a4ce259`, one model at a time, on
an NVIDIA GB10 (121 GB unified memory). Flags:

```
-ngl -1 -fa on -np 4 -c 8192 --host 127.0.0.1 --port 8085 --jinja [--reasoning off]
```

— full GPU offload, flash attention, the model's own chat template, reasoning
channels closed where the template has one. The sampler is pure temperature
(T = 0.3; top_k 0, top_p 1, min_p 0, all penalties off) via
`llm_runner.SAMPLER_PARAMS`, identical in every harness. The server exposes
four slots, but every measurement that feeds the paper is issued one request
at a time (§4).

**Grammar (GBNF).** Replies are constrained to `optional-whitespace +
MOVE|STAY (any casing)` by `llm_runner.MOVE_STAY_GRAMMAR`:

```
root   ::= ws answer
ws     ::= [ \t\n]*
answer ::= move | stay
move   ::= [Mm] [Oo] [Vv] [Ee]
stay   ::= [Ss] [Tt] [Aa] [Yy]
```

In the sweeps this
gave 0 unparseable replies in 473,625 samples. The measurement is therefore a
forced choice: the model cannot refuse, hedge or explain, and reasoning is
off. Caveats (per-token renormalisation is not conditioning on
grammaticality; models that will not answer unconstrained) are in
`GRAMMAR_GBNF_NOTES.md`.

## 3. Clock pin (found 2026-09-12)

llama.cpp injects the server's current date into every chat-template
rendering. The Llama 3.3 template prints it ("Today Date: …", falling back
to its built-in "26 Jul 2024" only when no date is supplied) and the Mistral
Small 4 template prints it with no fallback. Their prompts — and therefore
their choice functions — changed every calendar day: Llama's table moved by
up to 0.45 in P(MOVE) on transition cells between extraction days, while
being bit-stable within a day.

Every server is now started through `PinnedServer`, which freezes the
process's realtime clock with libfaketime (**Llama: 2024-07-26**, the
template's own default; **Mistral: 2026-03-16**, the model's release date),
renders a probe prompt and refuses to serve unless the pinned date appears
in it. The other seven templates print no date (verified bit-identical with
and without the pin). Extraction, the census and escalation all run under
the same pin; traces record `server_env` and `rendered_probe_prompt`. The
Llama and Mistral tables in the paper are the post-pin extractions (runs of
2026-09-12 and 2026-09-13). Details: `LLAMA_CPP_SERVING_NOTES.md` §6.

## 4. Batch-numerics artifact (found 2026-09-05)

With one request in flight, llama-server's returned probabilities are
bit-reproducible. With four in flight, the same byte-identical request
returns probabilities spread over tens of percentage points on transition
cells — flash attention on or off, whatever else shares the batch. Saturated
cells are unaffected. The spread tracks the weight quantisation: the four
Q8_0 models have a maximum concurrent span of 0.15–0.37 in P(MOVE), the
lower-bit models 0.30–1.00. T = 0.3 amplifies it, since post-temperature
logits are raw logits / 0.3.

Consequences:

- The sampled value functions (four in flight) are batch-state averages and
  were retired (03 §5).
- Exact extraction and all validation sampling run strictly sequentially.
- The earlier "KV-cache artifact" (2026-08-22: sampled probabilities
  depending on slot cache history, remediated with `cache_prompt:false` and
  per-request seeds) is documented in `KV_CACHE_SAMPLING_ARTIFACT.md`; its §8
  records the batch finding that superseded that remediation as the
  explanation of record.

The probe, its per-request data and plots for all nine models are in
`value_functions/batch_numerics/` (tracked).

## 5. Reproducibility tiers

- **Simulations** — fully deterministic given the frozen tables:
  `random_seed = run_id` plus keyed decision draws (01 §2).
- **Exact tables** — deterministic per (llama.cpp build, gguf file, hardware,
  pinned date), sequential requests. Re-extraction on the same stack
  reproduces a table bit for bit.
- **Sampling used for validation** — sequential, cache off, per-request seeds
  (crc32 of the sample's identity tuple).
- Cross-build, cross-hardware or cross-quantisation agreement is not claimed.
