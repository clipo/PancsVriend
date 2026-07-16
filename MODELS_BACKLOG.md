# Models backlog — evaluate later

Deferred candidates for the Schelling LLM-agent study, per discussion 2026-07-15.
The active additions (downloading now) are **Qwen3.6-27B dense** and
**Mistral-Small-4-119B**; see `~/.claude/plans/` plan and `prompt_refinement/README.md`
for the validation gate every model must pass (prompt eval on completions AND
chat+grammar; range ≥ ~0.9, spearman ≥ +0.95, ~0 bad parses on at least one endpoint).

## Queued for later — Mixture-of-Experts round

| model | HF GGUF repo | quant / size | why deferred / notes |
|---|---|---|---|
| **Qwen3.6-35B-A3B** (MoE, 3B active) | `unsloth/Qwen3.6-35B-A3B-GGUF` (non-MTP; 843K downloads) | Q4_K_M ~20GB | Modern MoE axis. ~10× decode speed vs dense → big win for 10⁵-call production runs. Compare against Qwen3.6-27B dense (same lab & generation) to isolate the MoE variable. |
| **Llama-4-Scout-17B-16E** (MoE, 109B total / 17B active) | `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF` | Q4_K_M ~65GB | Within-Meta generation jump vs Llama-3.3-70B + second MoE data point. `llama4` arch — verify the local llama.cpp build loads it (build runs Gemma-4, so likely OK; smoke-test first). |
| **DeepSeek-V4-Flash** — *PROMOTED 2026-07-15: downloading UD-IQ3_XXS (103GB, 4 shards; Q4-class is 138GB+ and does not fit), full suite queued* | `unsloth/DeepSeek-V4-Flash-GGUF` | UD-IQ3_XXS 103GB (3-bit — quant-fidelity caveat vs others' Q4/Q5) | NOT excluded for being a reasoning model — raw `/v1/completions` bypasses the thinking channel and `--reasoning off` tames chat mode (both proven on Gemma-4). Deferred because the `deepseek_v4` arch (released 2026-07-06) may postdate the local llama.cpp build (commit a4ce259): download → load smoke-test → if rejected, update llama.cpp (`git pull && cmake --build build`). Also check total/active param count fits 128GB unified memory at Q4. MIT license. |

## Considered and parked (with reasons)

- **Qwen2.5-72B-Instruct** (`Qwen/Qwen2.5-72B-Instruct` GGUFs, Q4_K_M ~44GB) — matched-size
  cross-lab pair with Llama-3.3-70B (isolates lab at fixed ~70B dense scale). Strong design,
  but previous generation (2024). Revisit if the size-matched comparison becomes a paper claim.
- **Mixtral-8x7B / 8x22B** — skipped: 2023/2024 models, two generations old; Mistral-the-lab is
  covered by Mistral-Small-4-119B, and the MoE axis is covered by Qwen3.6-35B-A3B above.
- **Tencent Hy3, GLM-5.2** — frontier/China axis; arch support unverified (`glm_moe_dsa` etc.),
  GLM-5.2 likely exceeds 128GB. Revisit alongside DeepSeek-V4-Flash if a frontier round happens.
- **Community fine-tunes / abliterated variants** (DavidAU, heretic, etc.) — excluded on
  principle: not official lab checkpoints, weak provenance for a research paper.

## Hardware constraint reminder

GB10: 128GB unified memory, ~110GB practical ceiling for model + KV cache at `-np 8 -c 16384`.
Q4_K_M ≈ 0.6GB per B params (dense). Anything ≥ ~180B dense does not fit.
