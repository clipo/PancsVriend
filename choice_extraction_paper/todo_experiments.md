# Experiment to-dos

Dates are **assigned**. "Offline" means computed from the stored records with no GPU time.
New runs on the DGX Spark need Srikanth's go-ahead before they start.

## Must-have for the January submission (offline)

| ID | What | Data | Output | Date |
|---|---|---|---|---|
| E1 | First token vs path sum per state: m_off, FT-canon (T=1 and T=0.3), FT-mask, path; on unsaturated states and on the logit scale | `vflp_*_states.jsonl.gz` | `value_functions/first_token/` script; Fig. 2 | 5 Dec |
| E0 | Finish Mistral and DeepSeek re-census; run the pre-registered Qwen-cell adjudication (2,000 draws); collect all 9 G values. Report the adjudication whichever way it falls | existing runs | Table 1 | 1 Dec |
| E2 | Mass by path type: canonical token, casing, leading space, sub-word split (ST + AY), whitespace depth | stored trees | Table 2 or one sentence | 8 Dec |
| E3 | Deviance test of each readout (FT-canon, FT-mask, path) against the census | census counts | §4 text | 10 Dec |
| E4 | Grammar-conditional probability (renormalised once over complete paths) vs the sampler's local-masking law | unmasked top-64 tables | §4 text | 10 Dec |

## Should-have (new runs, a few GPU hours each; needs go-ahead)

| ID | What | Size | Date |
|---|---|---|---|
| E5 | Grammar-off arm with the Sun and Zhang (2025) pivot-token baseline | ~30 states x 3 models | 15 Dec |
| E6 | Temperature sweep T = 0.3, 0.7, 1.0, path sum vs sequential samples | same states | 15 Dec |
| E7 | Concurrency replication with a fixed protocol, plus one Q8 or F16 model | 10 states per model | 15 Dec |

## Other choice types (must-do for the paper)

MOVE vs STAY has a property that flatters first-token reads: no token starts both options, so the first letter decides. The paper must show the method on choices where that is not true.

| ID | Task | Why it matters | Size (estimate) |
|---|---|---|---|
| X1 | **Shared-prefix verbal labels**: OpinionQA-style items (Santurkar et al., 2023) with "Strongly agree / Somewhat agree / Somewhat disagree / Strongly disagree" | First token does not decide; the human answer distribution gives a downstream metric | Pilot: 40 items x 2 models, ~9k requests, ~12 h. Full: 150 items x 4 models, ~44k requests |
| X2 | **Integer answers**: ultimatum offer 0 to 10, public-goods contribution 0 to 20 ("1" is a prefix of "10") | Multi-token numbers with shared prefixes | 100 states x 4 models, ~34k requests |
| X3 | **Other binary agent decisions**: prisoner's dilemma, ultimatum accept/reject | Shows the method outside Schelling | 100 states x 4 models, ~41k requests |
| X4 | **Backend and precision**: one model at Q8 or F16, and the path sum from full logits in HF transformers (removes the top-64 truncation) | Separates method from llama.cpp and quantisation | ~12k requests |

Request counts and hours are estimates assuming about 5 s per sequential request on the DGX Spark. Time one request before planning around them.

## Later (extended version)

- E8 Propagation to Schelling outcomes: rerun with first-token tables and compare the dissimilarity index. Deepens overlap with the Schelling paper, so last.
- E9 Three-option task; label-order bias results from the July estimator work.
- Do not reuse the July first-token vs sampling agreement as evidence: the sampler and the estimator read the same first-token distribution, and its seeds were correlated.
