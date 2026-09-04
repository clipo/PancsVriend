# 02 — Prompt sweeps: how the model was asked, and how that was chosen

Harnesses: `prompt_refinement/evaluate_prompts.py` (grid prompts),
`evaluate_ratio_prompts.py` (ratio prompts); shared sampling/parsing in
`sampling_common.py`. Deep notes: `prompt_refinement/NOTES.md`,
`CHAT_TEMPLATE_EFFECTS.md`. All sweeps use the production payload (same
sampler, `max_tokens=5`, T=0.3) and the production MOVE/STAY parse rule, so
measured rates are what agents would do in simulation.

## 1. The two prompt families

**Grid prompts** (`prompt_templates.py`, 8 candidates): the neighbourhood as
a 3×3 ASCII map with a legend (X self, S same, O opposite, E empty, # wall).
Candidates vary only the scaffolding — legend before/after map, question
placement and word order, persona present/absent. Swept over the 9-point
out-group gradient (0..8 of 8), fresh random layouts per sample, paired
across candidates by seed.

**Ratio prompts** (`ratio_prompt_templates.py`, 5 styles + a grid anchor):
composition stated in words, never arrangement — matching the mechanical
agent's information set. A 2×2 design of encoding (count vs percent) ×
valence (opposite- vs similar-framed), plus a symmetric dual-count control:

| style | example (5-of-8 opposite) |
|---|---|
| R1_count_opposite | "5 of your 8 neighbors are blue team residents." |
| R2_count_similar | "3 of your 8 neighbors are red team residents like you." |
| **R3_dual_count** | "You have 8 neighbors: 3 are red team residents like you and 5 are blue team residents." |
| R4_percent_opposite | "62.5% of your neighbors are blue team residents." |
| R5_percent_similar | "37.5% of your neighbors are red team residents like you." |
| G0_grid_anchor | the 3×3 map (ties the two families together) |

Wording rules: never mention vacancies, walls or "of the 8 slots"; no verbal
quantifiers; fixed text above the varying sentence; correct pluralisation and
articles for every identity label. Swept over all 45 (n_similar, n_occupied)
compositions, both roles, N=50/cell.

## 2. Endpoint × grammar arms

Every candidate was measured on four arms: raw `/v1/completions` vs
`/v1/chat/completions` (the server applies the model's chat template;
reasoning-capable templates run with `--reasoning off`), each with and
without the MOVE/STAY GBNF grammar. Three findings fixed the protocol:

- **Grammar**: 0 unparseable replies in 473,625 grammar samples, and
  structural immunity to *silent* misparse (truncated enumerations, embedded
  keywords) that biases plain-arm rates upward — `GRAMMAR_GBNF_NOTES.md`.
- **Chat**: every model answers voluntarily on chat (0.0% bad parses),
  whereas 3 of 5 models are forced-choice on plain completions (up to 59%
  unparseable), so grammar readings there are protocol-constructed.
- **Style**: R3 dual-count won the cross-arm consistency ranking on both
  roles; percent styles are treated as artifact probes (genre priors,
  aliasing) and barred from winning unless they agree with the count forms.

**Production protocol = R3 dual-count, chat + grammar.**

## 3. Headline findings

- Prompt structure is behaviourally massive: the legacy production prompt
  produced a flat 0–1% move rate across the whole gradient, while the refined
  candidates restore full-range monotone responses at ~10× lower token cost.
- Valence, not numeric encoding, is the large phrasing effect
  (opposite-framed thresholds ≪ similar-framed).
- Role asymmetry (red vs blue identity within the *same* scenario) is model-
  and arm-dependent, and is never pooled.

## 4. Caveat carried to the paper

The sweeps ran before the clean serving protocol, so transition-cell numeric
values and interpolated thresholds carry unquantified KV-cache episode bias
(`KV_CACHE_SAMPLING_ARTIFACT.md`); qualitative rankings and saturated-cell
structure are unaffected. The production value functions (03) were
re-measured under the clean protocol.
