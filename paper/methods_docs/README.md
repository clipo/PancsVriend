# Methods documentation — LLM Schelling segregation study

This README is the end-to-end narrative; the numbered documents carry the
reproducibility detail, with pointers into the technical notes in
`prompt_refinement/` and `value_functions/`.

Last full revision: 2026-11-16 (exact log-probability tables, nine models,
20×20 grid, pinned clocks). The previous revision (2026-09-07) described the
sampled value functions, which are retired — see 03 §5.

| doc | contents |
|---|---|
| [01_SCHELLING_MODEL.md](01_SCHELLING_MODEL.md) | grid, agents, decision rules, stopping, seeding, metrics, significance rules, run outputs |
| [02_PROMPT_SWEEPS.md](02_PROMPT_SWEEPS.md) | how the LLM was asked: prompt families, endpoint arms, protocol choice |
| [03_VALUE_FUNCTION.md](03_VALUE_FUNCTION.md) | the exact choice function: what is computed, how it is validated, how the simulation uses it |
| [04_INFRASTRUCTURE.md](04_INFRASTRUCTURE.md) | models and quantisations, serving stack, grammar, clock pin, batch-numerics artifact |
| [05_REPRODUCIBILITY.md](05_REPRODUCIBILITY.md) | environment, commands, data locations, caveats registry |
| [06_STATISTICAL_METHODOLOGY.md](06_STATISTICAL_METHODOLOGY.md) | pointer to the root `STATISTICAL_METHODOLOGY.md` (Welch t, Holm, Cohen's d floor) and the open points |
| [VALUE_FUNCTION_SAMPLING_FOR_REVIEW.md](VALUE_FUNCTION_SAMPLING_FOR_REVIEW.md) | RETIRED: review note on the sample-size rule of the sampled route |

---

## The study in plain language

**The question.** The Sakoda–Schelling model shows that mild individual
preferences about neighbours can produce strong residential segregation. We
ask: if the agents are a large language model making a housing decision —
told it is, say, a *white middle class family* or a *low-income household* —
how does the *social framing* of the same situation change the outcome, and
does the ordering of segregation across framings match the ordering observed
in US data (economic < political < racial)?

**The world.** A 20×20 grid holds 160 agents of each of two types, with 80
cells empty. Each step, every agent in turn (random order) looks at its
up-to-8 immediate neighbours and decides to stay or to move to a random empty
cell. A run ends after 5 consecutive steps with no move, or at 1,000 steps.
Six scenarios dress the same two types in different identities: two neutral
colour pairs (red/blue, green/yellow), race, ethnicity, income and politics.
Each model × scenario is simulated 10,000 times.

**Asking the model, and asking it well.** LLM answers depend heavily on how a
question is posed, so we first ran systematic prompt sweeps: how to present
the neighbourhood (a map, counts, or percentages), and how to ask across two
API channels (raw completion vs chat template), with and without a grammar
constraining the reply to exactly MOVE or STAY. The winner became the
production protocol: a dual-count sentence ("You have 8 neighbors: 3 are X
like you and 5 are Y") on the chat channel, with grammar, at temperature 0.3.

**The choice function, computed exactly.** Rather than calling the model live
inside every simulation, we measure it once. An agent can face 45
neighbourhood compositions (n_similar, n_occupied); with 2 roles and 6
scenarios that is 540 cells per model. For each cell we read the model's
next-token probabilities, apply the MOVE/STAY grammar ourselves, and sum the
probability of every token path that spells MOVE. That gives P(MOVE) exactly
— no sampling, so no sampling error — for the same prompt, endpoint, grammar
and temperature a live agent would see. Simulations then draw each decision
as a Bernoulli trial from the agent's cell: fast, seeded, exactly
reproducible.

**Nine models.** Olmo 2 32B, DeepSeek V4 Flash, Llama 3.3 70B, Qwen 3.6 27B,
Gemma 4 31B, Hermes 4.3 36B, Granite 4.2 30B, Mistral Small 4 119B and Phi 4
14B, all served locally as quantised gguf files through llama.cpp
([04](04_INFRASTRUCTURE.md) lists the files and quantisation levels).

**Methodological findings we did not go looking for.**

1. *Served probabilities depend on the batch.* With more than one request in
   flight, llama-server returns different probabilities for byte-identical
   requests — by tens of percentage points at torn compositions. The earlier
   sampled value functions were collected four-in-flight and carry this
   artifact, which is why they were retired in favour of exact, strictly
   sequential extraction ([04](04_INFRASTRUCTURE.md) §4).
2. *The prompt changed every calendar day.* llama.cpp injects today's date
   into the chat template, and the Llama 3.3 and Mistral templates print it,
   so their choice functions moved daily. Every server now runs under a
   pinned clock ([04](04_INFRASTRUCTURE.md) §3).
3. *The decision is not a function of the ratio alone.* Holding the fraction
   of unlike neighbours fixed while varying the neighbourhood size changes
   the answer, so the simulation looks up the full (n_similar, n_occupied)
   composition, never a ratio ([03](03_VALUE_FUNCTION.md) §3).

**Results** live in the paper and are regenerated from
`experiments_with_llama_cpp/cross_model/` (bump and level charts, test
tables) and `analysis_tools/ccs2026_presentation/` (the paper's ordering
table and figures); this folder documents method only.
