# Methods documentation — LLM Schelling segregation study

This README is the end-to-end narrative; the numbered documents carry the
reproducibility detail, with pointers into the technical notes in
`prompt_refinement/`.

| doc | contents |
|---|---|
| [01_SCHELLING_MODEL.md](01_SCHELLING_MODEL.md) | grid, agents, decision rules, convergence, metrics, run outputs |
| [02_PROMPT_SWEEPS.md](02_PROMPT_SWEEPS.md) | how the LLM was asked: prompt families, endpoint arms, protocol choice |
| [03_VALUE_FUNCTION.md](03_VALUE_FUNCTION.md) | the sampled value function: what is measured, sample sizes, artifacts, coupling |
| [04_INFRASTRUCTURE.md](04_INFRASTRUCTURE.md) | serving stack, grammar, the KV-cache artifact, seeds |
| [05_REPRODUCIBILITY.md](05_REPRODUCIBILITY.md) | environment, commands, data locations, caveats registry |
| [VALUE_FUNCTION_SAMPLING_FOR_REVIEW.md](VALUE_FUNCTION_SAMPLING_FOR_REVIEW.md) | standalone review note on the per-cell sample-size rule |

---

## The study in plain language

**The question.** Schelling's classic model shows that mild individual
preferences about neighbours can produce strong residential segregation. We
ask: if the agents are a large language model making human-like housing
decisions — told it is, say, a *white middle class family* or a *low-income
household* — how does its behaviour compare to the mechanical Schelling
agent, and how does the *social framing* of the same situation change the
outcome?

**The world.** A 10×10 grid holds 40 red-type and 40 blue-type agents, with
20 cells empty. Each round, every agent looks at its up-to-8 immediate
neighbours and decides to stay or to move to a random empty cell. The
mechanical reference agent moves exactly when more than half its neighbours
are unlike it. Six scenarios dress the same two types in different
identities: neutral team colours (two variants), race, ethnicity, income and
politics.

**Asking the model, and asking it well.** LLM answers depend heavily on how a
question is posed, so we first ran systematic prompt sweeps. Two families:
how to present the neighbourhood (a map, counts, or percentages), and how to
ask across two API channels (raw completion vs chat template), with and
without a grammar constraining the reply to exactly MOVE or STAY. The winner
became the production protocol: a dual-count sentence ("You have 8 neighbors:
3 are X like you and 5 are Y") on the chat channel, with grammar.

**The value function.** Rather than calling the model live inside every
simulation — slow, noisy, unreproducible — we measure it once. For each of
the 45 neighbourhood compositions an agent can face, we ask the same question
hundreds of times and record how often it answers MOVE. That table, per role
and per scenario, is the model's **sampled value function**, and simulations
draw their decisions from it: fast, seeded, exactly reproducible, and
statistically faithful to the model.

**How many samples are enough?** A two-stage design: a uniform pilot of 100
samples per composition, then a targeted top-up bringing every estimate's 95%
confidence interval under ±2 percentage points. This is cheap, because most
compositions produce unanimous answers and only the few torn ones need
thousands of samples. Sufficiency is then demonstrated rather than assumed:
rebuilding every table from half its samples and re-running all simulations
changes no metric by more than its own error bar (42/42 checks).

**What we found.** The social context dominates. With neutral labels the
model behaves like a reluctant Schelling agent, moving only at ~70–85% unlike
neighbours against the mechanical 50%. Under racial and ethnic labels it
essentially refuses to move at all, and those simulated neighbourhoods never
segregate beyond their random start. Under income labels it is the most
mechanical-like, and one-sided: high-income agents readily leave poor
neighbourhoods, and the reverse almost never happens. Every scenario
segregates far less than the mechanical baseline on all seven metrics.

**Two methodological findings we did not go looking for.** The pipeline's
self-checks caught the inference server's prompt-cache reuse *changing the
sampled probabilities themselves* at exactly the torn compositions — by tens
of percentage points, stably within an episode, and invisibly to any
within-batch check. All production measurements were redone with caching
disabled and per-request seeds ([04](04_INFRASTRUCTURE.md)). And holding the
ratio of unlike neighbours fixed while varying the neighbourhood size shows
the model's decision is not a function of that ratio alone: at small counts,
absolute numbers flip the behaviour outright, so the ratio abstraction the
classical model assumes is only an approximation of how these agents behave
([03](03_VALUE_FUNCTION.md)).
