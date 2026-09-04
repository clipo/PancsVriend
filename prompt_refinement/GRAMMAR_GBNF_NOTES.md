# GBNF grammar-constrained decoding: function, advantages, caveats

What the `grammar` field in our llama-server requests actually does, why the
grammar arm is this project's production measurement channel, and the caveats
that come with it. Companions: `LLAMA_CPP_SERVING_NOTES.md` (server-side
serving concepts), `KV_CACHE_SAMPLING_ARTIFACT.md` (the cache-state sampling
artifact — orthogonal to grammar), `NOTES.md` (results-reading caveats).

## 1. How it functions

GBNF ("GGML BNF") is llama.cpp's format for context-free grammars. At
generation time the grammar is compiled into a state machine consulted at
every decoding step: tokens that cannot legally extend the partial output are
masked to −∞ logit, and the model's probability mass over the *surviving*
tokens is renormalized before sampling. (As an optimization, llama.cpp first
samples unconstrained and only computes the full mask if the sampled token is
rejected — the resulting distribution is the same.) See the llama.cpp
grammar internals writeup ([DeepWiki: Grammar and Structured
Output](https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output))
and a general treatment of constrained decoding
([zeroentropy.dev](https://zeroentropy.dev/concepts/constrained-decoding/)).

Our production grammar (`llm_runner.MOVE_STAY_GRAMMAR`, re-exported as
`sampling_common.GRAMMAR`) admits exactly: optional whitespace, then MOVE or
STAY in any casing, then end. Deliberately *permissive* about surface form
(leading newlines, casing) so the model's natural first-token habits are
distorted as little as possible while still making every reply parseable and
halting generation at the keyword.

## 2. What it provides

1. **Parseability by construction.** The reply cannot be anything but
   MOVE/STAY(+whitespace). Measured: **0 bad parses in 473,625 grammar-arm
   samples** across all sweeps, vs up to 59% bad on plain arms for some
   models (see NOTES.md "Prefer GRAMMAR arms").
2. **Structural immunity to SILENT misparse.** The plain-endpoint substring
   parser can record a wrong-but-confident decision (truncated enumeration
   `'MOVE\nor\nST'`; embedded keywords `MOVEMENT`) — impossible under
   grammar, because those strings cannot be generated. This, not parse *rate*,
   is the decisive argument (NOTES.md, 2026-08-13 section).
3. **Deterministic halting** the moment the keyword completes — no run-on
   commentary, `max_tokens` slack never binds.
4. **Cross-tool byte parity**: every measurement harness imports the single
   production grammar constant, so grammar arms are byte-identical to what
   the simulation sends. (Historical note: three byte-variant copies of the
   same language once existed; `evaluate_prompts.py` still carries its legacy
   local copy for reproducibility of old red-sweep payloads.)

## 3. Caveats

1. **Grammar does not make an unwilling model willing (forced choice).** If
   the model would not answer MOVE/STAY unconstrained, the grammar-arm
   distribution is *protocol-constructed* — behaviour under our forced-choice
   protocol, not revealed preference. The voluntary/forced-choice verdicts in
   `ratio_consistency_summary.md` (plain-arm parse health + plain-vs-grammar
   agreement) are the instrument for telling these apart. Keep a plain arm as
   a validity probe; never present grammar-arm numbers from a forced-choice
   (model, endpoint) as preference measurements.
2. **Per-token renormalization ≠ conditioning on grammaticality.** Masking
   renormalizes *token by token*, which yields a distribution that is not, in
   general, the model's own distribution conditioned on producing a
   grammatical string — a documented distortion of grammar-constrained
   decoding, with measurable task-accuracy effects in the literature
   ([Lost in Space, arXiv:2502.14969](https://arxiv.org/html/2502.14969v1);
   [overview](https://tianpan.co/blog/2026-04-16-grammar-constrained-generation-output-reliability)).
   For OUR grammar the distortion is near-minimal — the decision is
   effectively a single content token and both alternatives are live at that
   position — but it is not exactly zero (whitespace/casing token paths carry
   slightly different mass), and any richer grammar added later should
   revisit this.
3. **Renormalization artifact with reasoning-tuned models.** A model whose
   unconstrained continuation wants to open a think-block has that mass
   redistributed onto MOVE/STAY by the mask. The measured rate is then "which
   keyword wins *inside a distribution that wanted to do something else*" —
   the sharpest form of caveat 1. This is why chat arms for
   Gemma/Qwen/DeepSeek require `--reasoning off` at the SERVER (closing the
   reasoning channel via the chat template) rather than relying on the
   grammar to suppress it (see FUTURE_EXPLORATIONS.md item 4).
4. **Tokenization/whitespace sensitivity.** Grammars interact with the
   tokenizer: which whitespace/casing paths the grammar admits changes which
   token sequences are reachable and hence the measured split
   ([arXiv:2502.14969](https://arxiv.org/html/2502.14969v1)). Our permissive
   `ws` rule is a design choice to keep the model's preferred surface form
   available; changing the grammar bytes in a way that changes the *language*
   would change measurements and must be treated as a new arm.
5. **Runtime overhead** of grammar checking exists but is negligible at our
   scale (≤5-token outputs; the sample-first-check-later optimization means
   the common path costs almost nothing).
6. **Orthogonal to serving-side artifacts.** Grammar fixes parsing, not
   logits: the KV-cache state dependence documented in
   `KV_CACHE_SAMPLING_ARTIFACT.md` biased grammar-arm measurements exactly as
   it would plain ones.

## Sources

- [llama.cpp grammar internals (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output)
- [Constrained decoding overview (zeroentropy.dev)](https://zeroentropy.dev/concepts/constrained-decoding/)
- [Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding (arXiv:2502.14969)](https://arxiv.org/html/2502.14969v1)
- [Grammar-constrained generation reliability overview (tianpan.co)](https://tianpan.co/blog/2026-04-16-grammar-constrained-generation-output-reliability)
- [llama-cpp-python grammar docs (DeepWiki)](https://deepwiki.com/abetlen/llama-cpp-python/6.1-grammar-based-generation)
