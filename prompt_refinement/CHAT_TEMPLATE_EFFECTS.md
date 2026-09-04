# What the chat endpoint and jinja templating do to our prompt, per model

Every claim in this document was independently re-verified on 2026-07-30 by a
verification agent against the actual GGUF headers, the llama.cpp source tree,
and the served binary — file:line references below. The extracted Jinja
templates themselves are preserved in `chat_templates/*.jinja` for inspection.

**Why this document exists**: production results differ sharply by endpoint
(llama chat near-freezes; qwen chat DI 0.73 vs 0.42 on completions), and the
endpoint difference is not just "URL" — it is a concrete, model-specific
transformation of the prompt. Interpreting degeneracy requires knowing exactly
what each model's chat arm added around our text.

## The pipeline

Our prompt (persona + neighborhood + "Answer with ONLY one word - MOVE or
STAY") is built by `llm_runner.build_llm_request` (llm_runner.py:91):

- **completions arms**: the prompt is sent raw. The model continues the text.
  NOTHING below applies — no roles, no system text, no reasoning scaffolds.
- **chat arms**: the prompt becomes one `user` message; the SERVER then renders
  the model's chat template around it before tokenization.

Server facts (verified):
- The served binary (`a4ce259`, built 2026-07-14, unchanged since) has
  **`--jinja` ON BY DEFAULT** — so the Jul-15 prompt sweep and all production
  chat runs used the models' real embedded templates, whether or not the flag
  was passed. There is no sweep-vs-production tagging difference.
- `--reasoning off` works ONLY through jinja: `common/arg.cpp:3364` maps it to
  the template variable `enable_thinking=false`; `common/chat.cpp:895` passes
  that into the render; `tools/server/server-context.cpp:1452` forces
  `enable_thinking=false` when jinja is off, and the built-in C++ path
  (`src/llama-chat.cpp`) has no flag-controlled reasoning handling at all.
  **Therefore `--no-jinja` silently disables reasoning suppression** — safe
  only for Llama-3.3 (no reasoning channel to suppress).
- The one pre-standardization run (June 5 llama, run_20260605_141404) was NOT
  served by native llama-server at all but by `python -m llama_cpp.server`
  (transition_to_gemma.sh:61, configs/llama_cpp_server.yaml) — a third,
  single-stream template implementation. Treat its results as a separate era.

## Exact wrappings (thinking off — our production configuration)

`<PROMPT>` marks our full prompt text; generation begins at `█`.

### Llama-3.3-70B — the ONLY model that injects a system block

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: <today>

<|eot_id|><|start_header_id|>user<|end_header_id|>

<PROMPT><|eot_id|><|start_header_id|>assistant<|end_header_id|>

█
```

The dates block is auto-inserted by the template whenever no system message is
supplied (we never supply one). It contains no "you are an assistant" words —
but it is the fingerprint of Llama's assistant deployment format from
post-training. Under `--no-jinja` (the built-in path,
src/llama-chat.cpp:485-493) the system block disappears and only the role
headers remain — that delta is what the sweep's `llama-3.3-70b-nojinja` probe
arm measures.

### Qwen3.6-27B — minimal wrapper + pre-closed think block

```
<|im_start|>user
<PROMPT><|im_end|>
<|im_start|>assistant
<think>

</think>

█
```

No system text, no dates. `--reasoning off` makes the template pre-write the
think block closed and empty ("deliberation finished, nothing written"); with
reasoning on it would end at a dangling `<think>` and our max_tokens=5 would
die inside it (the diagnosed Jul-14 failure).

### gemma-4-31B — thought channel opened and closed

```
<BOS><|turn>user
<PROMPT><turn|>
<|turn>model
<|channel>thought
<channel|>█
```

No system text with thinking off. (Nuance: with thinking ON, gemma's template
prepends a system turn containing only the control token `<|think|>` — not
instructions. We never run that configuration.)

### DeepSeek-V4-Flash — inverted reasoning convention

```
<BOS><｜User｜><PROMPT><｜Assistant｜></think>█
```

(`｜` is U+FF5C fullwidth bar.) No system text. DeepSeek's assistant turn
implicitly begins in thinking mode, so "off" is expressed by a bare CLOSING
`</think>` — thinking declared already over. With reasoning on it would open
`<think>` instead.

### Mistral-Small-4-119B — the extreme case: full assistant identity injected

```
<s>[SYSTEM_PROMPT]You are Mistral-Small-4-119B-2603, a Large Language Model
(LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on Friday, November 1, 2024.
The current date is <today>.
… (2,424 chars total: clarifying-question policy, date-resolution policy,
   # WEB BROWSING INSTRUCTIONS, # MULTI-MODAL INSTRUCTIONS,
   # TOOL CALLING INSTRUCTIONS) …
[/SYSTEM_PROMPT][MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]<PROMPT>[/INST]█
```

Full text in `chat_templates/mistral4.jinja` (default_system_message, template
line ~145). Two verified corrections vs earlier assumptions: (1) Mistral is
hybrid-reasoning, controlled by a dedicated `reasoning_effort` template
variable ("none"|"high") that DEFAULTS to "none" — `enable_thinking` (and thus
`--reasoning off`) appears nowhere in its template and has no effect on it;
(2) the `<s>` BOS is hardcoded in the template text.

## Relevance to degeneracy — the evidence table

| model · arm | wrapper adds | observed behavior |
|---|---|---|
| llama completions | nothing | moves, DI ~0.28, runs to step cap |
| llama chat | role scaffold + dates system block | near-freeze; total on race/ethnic/political, sparse on baseline/income/green_yellow |
| qwen completions | nothing | moves, DI 0.42, still rising at step 100 |
| qwen chat | role scaffold + empty think block ONLY | DI 0.73, fast segregation — largest effect in campaign |
| gemma chat | role scaffold + closed thought channel | pending (gemma chain queued) |
| deepseek chat | role scaffold + `</think>` | never run in production |
| mistral chat | role scaffold + **2.4k-char Le Chat identity** conflicting with our persona | never run in production |

Two inferences the table supports, and one warning:

1. **Role scaffolding alone is behaviorally massive.** Qwen's wrapper adds no
   instructions whatsoever — user/assistant tags and an empty think block —
   yet flips DI by +0.31 and collapses scenario ordering. The mechanism is
   conditioning: after the assistant header, tokens are drawn from the
   assistant-persona distribution (answering a question ABOUT a resident)
   rather than continuing the resident's own narrative.
2. **The direction of the shift is model-specific**: the same class of wrapper
   makes qwen segregate MORE and llama move LESS. Chat wrapping is not a
   uniform bias; endpoint must be treated as a first-class experimental
   factor, never averaged over.
3. **Warning for the sweep**: Mistral's chat arms carry a built-in identity
   conflict (told it is Le Chat before our persona line). If its chat value
   functions look anomalous relative to its completions arms, the injected
   system prompt is the first suspect — that comparison cannot distinguish
   "chat scaffolding effect" from "identity-conflict effect" without a custom
   `--chat-template` run.

## Reproducing / re-verifying

Templates were extracted by reading `tokenizer.chat_template` from the GGUF
header (key string, u32 type=8, u64 length, UTF-8 payload) and rendered with
jinja2 (`trim_blocks/lstrip_blocks`, stub `strftime_now`, messages = one user
turn, `add_generation_prompt=True`, `enable_thinking` as noted). Built-in path
claims: src/llama-chat.cpp:250-257 (ChatML), :485-493 (Llama-3). Reasoning
plumbing: common/arg.cpp:3364, common/chat.cpp:895,
tools/server/server-context.cpp:1452.
