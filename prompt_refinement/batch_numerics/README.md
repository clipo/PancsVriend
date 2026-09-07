# Batch-numerics artifact — does a llama-server probability depend on the batch?

Everything about this artifact lives here: the probe, its data and its plots.

* `batch_numerics_probe.py` — for one model (server up with the campaign's
  flags), reads P(MOVE) from `/v1/chat/completions` post-sampling probabilities
  for the most sensitive unsaturated cells under three conditions — sequential
  (one request in flight), concurrent (four identical requests in flight),
  interleaved (mixed with other cells' requests) — plus sequential
  campaign-payload sampling, and the campaign's own rate with its Wilson CI.
* `results/probe_<label>.csv` — every request (condition, repeat, P(MOVE), prompt tokens).
* `results/probe_<label>_summary.csv` — per cell: sequential value and its
  range over repeats, concurrent / interleaved mean-sd-min-max, campaign p and
  CI, sequential-sampling p and CI, campaign − sequential.
* `results/probe_<label>.json` — server build/flags and aggregates.
* `results/probe_<label>.png` — the per-cell picture.

* `run_batch_numerics_study.sh` — the ONE-TIME study: starts each model's
  server with its campaign flags, runs the probe, stops it. Not part of any
  pipeline; run once, keep the CSVs (`--wait` defers until the port is free).

All requests use the chat endpoint, the campaign's own; the first measurements
of this effect (2026-09-05) went through `/completion` and were discarded as
not comparable.

Finding (see `../KV_CACHE_SAMPLING_ARTIFACT.md` §8): sequential requests are
bit-reproducible; with four in flight the same request returns probabilities
spread over tens of percentage points on transition cells, flash attention on
or off, whatever shares the batch. The sampling campaigns ran four in flight,
so their `vf_*.json` tables are batch-state averages; the exact tables in
`../results/value_functions_logprob/` are sequential.
