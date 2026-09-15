# Rank stability — qwen3.6-27b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_124956_qwen3.6-27b-vf-lp/experiments, n=[10000])
- ruler: None
- ruler rng_scheme: exact
- decision metric(s): **dissimilarity_index** — only these trigger a top-up
- gap floor: 0.01 DI (pairs below it are `≈`, never priced)
- chance level on this board (20x20, 160+160; 2000 random allocations): DI 0.1254 (SD 0.033); DI at chance (no segregation): none
- other metrics: floor = 0.3 x chance SD (the DI ratio), polarity-aware: clusters 2.58, switch_rate 0.00606, distance 0.00804, mix_deviation 0.003, share 0.00486, ghetto_rate 1.18

Every chain reads MOST segregated first (clusters and switch_rate are sign-corrected: fewer clusters / lower switch rate = more segregated).

Legend: `>` certified · `?>` fixable (GPU top-up resolves) · `n>` needs more RUNS, not sampling (ruler already fine, gap under-measured) · `~` unresolved · `=` exact tie · `≈` below the practical-significance floor · `°` at the chance level (no more segregated than a random grid; marked as a tie)

## dissimilarity_index  **(DECISION METRIC)**

```
politics > income ≈ baseline ≈ green > race > ethnic
```
counts: {'CERTIFIED': 3, 'FLOOR-TIE': 2}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.3789 | 0.000911 | +0.2535 |  |
| income | 0.3554 | 0.000888 | +0.2299 |  |
| baseline | 0.3542 | 0.000877 | +0.2287 |  |
| green | 0.3455 | 0.00086 | +0.2201 |  |
| race | 0.2988 | 0.000783 | +0.1734 |  |
| ethnic | 0.1564 | 0.000427 | +0.031 |  |

## clusters

```
politics > green ≈ income ≈ baseline > race > ethnic
```
counts: {'CERTIFIED': 3, 'FLOOR-TIE': 2}

chance 96.21 (SD 8.61), floor 2.58 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 13.62 | 0.0334 | +82.59 |  |
| green | 17.55 | 0.0424 | +78.66 |  |
| income | 17.75 | 0.0429 | +78.46 |  |
| baseline | 18.39 | 0.0433 | +77.82 |  |
| race | 27.38 | 0.0642 | +68.83 |  |
| ethnic | 76.21 | 0.0992 | +20 |  |

## switch_rate

```
politics > income ≈ baseline > green > race > ethnic
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.5016 (SD 0.0202), floor 0.00606 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.2482 | 0.000383 | +0.2533 |  |
| income | 0.2859 | 0.000406 | +0.2157 |  |
| baseline | 0.2897 | 0.000397 | +0.2118 |  |
| green | 0.299 | 0.000401 | +0.2026 |  |
| race | 0.3601 | 0.000413 | +0.1415 |  |
| ethnic | 0.4939 | 0.000225 | +0.007648 |  |

## distance

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 1.152 (SD 0.0268), floor 0.00804

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 2.243 | 0.00259 | +1.091 |  |
| income | 2.103 | 0.00237 | +0.9512 |  |
| baseline | 2.092 | 0.00237 | +0.9403 |  |
| green | 2.056 | 0.00225 | +0.9041 |  |
| race | 1.814 | 0.00183 | +0.6623 |  |
| ethnic | 1.256 | 0.000526 | +0.1047 |  |

## mix_deviation

```
politics > income ≈ baseline > green > race > ethnic
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.168 (SD 0.00999), floor 0.003

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.3458 | 0.000216 | +0.1778 |  |
| income | 0.3264 | 0.000233 | +0.1584 |  |
| baseline | 0.3248 | 0.000228 | +0.1567 |  |
| green | 0.3189 | 0.000233 | +0.1508 |  |
| race | 0.2829 | 0.000252 | +0.1149 |  |
| ethnic | 0.1784 | 0.000138 | +0.01039 |  |

## share

```
politics > income ≈ baseline > green > race > ethnic
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.4989 (SD 0.0162), floor 0.00486

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.8347 | 0.000244 | +0.3358 |  |
| income | 0.8064 | 0.000275 | +0.3075 |  |
| baseline | 0.8045 | 0.000266 | +0.3056 |  |
| green | 0.7981 | 0.000273 | +0.2993 |  |
| race | 0.7444 | 0.000322 | +0.2455 |  |
| ethnic | 0.5463 | 0.000216 | +0.04744 |  |

## ghetto_rate

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 9.222 (SD 3.93), floor 1.18

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 163 | 0.203 | +153.8 |  |
| income | 144.7 | 0.212 | +135.4 |  |
| baseline | 143.3 | 0.208 | +134.1 |  |
| green | 139.6 | 0.208 | +130.4 |  |
| race | 108.3 | 0.21 | +99.04 |  |
| ethnic | 25.78 | 0.0841 | +16.56 |  |
