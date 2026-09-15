# Rank stability — olmo-2-32b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_131256_olmo-2-32b-vf-lp/experiments, n=[10000])
- ruler: None
- ruler rng_scheme: exact
- decision metric(s): **dissimilarity_index** — only these trigger a top-up
- gap floor: 0.01 DI (pairs below it are `≈`, never priced)
- chance level on this board (20x20, 160+160; 20000 random allocations): DI 0.1254 (SD 0.033); DI at chance (no segregation): none
- other metrics: floor = 0.3 x chance SD (the DI ratio), polarity-aware: clusters 2.55, switch_rate 0.00598, distance 0.008, mix_deviation 0.00301, share 0.00484, ghetto_rate 1.19

Every chain reads MOST segregated first (clusters and switch_rate are sign-corrected: fewer clusters / lower switch rate = more segregated).

Legend: `>` certified · `?>` fixable (GPU top-up resolves) · `n>` needs more RUNS, not sampling (ruler already fine, gap under-measured) · `~` unresolved · `=` exact tie · `≈` below the practical-significance floor · `°` at the chance level (no more segregated than a random grid; marked as a tie)

## dissimilarity_index  **(DECISION METRIC)**

```
green ≈ politics > baseline > ethnic > race > income
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 0.7377 | 0.000722 | +0.6123 |  |
| politics | 0.735 | 0.000823 | +0.6096 |  |
| baseline | 0.7184 | 0.000627 | +0.593 |  |
| ethnic | 0.6006 | 0.000948 | +0.4752 |  |
| race | 0.571 | 0.000907 | +0.4456 |  |
| income | 0.4583 | 0.000833 | +0.3329 |  |

## clusters

```
politics > green > ethnic ≈ baseline ≈ race > income
```
counts: {'CERTIFIED': 3, 'FLOOR-TIE': 2}

chance 96.12 (SD 8.5), floor 2.55 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 5.511 | 0.0192 | +90.61 |  |
| green | 8.944 | 0.03 | +87.18 |  |
| ethnic | 13.86 | 0.038 | +82.26 |  |
| baseline | 15.57 | 0.0429 | +80.55 |  |
| race | 17.93 | 0.044 | +78.19 |  |
| income | 37.55 | 0.0624 | +58.57 |  |

## switch_rate

```
politics > green > baseline > ethnic > race > income
```
counts: {'CERTIFIED': 5}

chance 0.502 (SD 0.0199), floor 0.00598 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.0427 | 0.000168 | +0.4593 |  |
| green | 0.0503 | 0.000185 | +0.4517 |  |
| baseline | 0.08586 | 0.000244 | +0.4161 |  |
| ethnic | 0.191 | 0.000353 | +0.311 |  |
| race | 0.2227 | 0.000368 | +0.2793 |  |
| income | 0.2883 | 0.000345 | +0.2137 |  |

## distance

```
politics > green > baseline > ethnic > race > income
```
counts: {'CERTIFIED': 5}

chance 1.151 (SD 0.0267), floor 0.008

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 4.726 | 0.00816 | +3.575 |  |
| green | 4.58 | 0.00753 | +3.429 |  |
| baseline | 3.648 | 0.00594 | +2.497 |  |
| ethnic | 3.016 | 0.00512 | +1.865 |  |
| race | 2.564 | 0.0038 | +1.413 |  |
| income | 1.821 | 0.00162 | +0.6697 |  |

## mix_deviation

```
politics ≈ green > baseline > ethnic > race > income
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.1678 (SD 0.01), floor 0.00301

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.4663 | 0.000104 | +0.2986 |  |
| green | 0.4641 | 0.000108 | +0.2963 |  |
| baseline | 0.4441 | 0.000138 | +0.2763 |  |
| ethnic | 0.3916 | 0.000201 | +0.2239 |  |
| race | 0.374 | 0.000213 | +0.2062 |  |
| income | 0.3263 | 0.000209 | +0.1585 |  |

## share

```
politics > green > baseline > ethnic > race > income
```
counts: {'CERTIFIED': 5}

chance 0.4983 (SD 0.0161), floor 0.00484

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.9654 | 0.000121 | +0.4672 |  |
| green | 0.9556 | 0.000155 | +0.4573 |  |
| baseline | 0.9223 | 0.000214 | +0.424 |  |
| ethnic | 0.8619 | 0.000249 | +0.3636 |  |
| race | 0.8279 | 0.000276 | +0.3297 |  |
| income | 0.7478 | 0.000287 | +0.2495 |  |

## ghetto_rate

```
politics > green > baseline > ethnic > race > income
```
counts: {'CERTIFIED': 5}

chance 9.089 (SD 3.95), floor 1.19

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 278.9 | 0.128 | +269.8 |  |
| green | 269.8 | 0.158 | +260.7 |  |
| baseline | 238.4 | 0.2 | +229.4 |  |
| ethnic | 197.6 | 0.211 | +188.5 |  |
| race | 171 | 0.218 | +161.9 |  |
| income | 107.5 | 0.19 | +98.44 |  |
