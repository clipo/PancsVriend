# Rank stability — llama-3.3-70b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260912_200213_llama-3.3-70b-vf-lp/experiments, n=[10000])
- ruler: None
- ruler rng_scheme: exact
- decision metric(s): **dissimilarity_index** — only these trigger a top-up
- gap floor: 0.01 DI (pairs below it are `≈`, never priced)
- chance level on this board (20x20, 160+160; 20000 random allocations): DI 0.1254 (SD 0.033); DI at chance (no segregation): ethnic
- other metrics: floor = 0.3 x chance SD (the DI ratio), polarity-aware: clusters 2.55, switch_rate 0.00598, distance 0.008, mix_deviation 0.00301, share 0.00484, ghetto_rate 1.19

Every chain reads MOST segregated first (clusters and switch_rate are sign-corrected: fewer clusters / lower switch rate = more segregated).

Legend: `>` certified · `?>` fixable (GPU top-up resolves) · `n>` needs more RUNS, not sampling (ruler already fine, gap under-measured) · `~` unresolved · `=` exact tie · `≈` below the practical-significance floor · `°` at the chance level (no more segregated than a random grid; marked as a tie)

## dissimilarity_index  **(DECISION METRIC)**

```
baseline > politics > income > green > race ≈ ethnic°
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.3423 | 0.000855 | +0.2168 |  |
| politics | 0.3263 | 0.000825 | +0.2009 |  |
| income | 0.2998 | 0.000794 | +0.1743 |  |
| green | 0.1757 | 0.000486 | +0.05031 |  |
| race | 0.139 | 0.000374 | +0.01361 |  |
| ethnic | 0.1333 | 0.000358 | +0.00792 | ° |

## clusters

```
baseline > politics > income > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 96.12 (SD 8.5), floor 2.55 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 16.68 | 0.0392 | +79.44 |  |
| politics | 19.92 | 0.046 | +76.2 |  |
| income | 29.51 | 0.0619 | +66.61 |  |
| green | 66.45 | 0.104 | +29.67 |  |
| race | 84.6 | 0.0927 | +11.52 |  |
| ethnic | 89.25 | 0.091 | +6.867 |  |

## switch_rate

```
baseline > politics > income > green > race° ≈ ethnic°
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.502 (SD 0.0199), floor 0.00598 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.3043 | 0.000384 | +0.1976 |  |
| politics | 0.3235 | 0.000389 | +0.1784 |  |
| income | 0.3638 | 0.000398 | +0.1381 |  |
| green | 0.4848 | 0.000254 | +0.01714 |  |
| race | 0.5013 | 0.000208 | +0.0006381 | ° |
| ethnic | 0.5019 | 0.000203 | +3.076e-05 | ° |

## distance

```
baseline > politics > income > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 1.151 (SD 0.0267), floor 0.008

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 2.048 | 0.00221 | +0.8971 |  |
| politics | 1.962 | 0.00204 | +0.8116 |  |
| income | 1.805 | 0.00184 | +0.6546 |  |
| green | 1.323 | 0.000719 | +0.172 |  |
| race | 1.204 | 0.000385 | +0.05272 |  |
| ethnic | 1.182 | 0.000342 | +0.0313 |  |

## mix_deviation

```
baseline > politics > income > green > race° ≈ ethnic°
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.1678 (SD 0.01), floor 0.00301

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.3175 | 0.000223 | +0.1498 |  |
| politics | 0.3057 | 0.00023 | +0.138 |  |
| income | 0.2828 | 0.000245 | +0.115 |  |
| green | 0.1895 | 0.00017 | +0.02171 |  |
| race | 0.1691 | 0.000114 | +0.001347 | ° |
| ethnic | 0.1686 | 0.000108 | +0.0008509 | ° |

## share

```
baseline > politics > income > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 0.4983 (SD 0.0161), floor 0.00484

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.7961 | 0.000255 | +0.2978 |  |
| politics | 0.7808 | 0.000266 | +0.2825 |  |
| income | 0.7417 | 0.000305 | +0.2435 |  |
| green | 0.5751 | 0.000258 | +0.07682 |  |
| race | 0.5201 | 0.000184 | +0.02188 |  |
| ethnic | 0.5117 | 0.000178 | +0.01344 |  |

## ghetto_rate

```
baseline > politics > income > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 9.089 (SD 3.95), floor 1.19

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 138.7 | 0.201 | +129.6 |  |
| politics | 128.4 | 0.202 | +119.3 |  |
| income | 106.3 | 0.204 | +97.24 |  |
| green | 36.32 | 0.111 | +27.23 |  |
| race | 17.53 | 0.0608 | +8.445 |  |
| ethnic | 14.1 | 0.0535 | +5.01 |  |
