# Rank stability — gemma-4-31b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_003730_gemma-4-31b-vf-lp/experiments, n=[10000])
- ruler: None
- ruler rng_scheme: exact
- decision metric(s): **dissimilarity_index** — only these trigger a top-up
- gap floor: 0.01 DI (pairs below it are `≈`, never priced)
- chance level on this board (20x20, 160+160; 2000 random allocations): DI 0.1254 (SD 0.033); DI at chance (no segregation): ethnic, race
- other metrics: floor = 0.3 x chance SD (the DI ratio), polarity-aware: clusters 2.58, switch_rate 0.00606, distance 0.00804, mix_deviation 0.003, share 0.00486, ghetto_rate 1.18

Every chain reads MOST segregated first (clusters and switch_rate are sign-corrected: fewer clusters / lower switch rate = more segregated).

Legend: `>` certified · `?>` fixable (GPU top-up resolves) · `n>` needs more RUNS, not sampling (ruler already fine, gap under-measured) · `~` unresolved · `=` exact tie · `≈` below the practical-significance floor · `°` at the chance level (no more segregated than a random grid; marked as a tie)

## dissimilarity_index  **(DECISION METRIC)**

```
green > income ≈ politics ≈ baseline > race° ≈ ethnic°
```
counts: {'FLOOR-TIE': 3, 'CERTIFIED': 2}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 0.2557 | 0.000691 | +0.1303 |  |
| income | 0.2181 | 0.000574 | +0.0927 |  |
| politics | 0.2084 | 0.000575 | +0.08299 |  |
| baseline | 0.1997 | 0.000539 | +0.07424 |  |
| race | 0.1346 | 0.000363 | +0.009152 | ° |
| ethnic | 0.1299 | 0.000348 | +0.00446 | ° |

## clusters

```
green > politics ≈ income > baseline > race > ethnic
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 96.21 (SD 8.61), floor 2.58 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 36.03 | 0.0785 | +60.18 |  |
| politics | 53.44 | 0.101 | +42.77 |  |
| income | 53.73 | 0.0739 | +42.48 |  |
| baseline | 57.53 | 0.0895 | +38.68 |  |
| race | 88.71 | 0.0917 | +7.5 |  |
| ethnic | 92.36 | 0.0882 | +3.845 |  |

## switch_rate

```
green > income > politics > baseline > ethnic° ≈ race°
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.5016 (SD 0.0202), floor 0.00606 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 0.4152 | 0.000362 | +0.08636 |  |
| income | 0.4513 | 0.000266 | +0.05025 |  |
| politics | 0.4609 | 0.000306 | +0.0407 |  |
| baseline | 0.4681 | 0.00027 | +0.0335 |  |
| ethnic | 0.502 | 0.000202 | -0.000412 | ° |
| race | 0.502 | 0.000205 | -0.0004699 | ° |

## distance

```
green > income > politics > baseline > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 1.152 (SD 0.0268), floor 0.00804

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 1.63 | 0.00139 | +0.478 |  |
| income | 1.459 | 0.000857 | +0.3078 |  |
| politics | 1.435 | 0.000975 | +0.2828 |  |
| baseline | 1.402 | 0.000836 | +0.2508 |  |
| race | 1.187 | 0.000354 | +0.03527 |  |
| ethnic | 1.169 | 0.00031 | +0.0175 |  |

## mix_deviation

```
green > income > politics > baseline > race° ≈ ethnic°
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.168 (SD 0.00999), floor 0.003

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 0.2485 | 0.000234 | +0.08041 |  |
| income | 0.2227 | 0.000173 | +0.05462 |  |
| politics | 0.2122 | 0.000206 | +0.04418 |  |
| baseline | 0.2067 | 0.000182 | +0.0387 |  |
| race | 0.1693 | 0.000109 | +0.001229 | ° |
| ethnic | 0.1686 | 0.000104 | +0.0005881 | ° |

## share

```
green > income > politics > baseline > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 0.4989 (SD 0.0162), floor 0.00486

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 0.6907 | 0.000303 | +0.1919 |  |
| income | 0.6318 | 0.000228 | +0.133 |  |
| politics | 0.6252 | 0.000293 | +0.1263 |  |
| baseline | 0.6077 | 0.000253 | +0.1088 |  |
| race | 0.5139 | 0.000181 | +0.01507 |  |
| ethnic | 0.5063 | 0.000171 | +0.007414 |  |

## ghetto_rate

```
green > income > politics > baseline > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 9.222 (SD 3.93), floor 1.18

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| green | 82.64 | 0.179 | +73.42 |  |
| income | 57.83 | 0.115 | +48.6 |  |
| politics | 54.03 | 0.146 | +44.81 |  |
| baseline | 48.51 | 0.121 | +39.29 |  |
| race | 14.88 | 0.0561 | +5.66 |  |
| ethnic | 12.03 | 0.0478 | +2.808 |  |
