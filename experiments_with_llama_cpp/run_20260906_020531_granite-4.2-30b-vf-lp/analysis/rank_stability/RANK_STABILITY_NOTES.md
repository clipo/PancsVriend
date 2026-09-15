# Rank stability — granite-4.2-30b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_020531_granite-4.2-30b-vf-lp/experiments, n=[10000])
- ruler: None
- ruler rng_scheme: exact
- decision metric(s): **dissimilarity_index** — only these trigger a top-up
- gap floor: 0.01 DI (pairs below it are `≈`, never priced)
- chance level on this board (20x20, 160+160; 2000 random allocations): DI 0.1254 (SD 0.033); DI at chance (no segregation): baseline, ethnic, green, income, politics, race
- other metrics: floor = 0.3 x chance SD (the DI ratio), polarity-aware: clusters 2.58, switch_rate 0.00606, distance 0.00804, mix_deviation 0.003, share 0.00486, ghetto_rate 1.18

Every chain reads MOST segregated first (clusters and switch_rate are sign-corrected: fewer clusters / lower switch rate = more segregated).

Legend: `>` certified · `?>` fixable (GPU top-up resolves) · `n>` needs more RUNS, not sampling (ruler already fine, gap under-measured) · `~` unresolved · `=` exact tie · `≈` below the practical-significance floor · `°` at the chance level (no more segregated than a random grid; marked as a tie)

## dissimilarity_index  **(DECISION METRIC)**

```
baseline° ≈ politics° ≈ income° ≈ race° ≈ green° ≈ ethnic°
```
counts: {'FLOOR-TIE': 5}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.1327 | 0.000359 | +0.007307 | ° |
| politics | 0.1324 | 0.000357 | +0.006935 | ° |
| income | 0.1267 | 0.00034 | +0.001233 | ° |
| race | 0.125 | 0.000334 | -0.0004316 | ° |
| green | 0.125 | 0.000333 | -0.0004503 | ° |
| ethnic | 0.1249 | 0.000333 | -0.0004722 | ° |

## clusters

```
politics ≈ baseline > income° ≈ race° ≈ green° ≈ ethnic°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 96.21 (SD 8.61), floor 2.58 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 90.19 | 0.0909 | +6.017 |  |
| baseline | 90.2 | 0.0916 | +6.013 |  |
| income | 94.89 | 0.0865 | +1.32 | ° |
| race | 96.05 | 0.0844 | +0.1612 | ° |
| green | 96.06 | 0.0844 | +0.1492 | ° |
| ethnic | 96.07 | 0.0844 | +0.136 | ° |

## switch_rate

```
ethnic° ≈ race° ≈ green° ≈ income° ≈ politics° ≈ baseline°
```
counts: {'FLOOR-TIE': 5}

chance 0.5016 (SD 0.0202), floor 0.00606 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| ethnic | 0.5017 | 0.0002 | -0.0001289 | ° |
| race | 0.5017 | 0.0002 | -0.0001316 | ° |
| green | 0.5017 | 0.0002 | -0.0001347 | ° |
| income | 0.5019 | 0.000201 | -0.0003009 | ° |
| politics | 0.502 | 0.000204 | -0.0004356 | ° |
| baseline | 0.502 | 0.000204 | -0.0004661 | ° |

## distance

```
baseline ≈ politics > income° ≈ race° ≈ green° ≈ ethnic°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 1.152 (SD 0.0268), floor 0.00804

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 1.179 | 0.000342 | +0.02777 |  |
| politics | 1.179 | 0.000342 | +0.02729 |  |
| income | 1.157 | 0.000288 | +0.005609 | ° |
| race | 1.151 | 0.000268 | -0.0003625 | ° |
| green | 1.151 | 0.000268 | -0.0004203 | ° |
| ethnic | 1.151 | 0.000268 | -0.0004859 | ° |

## mix_deviation

```
baseline° ≈ politics° ≈ income° ≈ race° ≈ green° ≈ ethnic°
```
counts: {'FLOOR-TIE': 5}

chance 0.168 (SD 0.00999), floor 0.003

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.169 | 0.000108 | +0.0009509 | ° |
| politics | 0.1688 | 0.000108 | +0.0007515 | ° |
| income | 0.1683 | 0.000102 | +0.0002547 | ° |
| race | 0.1679 | 0.0001 | -0.000136 | ° |
| green | 0.1679 | 0.0001 | -0.0001399 | ° |
| ethnic | 0.1679 | 0.0001 | -0.000143 | ° |

## share

```
baseline ≈ politics > income° ≈ race° ≈ green° ≈ ethnic°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 0.4989 (SD 0.0162), floor 0.00486

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.5107 | 0.000178 | +0.01187 |  |
| politics | 0.5106 | 0.000179 | +0.01172 |  |
| income | 0.5013 | 0.000168 | +0.002433 | ° |
| race | 0.4985 | 0.000163 | -0.0004103 | ° |
| green | 0.4984 | 0.000163 | -0.0004363 | ° |
| ethnic | 0.4984 | 0.000163 | -0.0004718 | ° |

## ghetto_rate

```
baseline ≈ politics > income° ≈ race° ≈ green° ≈ ethnic°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 9.222 (SD 3.93), floor 1.18

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 13.68 | 0.0536 | +4.462 |  |
| politics | 13.58 | 0.0539 | +4.358 |  |
| income | 10.11 | 0.0438 | +0.8899 | ° |
| race | 9.164 | 0.04 | -0.0575 | ° |
| green | 9.154 | 0.0401 | -0.0674 | ° |
| ethnic | 9.144 | 0.04 | -0.0774 | ° |
