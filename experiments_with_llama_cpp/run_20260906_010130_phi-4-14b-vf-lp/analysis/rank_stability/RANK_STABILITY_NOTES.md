# Rank stability — phi-4-14b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_010130_phi-4-14b-vf-lp/experiments, n=[10000])
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
baseline° ≈ politics° ≈ green° ≈ income° ≈ race° = ethnic°
```
counts: {'FLOOR-TIE': 4, 'EXACT-TIE': 1}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.1317 | 0.000356 | +0.006285 | ° |
| politics | 0.1291 | 0.000347 | +0.00367 | ° |
| green | 0.1271 | 0.000341 | +0.001638 | ° |
| income | 0.125 | 0.000333 | -0.0004491 | ° |
| race | 0.1249 | 0.000333 | -0.0004722 | ° |
| ethnic | 0.1249 | 0.000333 | -0.0004722 | ° |

## clusters

```
baseline ≈ politics ≈ green° ≈ income° ≈ race° ≈ ethnic°
```
counts: {'FLOOR-TIE': 5}

chance 96.21 (SD 8.61), floor 2.58 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 91.01 | 0.0904 | +5.203 |  |
| politics | 92.95 | 0.088 | +3.255 |  |
| green | 94.73 | 0.0871 | +1.48 | ° |
| income | 96.06 | 0.0844 | +0.15 | ° |
| race | 96.07 | 0.0844 | +0.1351 | ° |
| ethnic | 96.08 | 0.0844 | +0.1345 | ° |

## switch_rate

```
ethnic° ≈ race° ≈ income° ≈ green° ≈ baseline° ≈ politics°
```
counts: {'FLOOR-TIE': 5}

chance 0.5016 (SD 0.0202), floor 0.00606 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| ethnic | 0.5017 | 0.0002 | -0.0001289 | ° |
| race | 0.5017 | 0.0002 | -0.000129 | ° |
| income | 0.5017 | 0.0002 | -0.0001333 | ° |
| green | 0.5019 | 0.000201 | -0.0003684 | ° |
| baseline | 0.502 | 0.000203 | -0.0004137 | ° |
| politics | 0.5021 | 0.000202 | -0.0005129 | ° |

## distance

```
baseline > politics ≈ green° ≈ income° ≈ race° ≈ ethnic°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 1.152 (SD 0.0268), floor 0.00804

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 1.176 | 0.000332 | +0.02396 |  |
| politics | 1.166 | 0.000307 | +0.01475 |  |
| green | 1.158 | 0.000293 | +0.006792 | ° |
| income | 1.151 | 0.000268 | -0.0004131 | ° |
| race | 1.151 | 0.000268 | -0.0004912 | ° |
| ethnic | 1.151 | 0.000267 | -0.0004966 | ° |

## mix_deviation

```
baseline° ≈ politics° ≈ green° ≈ income° ≈ race° ≈ ethnic°
```
counts: {'FLOOR-TIE': 5}

chance 0.168 (SD 0.00999), floor 0.003

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.1688 | 0.000107 | +0.0007637 | ° |
| politics | 0.1686 | 0.000105 | +0.0005296 | ° |
| green | 0.1685 | 0.000103 | +0.0004409 | ° |
| income | 0.1679 | 0.0001 | -0.0001361 | ° |
| race | 0.1679 | 0.0001 | -0.0001426 | ° |
| ethnic | 0.1679 | 0.0001 | -0.0001433 | ° |

## share

```
baseline ≈ politics ≈ green° ≈ income° ≈ race° ≈ ethnic°
```
counts: {'FLOOR-TIE': 5}

chance 0.4989 (SD 0.0162), floor 0.00486

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 0.509 | 0.000177 | +0.01018 |  |
| politics | 0.5053 | 0.000172 | +0.006386 |  |
| green | 0.5019 | 0.000171 | +0.003068 | ° |
| income | 0.4984 | 0.000163 | -0.0004358 | ° |
| race | 0.4984 | 0.000163 | -0.0004738 | ° |
| ethnic | 0.4984 | 0.000163 | -0.0004753 | ° |

## ghetto_rate

```
baseline > politics > green° ≈ income° ≈ race° ≈ ethnic°
```
counts: {'FLOOR-TIE': 3, 'CERTIFIED': 2}

chance 9.222 (SD 3.93), floor 1.18

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| baseline | 13.06 | 0.0518 | +3.843 |  |
| politics | 11.58 | 0.0475 | +2.356 |  |
| green | 10.3 | 0.0447 | +1.076 | ° |
| income | 9.157 | 0.04 | -0.0649 | ° |
| race | 9.143 | 0.04 | -0.0784 | ° |
| ethnic | 9.143 | 0.04 | -0.079 | ° |
