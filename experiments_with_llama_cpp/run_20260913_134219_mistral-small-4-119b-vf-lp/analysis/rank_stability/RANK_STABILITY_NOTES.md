# Rank stability — mistral-small-4-119b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260913_134219_mistral-small-4-119b-vf-lp/experiments, n=[10000])
- ruler: None
- ruler rng_scheme: exact
- decision metric(s): **dissimilarity_index** — only these trigger a top-up
- gap floor: 0.01 DI (pairs below it are `≈`, never priced)
- chance level on this board (20x20, 160+160; 20000 random allocations): DI 0.1254 (SD 0.033); DI at chance (no segregation): baseline, ethnic, green, income, politics, race
- other metrics: floor = 0.3 x chance SD (the DI ratio), polarity-aware: clusters 2.55, switch_rate 0.00598, distance 0.008, mix_deviation 0.00301, share 0.00484, ghetto_rate 1.19

Every chain reads MOST segregated first (clusters and switch_rate are sign-corrected: fewer clusters / lower switch rate = more segregated).

Legend: `>` certified · `?>` fixable (GPU top-up resolves) · `n>` needs more RUNS, not sampling (ruler already fine, gap under-measured) · `~` unresolved · `=` exact tie · `≈` below the practical-significance floor · `°` at the chance level (no more segregated than a random grid; marked as a tie)

## dissimilarity_index  **(DECISION METRIC)**

```
politics° ≈ income° ≈ ethnic° ≈ baseline° ≈ green° ≈ race°
```
counts: {'FLOOR-TIE': 5}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.1294 | 0.000348 | +0.003947 | ° |
| income | 0.1261 | 0.000337 | +0.0006591 | ° |
| ethnic | 0.1252 | 0.000334 | -0.0001884 | ° |
| baseline | 0.1252 | 0.000334 | -0.0002416 | ° |
| green | 0.1249 | 0.000333 | -0.0004716 | ° |
| race | 0.1249 | 0.000333 | -0.0005216 | ° |

## clusters

```
politics ≈ income° ≈ ethnic° ≈ baseline° ≈ race° ≈ green°
```
counts: {'FLOOR-TIE': 5}

chance 96.12 (SD 8.5), floor 2.55 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 92.93 | 0.0884 | +3.192 |  |
| income | 95.35 | 0.0854 | +0.7731 | ° |
| ethnic | 95.87 | 0.0844 | +0.2535 | ° |
| baseline | 95.92 | 0.0847 | +0.2019 | ° |
| race | 95.99 | 0.0845 | +0.1305 | ° |
| green | 96.07 | 0.0844 | +0.04645 | ° |

## switch_rate

```
ethnic° ≈ income° ≈ green° ≈ politics° ≈ baseline° ≈ race°
```
counts: {'FLOOR-TIE': 5}

chance 0.502 (SD 0.0199), floor 0.00598 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| ethnic | 0.5017 | 0.0002 | +0.0002966 | ° |
| income | 0.5017 | 0.000201 | +0.0002858 | ° |
| green | 0.5017 | 0.0002 | +0.0002747 | ° |
| politics | 0.5017 | 0.000202 | +0.0002658 | ° |
| baseline | 0.5017 | 0.0002 | +0.0002422 | ° |
| race | 0.5019 | 0.000199 | +3.002e-05 | ° |

## distance

```
politics > income° ≈ ethnic° ≈ baseline° ≈ race° ≈ green°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 1.151 (SD 0.0267), floor 0.008

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 1.166 | 0.000307 | +0.01471 |  |
| income | 1.154 | 0.000277 | +0.003498 | ° |
| ethnic | 1.152 | 0.000269 | +0.001117 | ° |
| baseline | 1.152 | 0.00027 | +0.001099 | ° |
| race | 1.151 | 0.000268 | +0.0004862 | ° |
| green | 1.151 | 0.000267 | +0.0002962 | ° |

## mix_deviation

```
politics° ≈ income° ≈ baseline° ≈ ethnic° ≈ green° ≈ race°
```
counts: {'FLOOR-TIE': 5}

chance 0.1678 (SD 0.01), floor 0.00301

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.1686 | 0.000105 | +0.0008289 | ° |
| income | 0.1681 | 0.000102 | +0.0003467 | ° |
| baseline | 0.168 | 0.000101 | +0.0002052 | ° |
| ethnic | 0.1679 | 0.0001 | +0.0001955 | ° |
| green | 0.1679 | 0.0001 | +0.0001447 | ° |
| race | 0.1677 | 0.0001 | -2.34e-05 | ° |

## share

```
politics > income° ≈ ethnic° ≈ baseline° ≈ green° ≈ race°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 0.4983 (SD 0.0161), floor 0.00484

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.5054 | 0.000173 | +0.007173 |  |
| income | 0.5001 | 0.000166 | +0.001861 | ° |
| ethnic | 0.4988 | 0.000163 | +0.000552 | ° |
| baseline | 0.4988 | 0.000164 | +0.0005224 | ° |
| green | 0.4984 | 0.000163 | +0.000136 | ° |
| race | 0.4983 | 0.000163 | +5.923e-05 | ° |

## ghetto_rate

```
politics > income° ≈ baseline° ≈ ethnic° ≈ race° ≈ green°
```
counts: {'FLOOR-TIE': 4, 'CERTIFIED': 1}

chance 9.089 (SD 3.95), floor 1.19

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 11.41 | 0.0475 | +2.317 |  |
| income | 9.619 | 0.0417 | +0.53 | ° |
| baseline | 9.27 | 0.0404 | +0.1809 | ° |
| ethnic | 9.267 | 0.0403 | +0.178 | ° |
| race | 9.201 | 0.0402 | +0.1122 | ° |
| green | 9.143 | 0.04 | +0.0546 | ° |
