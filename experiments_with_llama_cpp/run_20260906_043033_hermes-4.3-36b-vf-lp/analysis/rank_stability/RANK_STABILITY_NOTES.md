# Rank stability — hermes-4.3-36b-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_043033_hermes-4.3-36b-vf-lp/experiments, n=[10000])
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
income ≈ politics > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 4, 'FLOOR-TIE': 1}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| income | 0.4842 | 0.00105 | +0.3588 |  |
| politics | 0.4788 | 0.00103 | +0.3534 |  |
| baseline | 0.4515 | 0.000997 | +0.3261 |  |
| green | 0.3886 | 0.000929 | +0.2632 |  |
| race | 0.2598 | 0.000704 | +0.1343 |  |
| ethnic | 0.1618 | 0.000437 | +0.03636 |  |

## clusters

```
politics ≈ baseline ≈ income ≈ green > race > ethnic
```
counts: {'FLOOR-TIE': 3, 'CERTIFIED': 2}

chance 96.21 (SD 8.61), floor 2.58 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 7.867 | 0.0187 | +88.34 |  |
| baseline | 9.202 | 0.023 | +87.01 |  |
| income | 10 | 0.0257 | +86.21 |  |
| green | 11.98 | 0.0298 | +84.23 |  |
| race | 35.84 | 0.0819 | +60.37 |  |
| ethnic | 72.84 | 0.0964 | +23.37 |  |

## switch_rate

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 0.5016 (SD 0.0202), floor 0.00606 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.1128 | 0.000261 | +0.3888 |  |
| income | 0.1493 | 0.000314 | +0.3523 |  |
| baseline | 0.1574 | 0.000305 | +0.3442 |  |
| green | 0.2365 | 0.00037 | +0.265 |  |
| race | 0.4101 | 0.000374 | +0.09142 |  |
| ethnic | 0.4922 | 0.000226 | +0.009399 |  |

## distance

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 1.152 (SD 0.0268), floor 0.00804

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 2.86 | 0.00369 | +1.708 |  |
| income | 2.832 | 0.00426 | +1.681 |  |
| baseline | 2.671 | 0.00348 | +1.519 |  |
| green | 2.306 | 0.00272 | +1.155 |  |
| race | 1.642 | 0.00146 | +0.4902 |  |
| ethnic | 1.275 | 0.000552 | +0.123 |  |

## mix_deviation

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 0.168 (SD 0.00999), floor 0.003

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.4158 | 0.000152 | +0.2477 |  |
| income | 0.403 | 0.000179 | +0.235 |  |
| baseline | 0.3951 | 0.000173 | +0.227 |  |
| green | 0.353 | 0.000209 | +0.185 |  |
| race | 0.2508 | 0.000244 | +0.08271 |  |
| ethnic | 0.1806 | 0.000142 | +0.01257 |  |

## share

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 0.4989 (SD 0.0162), floor 0.00486

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.9177 | 0.000163 | +0.4188 |  |
| income | 0.898 | 0.000198 | +0.3991 |  |
| baseline | 0.892 | 0.000189 | +0.3931 |  |
| green | 0.8431 | 0.000233 | +0.3442 |  |
| race | 0.6921 | 0.00033 | +0.1933 |  |
| ethnic | 0.553 | 0.000214 | +0.05412 |  |

## ghetto_rate

```
politics > income > baseline > green > race > ethnic
```
counts: {'CERTIFIED': 5}

chance 9.222 (SD 3.93), floor 1.18

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 225.3 | 0.162 | +216 |  |
| income | 213.4 | 0.189 | +204.2 |  |
| baseline | 206 | 0.179 | +196.7 |  |
| green | 170 | 0.2 | +160.8 |  |
| race | 84.09 | 0.188 | +74.86 |  |
| ethnic | 28.79 | 0.0865 | +19.57 |  |
