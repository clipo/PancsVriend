# Rank stability — deepseek-v4-flash-chat-grammar-lp

- gaps: production run_summary finals (experiments_with_llama_cpp/run_20260906_092955_deepseek-v4-flash-vf-lp/experiments, n=[10000])
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
politics > baseline > income > race > ethnic > green
```
counts: {'CERTIFIED': 5}

chance 0.1254 (SD 0.0334), floor 0.01

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.4216 | 0.000974 | +0.2962 |  |
| baseline | 0.3743 | 0.000911 | +0.2488 |  |
| income | 0.3081 | 0.0008 | +0.1827 |  |
| race | 0.2556 | 0.000655 | +0.1302 |  |
| ethnic | 0.173 | 0.000464 | +0.04757 |  |
| green | 0.1407 | 0.000384 | +0.01525 |  |

## clusters

```
politics > baseline > income > race > ethnic > green
```
counts: {'CERTIFIED': 5}

chance 96.21 (SD 8.61), floor 2.58 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 10.54 | 0.0249 | +85.67 |  |
| baseline | 14.03 | 0.0337 | +82.18 |  |
| income | 31.14 | 0.0608 | +65.07 |  |
| race | 52.93 | 0.088 | +43.28 |  |
| ethnic | 66.78 | 0.0914 | +29.43 |  |
| green | 85.47 | 0.0975 | +10.74 |  |

## switch_rate

```
politics > baseline > income > race > ethnic > green°
```
counts: {'CERTIFIED': 5}

chance 0.5016 (SD 0.0202), floor 0.00606 — higher = LESS segregation; excess is sign-corrected

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.182 | 0.000319 | +0.3196 |  |
| baseline | 0.2498 | 0.000377 | +0.2517 |  |
| income | 0.3504 | 0.000387 | +0.1512 |  |
| race | 0.4049 | 0.000322 | +0.09663 |  |
| ethnic | 0.4864 | 0.000229 | +0.01517 |  |
| green | 0.4995 | 0.000214 | +0.002086 | ° |

## distance

```
politics > baseline > income > race > ethnic > green
```
counts: {'CERTIFIED': 5}

chance 1.152 (SD 0.0268), floor 0.00804

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 2.493 | 0.00299 | +1.342 |  |
| baseline | 2.219 | 0.00253 | +1.067 |  |
| income | 1.844 | 0.00195 | +0.6928 |  |
| race | 1.501 | 0.00109 | +0.3488 |  |
| ethnic | 1.312 | 0.000612 | +0.1608 |  |
| green | 1.203 | 0.000411 | +0.05175 |  |

## mix_deviation

```
politics > baseline > income > race > ethnic > green
```
counts: {'CERTIFIED': 5}

chance 0.168 (SD 0.00999), floor 0.003

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.3793 | 0.000182 | +0.2113 |  |
| baseline | 0.3444 | 0.000215 | +0.1763 |  |
| income | 0.2917 | 0.000236 | +0.1236 |  |
| race | 0.2533 | 0.000208 | +0.08523 |  |
| ethnic | 0.1871 | 0.00015 | +0.01911 |  |
| green | 0.1721 | 0.000119 | +0.004042 |  |

## share

```
politics > baseline > income > race > ethnic > green
```
counts: {'CERTIFIED': 5}

chance 0.4989 (SD 0.0162), floor 0.00486

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 0.8769 | 0.000197 | +0.3781 |  |
| baseline | 0.8328 | 0.000243 | +0.3339 |  |
| income | 0.7491 | 0.000301 | +0.2502 |  |
| race | 0.6702 | 0.000309 | +0.1714 |  |
| ethnic | 0.566 | 0.000214 | +0.06711 |  |
| green | 0.5233 | 0.0002 | +0.02445 |  |

## ghetto_rate

```
politics > baseline > income > race > ethnic > green
```
counts: {'CERTIFIED': 5}

chance 9.222 (SD 3.93), floor 1.18

| scenario | mean | SE | excess over chance | at chance |
|---|---|---|---|---|
| politics | 192.7 | 0.183 | +183.5 |  |
| baseline | 160.9 | 0.203 | +151.7 |  |
| income | 109.8 | 0.205 | +100.6 |  |
| race | 64.12 | 0.164 | +54.9 |  |
| ethnic | 34.15 | 0.0911 | +24.92 |  |
| green | 17.32 | 0.0661 | +8.1 |  |
