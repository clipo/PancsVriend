# Segregation Ranking by Scenario (qwen3.6-27b-vf-lp)

Each scenario is compared with the NEXT-ranked one with a paired
t-test on the per-run differences (runs matched by run_id: the same
seed, hence the same initial grid, in every scenario), and with its
own INITIAL grids — a random allocation, i.e. chance — by the same
paired t (final - initial). 'Excess' is the mean final - initial in
the segregating direction; a scenario not significantly above its
initial grids shows no segregation on that metric.
Significance levels: `***` p<0.001, `**` p<0.01, `*` p<0.05.

Metrics are ordered with dissimilarity first when available.

## dissimilarity_index

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.3789 | 0.0911 | 10000 | *** | 0.000000 | paired t | +0.2540 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.3554 | 0.0888 | 10000 |  | 0.197442 | paired t | +0.2304 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.3542 | 0.0877 | 10000 | *** | 0.000000 | paired t | +0.2292 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.3455 | 0.0860 | 10000 | *** | 0.000000 | paired t | +0.2206 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.2988 | 0.0783 | 10000 | *** | 0.000000 | paired t | +0.1739 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1564 | 0.0427 | 10000 |  |  |  | +0.0315 | *** | 0.000000 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 76.2108 | 9.9224 | 10000 | *** | 0.000000 | paired t | +19.8645 | *** | 0.000000 |
| 2 | Racial (White/Black) | 27.3789 | 6.4236 | 10000 | *** | 0.000000 | paired t | +68.6964 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 18.3911 | 4.3294 | 10000 | *** | 0.000000 | paired t | +77.6842 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 17.7528 | 4.2863 | 10000 | *** | 0.000295 | paired t | +78.3225 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 17.5542 | 4.2374 | 10000 | *** | 0.000000 | paired t | +78.5211 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 13.6150 | 3.3400 | 10000 |  |  |  | +82.4603 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 0.4939 | 0.0225 | 10000 | *** | 0.000000 | paired t | +0.0078 | *** | 0.000000 |
| 2 | Racial (White/Black) | 0.3601 | 0.0413 | 10000 | *** | 0.000000 | paired t | +0.1416 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 0.2990 | 0.0401 | 10000 | *** | 0.000000 | paired t | +0.2027 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.2897 | 0.0397 | 10000 | *** | 0.000000 | paired t | +0.2120 | *** | 0.000000 |
| 5 | Economic (High/Low Income) | 0.2859 | 0.0406 | 10000 | *** | 0.000000 | paired t | +0.2158 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 0.2482 | 0.0383 | 10000 |  |  |  | +0.2535 | *** | 0.000000 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 2.2426 | 0.2588 | 10000 | *** | 0.000000 | paired t | +1.0914 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 2.1029 | 0.2371 | 10000 | *** | 0.000045 | paired t | +0.9517 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 2.0920 | 0.2372 | 10000 | *** | 0.000000 | paired t | +0.9408 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 2.0558 | 0.2250 | 10000 | *** | 0.000000 | paired t | +0.9046 | *** | 0.000000 |
| 5 | Racial (White/Black) | 1.8140 | 0.1834 | 10000 | *** | 0.000000 | paired t | +0.6628 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 1.2563 | 0.0526 | 10000 |  |  |  | +0.1052 | *** | 0.000000 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.3458 | 0.0216 | 10000 | *** | 0.000000 | paired t | +0.1779 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.3264 | 0.0233 | 10000 | *** | 0.000000 | paired t | +0.1585 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.3248 | 0.0228 | 10000 | *** | 0.000000 | paired t | +0.1569 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.3189 | 0.0233 | 10000 | *** | 0.000000 | paired t | +0.1510 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.2829 | 0.0252 | 10000 | *** | 0.000000 | paired t | +0.1150 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1784 | 0.0138 | 10000 |  |  |  | +0.0105 | *** | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.8347 | 0.0244 | 10000 | *** | 0.000000 | paired t | +0.3363 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.8064 | 0.0275 | 10000 | *** | 0.000000 | paired t | +0.3080 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.8045 | 0.0266 | 10000 | *** | 0.000000 | paired t | +0.3061 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.7981 | 0.0273 | 10000 | *** | 0.000000 | paired t | +0.2997 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.7444 | 0.0322 | 10000 | *** | 0.000000 | paired t | +0.2460 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.5463 | 0.0216 | 10000 |  |  |  | +0.0479 | *** | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 163.0221 | 20.3261 | 10000 | *** | 0.000000 | paired t | +153.8799 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 144.6585 | 21.2120 | 10000 | *** | 0.000000 | paired t | +135.5163 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 143.3094 | 20.7984 | 10000 | *** | 0.000000 | paired t | +134.1672 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 139.6437 | 20.8384 | 10000 | *** | 0.000000 | paired t | +130.5015 | *** | 0.000000 |
| 5 | Racial (White/Black) | 108.2628 | 20.9510 | 10000 | *** | 0.000000 | paired t | +99.1206 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 25.7831 | 8.4122 | 10000 |  |  |  | +16.6409 | *** | 0.000000 |
