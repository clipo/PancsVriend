# Segregation Ranking by Scenario (phi-4-14b-vf-lp)

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
| 1 | Color (Red/Blue) | 0.1317 | 0.0356 | 10000 | *** | 0.000000 | paired t | +0.0068 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.1291 | 0.0347 | 10000 | *** | 0.000000 | paired t | +0.0041 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 0.1271 | 0.0341 | 10000 | *** | 0.000000 | paired t | +0.0021 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 0.1250 | 0.0333 | 10000 | *** | 0.000003 | paired t | +0.0000 | *** | 0.000002 |
| 5 | Ethnic (Asian/Hispanic) | 0.1249 | 0.0333 | 10000 |  | 1.000000 | paired t | +0.0000 |  | 0.317335 |
| 6 | Racial (White/Black) | 0.1249 | 0.0333 | 10000 |  |  |  | +0.0000 |  | 0.563729 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 96.0750 | 8.4395 | 10000 |  | 0.108813 | paired t | +0.0003 |  | 0.179726 |
| 2 | Racial (White/Black) | 96.0744 | 8.4397 | 10000 | *** | 0.000000 | paired t | +0.0009 | * | 0.029042 |
| 3 | Economic (High/Low Income) | 96.0595 | 8.4406 | 10000 | *** | 0.000000 | paired t | +0.0158 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 94.7291 | 8.7110 | 10000 | *** | 0.000000 | paired t | +1.3462 | *** | 0.000000 |
| 5 | Political (Liberal/Conservative) | 92.9546 | 8.7953 | 10000 | *** | 0.000000 | paired t | +3.1207 | *** | 0.000000 |
| 6 | Color (Red/Blue) | 91.0063 | 9.0429 | 10000 |  |  |  | +5.0690 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.5021 | 0.0202 | 10000 |  | 0.124187 | paired t | -0.0004 |  | 0.000000 |
| 2 | Color (Red/Blue) | 0.5020 | 0.0203 | 10000 |  | 0.386986 | paired t | -0.0003 |  | 0.000000 |
| 3 | Color (Green/Yellow) | 0.5019 | 0.0201 | 10000 | *** | 0.000000 | paired t | -0.0002 |  | 0.000000 |
| 4 | Economic (High/Low Income) | 0.5017 | 0.0200 | 10000 |  | 0.152445 | paired t | -0.0000 |  | 0.168066 |
| 5 | Racial (White/Black) | 0.5017 | 0.0200 | 10000 |  | 0.851912 | paired t | +0.0000 |  | 0.849803 |
| 6 | Ethnic (Asian/Hispanic) | 0.5017 | 0.0200 | 10000 |  |  |  | +0.0000 |  | 0.313828 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 1.1756 | 0.0332 | 10000 | *** | 0.000000 | paired t | +0.0245 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 1.1664 | 0.0307 | 10000 | *** | 0.000000 | paired t | +0.0152 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 1.1585 | 0.0293 | 10000 | *** | 0.000000 | paired t | +0.0073 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 1.1513 | 0.0268 | 10000 | *** | 0.000000 | paired t | +0.0001 | *** | 0.000000 |
| 5 | Racial (White/Black) | 1.1512 | 0.0268 | 10000 |  | 0.125322 | paired t | +0.0000 | * | 0.042835 |
| 6 | Ethnic (Asian/Hispanic) | 1.1512 | 0.0267 | 10000 |  |  |  | +0.0000 |  | 0.165529 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.1688 | 0.0107 | 10000 | *** | 0.000000 | paired t | +0.0009 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.1686 | 0.0105 | 10000 | *** | 0.000177 | paired t | +0.0007 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 0.1685 | 0.0103 | 10000 | *** | 0.000000 | paired t | +0.0006 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 0.1679 | 0.0100 | 10000 | *** | 0.000004 | paired t | +0.0000 | *** | 0.000001 |
| 5 | Racial (White/Black) | 0.1679 | 0.0100 | 10000 |  | 0.216347 | paired t | +0.0000 |  | 0.098183 |
| 6 | Ethnic (Asian/Hispanic) | 0.1679 | 0.0100 | 10000 |  |  |  | +0.0000 |  | 0.175497 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.5090 | 0.0177 | 10000 | *** | 0.000000 | paired t | +0.0107 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.5053 | 0.0172 | 10000 | *** | 0.000000 | paired t | +0.0069 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 0.5019 | 0.0171 | 10000 | *** | 0.000000 | paired t | +0.0035 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 0.4984 | 0.0163 | 10000 | *** | 0.000000 | paired t | +0.0000 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.4984 | 0.0163 | 10000 |  | 0.193996 | paired t | +0.0000 | * | 0.031596 |
| 6 | Ethnic (Asian/Hispanic) | 0.4984 | 0.0163 | 10000 |  |  |  | +0.0000 |  | 0.158422 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 13.0641 | 5.1755 | 10000 | *** | 0.000000 | paired t | +3.9219 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 11.5780 | 4.7463 | 10000 | *** | 0.000000 | paired t | +2.4358 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 10.2971 | 4.4711 | 10000 | *** | 0.000000 | paired t | +1.1549 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 9.1566 | 4.0033 | 10000 | *** | 0.000000 | paired t | +0.0144 | *** | 0.000000 |
| 5 | Racial (White/Black) | 9.1431 | 3.9978 | 10000 |  | 0.317335 | paired t | +0.0009 |  | 0.083265 |
| 6 | Ethnic (Asian/Hispanic) | 9.1425 | 3.9974 | 10000 |  |  |  | +0.0003 |  | 0.317335 |
