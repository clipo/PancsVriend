# Segregation Ranking by Scenario (mistral-small-4-119b-vf-lp)

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
| 1 | Political (Liberal/Conservative) | 0.1294 | 0.0348 | 10000 | *** | 0.000000 | paired t | +0.0044 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.1261 | 0.0337 | 10000 | *** | 0.000000 | paired t | +0.0011 | *** | 0.000000 |
| 3 | Ethnic (Asian/Hispanic) | 0.1252 | 0.0334 | 10000 | * | 0.015568 | paired t | +0.0003 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.1252 | 0.0334 | 10000 | *** | 0.000000 | paired t | +0.0002 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 0.1249 | 0.0333 | 10000 | * | 0.017213 | paired t | +0.0000 |  | 0.317335 |
| 6 | Racial (White/Black) | 0.1249 | 0.0333 | 10000 |  |  |  | -0.0000 |  | 0.020092 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 96.0745 | 8.4394 | 10000 | *** | 0.000000 | paired t | +0.0008 | * | 0.020914 |
| 2 | Racial (White/Black) | 95.9904 | 8.4498 | 10000 | *** | 0.000000 | paired t | +0.0849 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 95.9190 | 8.4650 | 10000 | *** | 0.000000 | paired t | +0.1563 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 95.8674 | 8.4442 | 10000 | *** | 0.000000 | paired t | +0.2079 | *** | 0.000000 |
| 5 | Economic (High/Low Income) | 95.3478 | 8.5374 | 10000 | *** | 0.000000 | paired t | +0.7275 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 92.9288 | 8.8448 | 10000 |  |  |  | +3.1465 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Racial (White/Black) | 0.5019 | 0.0199 | 10000 | *** | 0.000000 | paired t | -0.0002 |  | 0.000000 |
| 2 | Color (Red/Blue) | 0.5017 | 0.0200 | 10000 |  | 0.614262 | paired t | -0.0000 |  | 0.001141 |
| 3 | Political (Liberal/Conservative) | 0.5017 | 0.0202 | 10000 |  | 0.849049 | paired t | -0.0000 |  | 0.851882 |
| 4 | Color (Green/Yellow) | 0.5017 | 0.0200 | 10000 |  | 0.644502 | paired t | +0.0000 |  | 0.817564 |
| 5 | Economic (High/Low Income) | 0.5017 | 0.0201 | 10000 |  | 0.692217 | paired t | +0.0000 |  | 0.639535 |
| 6 | Ethnic (Asian/Hispanic) | 0.5017 | 0.0200 | 10000 |  |  |  | +0.0000 |  | 0.091221 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 1.1656 | 0.0307 | 10000 | *** | 0.000000 | paired t | +0.0144 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 1.1544 | 0.0277 | 10000 | *** | 0.000000 | paired t | +0.0032 | *** | 0.000000 |
| 3 | Ethnic (Asian/Hispanic) | 1.1520 | 0.0269 | 10000 |  | 0.621900 | paired t | +0.0008 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 1.1520 | 0.0270 | 10000 | *** | 0.000000 | paired t | +0.0008 | *** | 0.000000 |
| 5 | Racial (White/Black) | 1.1514 | 0.0268 | 10000 | *** | 0.000000 | paired t | +0.0002 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 1.1512 | 0.0267 | 10000 |  |  |  | +0.0000 | * | 0.013369 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.1686 | 0.0105 | 10000 | *** | 0.000000 | paired t | +0.0007 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.1681 | 0.0102 | 10000 | *** | 0.000000 | paired t | +0.0002 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.1680 | 0.0101 | 10000 |  | 0.180748 | paired t | +0.0001 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 0.1679 | 0.0100 | 10000 | *** | 0.000000 | paired t | +0.0001 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 0.1679 | 0.0100 | 10000 | *** | 0.000000 | paired t | +0.0000 |  | 0.092440 |
| 6 | Racial (White/Black) | 0.1677 | 0.0100 | 10000 |  |  |  | -0.0002 |  | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.5054 | 0.0173 | 10000 | *** | 0.000000 | paired t | +0.0070 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.5001 | 0.0166 | 10000 | *** | 0.000000 | paired t | +0.0017 | *** | 0.000000 |
| 3 | Ethnic (Asian/Hispanic) | 0.4988 | 0.0163 | 10000 |  | 0.053339 | paired t | +0.0004 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.4988 | 0.0164 | 10000 | *** | 0.000000 | paired t | +0.0004 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 0.4984 | 0.0163 | 10000 | *** | 0.000000 | paired t | +0.0000 | * | 0.017560 |
| 6 | Racial (White/Black) | 0.4983 | 0.0163 | 10000 |  |  |  | -0.0001 |  | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 11.4059 | 4.7549 | 10000 | *** | 0.000000 | paired t | +2.2637 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 9.6187 | 4.1651 | 10000 | *** | 0.000000 | paired t | +0.4765 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 9.2696 | 4.0397 | 10000 |  | 0.673843 | paired t | +0.1274 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 9.2667 | 4.0320 | 10000 | *** | 0.000000 | paired t | +0.1245 | *** | 0.000000 |
| 5 | Racial (White/Black) | 9.2009 | 4.0207 | 10000 | *** | 0.000000 | paired t | +0.0587 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 9.1433 | 3.9976 | 10000 |  |  |  | +0.0011 | * | 0.021803 |
