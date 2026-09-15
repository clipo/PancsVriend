# Segregation Ranking by Scenario (gemma-4-31b-vf-lp)

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
| 1 | Color (Green/Yellow) | 0.2557 | 0.0691 | 10000 | *** | 0.000000 | paired t | +0.1308 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.2181 | 0.0574 | 10000 | *** | 0.000000 | paired t | +0.0932 | *** | 0.000000 |
| 3 | Political (Liberal/Conservative) | 0.2084 | 0.0575 | 10000 | *** | 0.000000 | paired t | +0.0835 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.1997 | 0.0539 | 10000 | *** | 0.000000 | paired t | +0.0747 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.1346 | 0.0363 | 10000 | *** | 0.000000 | paired t | +0.0096 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1299 | 0.0348 | 10000 |  |  |  | +0.0049 | *** | 0.000000 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 92.3644 | 8.8246 | 10000 | *** | 0.000000 | paired t | +3.7109 | *** | 0.000000 |
| 2 | Racial (White/Black) | 88.7091 | 9.1706 | 10000 | *** | 0.000000 | paired t | +7.3662 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 57.5284 | 8.9489 | 10000 | *** | 0.000000 | paired t | +38.5469 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 53.7315 | 7.3873 | 10000 | ** | 0.001763 | paired t | +42.3438 | *** | 0.000000 |
| 5 | Political (Liberal/Conservative) | 53.4406 | 10.1198 | 10000 | *** | 0.000000 | paired t | +42.6347 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 36.0343 | 7.8471 | 10000 |  |  |  | +60.0410 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Racial (White/Black) | 0.5020 | 0.0205 | 10000 |  | 0.277788 | paired t | -0.0003 |  | 0.000000 |
| 2 | Ethnic (Asian/Hispanic) | 0.5020 | 0.0202 | 10000 | *** | 0.000000 | paired t | -0.0003 |  | 0.000000 |
| 3 | Color (Red/Blue) | 0.4681 | 0.0270 | 10000 | *** | 0.000000 | paired t | +0.0336 | *** | 0.000000 |
| 4 | Political (Liberal/Conservative) | 0.4609 | 0.0306 | 10000 | *** | 0.000000 | paired t | +0.0408 | *** | 0.000000 |
| 5 | Economic (High/Low Income) | 0.4513 | 0.0266 | 10000 | *** | 0.000000 | paired t | +0.0504 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 0.4152 | 0.0362 | 10000 |  |  |  | +0.0865 | *** | 0.000000 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 1.6297 | 0.1388 | 10000 | *** | 0.000000 | paired t | +0.4785 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 1.4595 | 0.0857 | 10000 | *** | 0.000000 | paired t | +0.3083 | *** | 0.000000 |
| 3 | Political (Liberal/Conservative) | 1.4345 | 0.0975 | 10000 | *** | 0.000000 | paired t | +0.2833 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 1.4025 | 0.0836 | 10000 | *** | 0.000000 | paired t | +0.2513 | *** | 0.000000 |
| 5 | Racial (White/Black) | 1.1869 | 0.0354 | 10000 | *** | 0.000000 | paired t | +0.0358 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 1.1692 | 0.0310 | 10000 |  |  |  | +0.0180 | *** | 0.000000 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 0.2485 | 0.0234 | 10000 | *** | 0.000000 | paired t | +0.0806 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.2227 | 0.0173 | 10000 | *** | 0.000000 | paired t | +0.0548 | *** | 0.000000 |
| 3 | Political (Liberal/Conservative) | 0.2122 | 0.0206 | 10000 | *** | 0.000000 | paired t | +0.0443 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.2067 | 0.0182 | 10000 | *** | 0.000000 | paired t | +0.0388 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.1693 | 0.0109 | 10000 | *** | 0.000000 | paired t | +0.0014 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1686 | 0.0104 | 10000 |  |  |  | +0.0007 | *** | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 0.6907 | 0.0303 | 10000 | *** | 0.000000 | paired t | +0.1923 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.6318 | 0.0228 | 10000 | *** | 0.000000 | paired t | +0.1334 | *** | 0.000000 |
| 3 | Political (Liberal/Conservative) | 0.6252 | 0.0293 | 10000 | *** | 0.000000 | paired t | +0.1268 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.6077 | 0.0253 | 10000 | *** | 0.000000 | paired t | +0.1093 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.5139 | 0.0181 | 10000 | *** | 0.000000 | paired t | +0.0155 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.5063 | 0.0171 | 10000 |  |  |  | +0.0079 | *** | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 82.6426 | 17.9303 | 10000 | *** | 0.000000 | paired t | +73.5004 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 57.8254 | 11.5074 | 10000 | *** | 0.000000 | paired t | +48.6832 | *** | 0.000000 |
| 3 | Political (Liberal/Conservative) | 54.0284 | 14.6208 | 10000 | *** | 0.000000 | paired t | +44.8862 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 48.5092 | 12.0628 | 10000 | *** | 0.000000 | paired t | +39.3670 | *** | 0.000000 |
| 5 | Racial (White/Black) | 14.8814 | 5.6090 | 10000 | *** | 0.000000 | paired t | +5.7392 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 12.0291 | 4.7847 | 10000 |  |  |  | +2.8869 | *** | 0.000000 |
