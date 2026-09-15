# Segregation Ranking by Scenario (deepseek-v4-flash-vf-lp)

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
| 1 | Political (Liberal/Conservative) | 0.4216 | 0.0974 | 10000 | *** | 0.000000 | paired t | +0.2967 | *** | 0.000000 |
| 2 | Color (Red/Blue) | 0.3743 | 0.0911 | 10000 | *** | 0.000000 | paired t | +0.2493 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.3081 | 0.0800 | 10000 | *** | 0.000000 | paired t | +0.1832 | *** | 0.000000 |
| 4 | Racial (White/Black) | 0.2556 | 0.0655 | 10000 | *** | 0.000000 | paired t | +0.1307 | *** | 0.000000 |
| 5 | Ethnic (Asian/Hispanic) | 0.1730 | 0.0464 | 10000 | *** | 0.000000 | paired t | +0.0480 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 0.1407 | 0.0384 | 10000 |  |  |  | +0.0157 | *** | 0.000000 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 85.4725 | 9.7515 | 10000 | *** | 0.000000 | paired t | +10.6028 | *** | 0.000000 |
| 2 | Ethnic (Asian/Hispanic) | 66.7816 | 9.1388 | 10000 | *** | 0.000000 | paired t | +29.2937 | *** | 0.000000 |
| 3 | Racial (White/Black) | 52.9251 | 8.7966 | 10000 | *** | 0.000000 | paired t | +43.1502 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 31.1361 | 6.0827 | 10000 | *** | 0.000000 | paired t | +64.9392 | *** | 0.000000 |
| 5 | Color (Red/Blue) | 14.0290 | 3.3666 | 10000 | *** | 0.000000 | paired t | +82.0463 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 10.5352 | 2.4914 | 10000 |  |  |  | +85.5401 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Green/Yellow) | 0.4995 | 0.0214 | 10000 | *** | 0.000000 | paired t | +0.0022 | *** | 0.000000 |
| 2 | Ethnic (Asian/Hispanic) | 0.4864 | 0.0229 | 10000 | *** | 0.000000 | paired t | +0.0153 | *** | 0.000000 |
| 3 | Racial (White/Black) | 0.4049 | 0.0322 | 10000 | *** | 0.000000 | paired t | +0.0968 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 0.3504 | 0.0387 | 10000 | *** | 0.000000 | paired t | +0.1513 | *** | 0.000000 |
| 5 | Color (Red/Blue) | 0.2498 | 0.0377 | 10000 | *** | 0.000000 | paired t | +0.2518 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 0.1820 | 0.0319 | 10000 |  |  |  | +0.3197 | *** | 0.000000 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 2.4934 | 0.2989 | 10000 | *** | 0.000000 | paired t | +1.3422 | *** | 0.000000 |
| 2 | Color (Red/Blue) | 2.2188 | 0.2526 | 10000 | *** | 0.000000 | paired t | +1.0677 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 1.8444 | 0.1949 | 10000 | *** | 0.000000 | paired t | +0.6933 | *** | 0.000000 |
| 4 | Racial (White/Black) | 1.5005 | 0.1087 | 10000 | *** | 0.000000 | paired t | +0.3493 | *** | 0.000000 |
| 5 | Ethnic (Asian/Hispanic) | 1.3125 | 0.0612 | 10000 | *** | 0.000000 | paired t | +0.1613 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 1.2034 | 0.0411 | 10000 |  |  |  | +0.0522 | *** | 0.000000 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.3793 | 0.0182 | 10000 | *** | 0.000000 | paired t | +0.2114 | *** | 0.000000 |
| 2 | Color (Red/Blue) | 0.3444 | 0.0215 | 10000 | *** | 0.000000 | paired t | +0.1765 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.2917 | 0.0236 | 10000 | *** | 0.000000 | paired t | +0.1238 | *** | 0.000000 |
| 4 | Racial (White/Black) | 0.2533 | 0.0208 | 10000 | *** | 0.000000 | paired t | +0.0854 | *** | 0.000000 |
| 5 | Ethnic (Asian/Hispanic) | 0.1871 | 0.0150 | 10000 | *** | 0.000000 | paired t | +0.0193 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 0.1721 | 0.0119 | 10000 |  |  |  | +0.0042 | *** | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.8769 | 0.0197 | 10000 | *** | 0.000000 | paired t | +0.3785 | *** | 0.000000 |
| 2 | Color (Red/Blue) | 0.8328 | 0.0243 | 10000 | *** | 0.000000 | paired t | +0.3344 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.7491 | 0.0301 | 10000 | *** | 0.000000 | paired t | +0.2507 | *** | 0.000000 |
| 4 | Racial (White/Black) | 0.6702 | 0.0309 | 10000 | *** | 0.000000 | paired t | +0.1718 | *** | 0.000000 |
| 5 | Ethnic (Asian/Hispanic) | 0.5660 | 0.0214 | 10000 | *** | 0.000000 | paired t | +0.0676 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 0.5233 | 0.0200 | 10000 |  |  |  | +0.0249 | *** | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 192.6839 | 18.3156 | 10000 | *** | 0.000000 | paired t | +183.5417 | *** | 0.000000 |
| 2 | Color (Red/Blue) | 160.8963 | 20.3324 | 10000 | *** | 0.000000 | paired t | +151.7541 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 109.7849 | 20.5234 | 10000 | *** | 0.000000 | paired t | +100.6427 | *** | 0.000000 |
| 4 | Racial (White/Black) | 64.1200 | 16.3525 | 10000 | *** | 0.000000 | paired t | +54.9778 | *** | 0.000000 |
| 5 | Ethnic (Asian/Hispanic) | 34.1461 | 9.1113 | 10000 | *** | 0.000000 | paired t | +25.0039 | *** | 0.000000 |
| 6 | Color (Green/Yellow) | 17.3220 | 6.6075 | 10000 |  |  |  | +8.1798 | *** | 0.000000 |
