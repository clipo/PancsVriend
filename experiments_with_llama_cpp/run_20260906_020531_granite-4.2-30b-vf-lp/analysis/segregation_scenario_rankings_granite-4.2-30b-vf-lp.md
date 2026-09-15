# Segregation Ranking by Scenario (granite-4.2-30b-vf-lp)

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
| 1 | Color (Red/Blue) | 0.1327 | 0.0359 | 10000 | ** | 0.001265 | paired t | +0.0078 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.1324 | 0.0357 | 10000 | *** | 0.000000 | paired t | +0.0074 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.1267 | 0.0340 | 10000 | *** | 0.000000 | paired t | +0.0017 | *** | 0.000000 |
| 4 | Racial (White/Black) | 0.1250 | 0.0334 | 10000 | ** | 0.005744 | paired t | +0.0000 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 0.1250 | 0.0333 | 10000 | *** | 0.000014 | paired t | +0.0000 | *** | 0.000005 |
| 6 | Ethnic (Asian/Hispanic) | 0.1249 | 0.0333 | 10000 |  |  |  | +0.0000 |  | 0.654743 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 96.0735 | 8.4389 | 10000 | *** | 0.000000 | paired t | +0.0018 | ** | 0.001013 |
| 2 | Color (Green/Yellow) | 96.0603 | 8.4423 | 10000 | *** | 0.000000 | paired t | +0.0150 | *** | 0.000000 |
| 3 | Racial (White/Black) | 96.0483 | 8.4385 | 10000 | *** | 0.000000 | paired t | +0.0270 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 94.8891 | 8.6537 | 10000 | *** | 0.000000 | paired t | +1.1862 | *** | 0.000000 |
| 5 | Color (Red/Blue) | 90.1961 | 9.1583 | 10000 |  | 0.933067 | paired t | +5.8792 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 90.1929 | 9.0897 | 10000 |  |  |  | +5.8824 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.5020 | 0.0204 | 10000 |  | 0.664978 | paired t | -0.0003 |  | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.5020 | 0.0204 | 10000 | * | 0.016517 | paired t | -0.0003 |  | 0.000000 |
| 3 | Economic (High/Low Income) | 0.5019 | 0.0201 | 10000 | *** | 0.000000 | paired t | -0.0002 |  | 0.000000 |
| 4 | Color (Green/Yellow) | 0.5017 | 0.0200 | 10000 |  | 0.453934 | paired t | -0.0000 |  | 0.071080 |
| 5 | Racial (White/Black) | 0.5017 | 0.0200 | 10000 |  | 0.475612 | paired t | -0.0000 |  | 0.516623 |
| 6 | Ethnic (Asian/Hispanic) | 0.5017 | 0.0200 | 10000 |  |  |  | +0.0000 |  | 0.861486 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 1.1794 | 0.0342 | 10000 | * | 0.018044 | paired t | +0.0283 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 1.1790 | 0.0342 | 10000 | *** | 0.000000 | paired t | +0.0278 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 1.1573 | 0.0288 | 10000 | *** | 0.000000 | paired t | +0.0061 | *** | 0.000000 |
| 4 | Racial (White/Black) | 1.1513 | 0.0268 | 10000 | *** | 0.000007 | paired t | +0.0001 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 1.1513 | 0.0268 | 10000 | *** | 0.000000 | paired t | +0.0001 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 1.1512 | 0.0268 | 10000 |  |  |  | +0.0000 | ** | 0.002541 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.1690 | 0.0108 | 10000 | *** | 0.000000 | paired t | +0.0011 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.1688 | 0.0108 | 10000 | *** | 0.000000 | paired t | +0.0009 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.1683 | 0.0102 | 10000 | *** | 0.000000 | paired t | +0.0004 | *** | 0.000000 |
| 4 | Racial (White/Black) | 0.1679 | 0.0100 | 10000 |  | 0.051766 | paired t | +0.0000 | *** | 0.000189 |
| 5 | Color (Green/Yellow) | 0.1679 | 0.0100 | 10000 | * | 0.044437 | paired t | +0.0000 | * | 0.015732 |
| 6 | Ethnic (Asian/Hispanic) | 0.1679 | 0.0100 | 10000 |  |  |  | +0.0000 |  | 0.304995 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.5107 | 0.0178 | 10000 | * | 0.030192 | paired t | +0.0124 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.5106 | 0.0179 | 10000 | *** | 0.000000 | paired t | +0.0122 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.5013 | 0.0168 | 10000 | *** | 0.000000 | paired t | +0.0029 | *** | 0.000000 |
| 4 | Racial (White/Black) | 0.4985 | 0.0163 | 10000 | *** | 0.000002 | paired t | +0.0001 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 0.4984 | 0.0163 | 10000 | *** | 0.000000 | paired t | +0.0000 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.4984 | 0.0163 | 10000 |  |  |  | +0.0000 | *** | 0.000812 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 13.6837 | 5.3568 | 10000 | ** | 0.002944 | paired t | +4.5415 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 13.5799 | 5.3903 | 10000 | *** | 0.000000 | paired t | +4.4377 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 10.1114 | 4.3753 | 10000 | *** | 0.000000 | paired t | +0.9692 | *** | 0.000000 |
| 4 | Racial (White/Black) | 9.1640 | 4.0043 | 10000 | *** | 0.000021 | paired t | +0.0218 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 9.1541 | 4.0077 | 10000 | *** | 0.000001 | paired t | +0.0119 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 9.1441 | 3.9982 | 10000 |  |  |  | +0.0019 | ** | 0.004616 |
