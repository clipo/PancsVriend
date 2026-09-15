# Segregation Ranking by Scenario (llama-3.3-70b-vf-lp)

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
| 1 | Color (Red/Blue) | 0.3423 | 0.0855 | 10000 | *** | 0.000000 | paired t | +0.2173 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.3263 | 0.0825 | 10000 | *** | 0.000000 | paired t | +0.2013 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.2998 | 0.0794 | 10000 | *** | 0.000000 | paired t | +0.1748 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.1757 | 0.0486 | 10000 | *** | 0.000000 | paired t | +0.0508 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.1390 | 0.0374 | 10000 | *** | 0.000000 | paired t | +0.0141 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1333 | 0.0358 | 10000 |  |  |  | +0.0084 | *** | 0.000000 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 89.2540 | 9.1030 | 10000 | *** | 0.000000 | paired t | +6.8213 | *** | 0.000000 |
| 2 | Racial (White/Black) | 84.6000 | 9.2703 | 10000 | *** | 0.000000 | paired t | +11.4753 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 66.4513 | 10.4256 | 10000 | *** | 0.000000 | paired t | +29.6240 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 29.5114 | 6.1855 | 10000 | *** | 0.000000 | paired t | +66.5639 | *** | 0.000000 |
| 5 | Political (Liberal/Conservative) | 19.9172 | 4.6016 | 10000 | *** | 0.000000 | paired t | +76.1581 | *** | 0.000000 |
| 6 | Color (Red/Blue) | 16.6850 | 3.9201 | 10000 |  |  |  | +79.3903 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 0.5019 | 0.0203 | 10000 | *** | 0.000000 | paired t | -0.0002 |  | 0.000073 |
| 2 | Racial (White/Black) | 0.5013 | 0.0208 | 10000 | *** | 0.000000 | paired t | +0.0004 | *** | 0.000006 |
| 3 | Color (Green/Yellow) | 0.4848 | 0.0254 | 10000 | *** | 0.000000 | paired t | +0.0169 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 0.3638 | 0.0398 | 10000 | *** | 0.000000 | paired t | +0.1378 | *** | 0.000000 |
| 5 | Political (Liberal/Conservative) | 0.3235 | 0.0389 | 10000 | *** | 0.000000 | paired t | +0.1781 | *** | 0.000000 |
| 6 | Color (Red/Blue) | 0.3043 | 0.0384 | 10000 |  |  |  | +0.1974 | *** | 0.000000 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 2.0480 | 0.2211 | 10000 | *** | 0.000000 | paired t | +0.8969 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 1.9624 | 0.2042 | 10000 | *** | 0.000000 | paired t | +0.8113 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 1.8055 | 0.1844 | 10000 | *** | 0.000000 | paired t | +0.6543 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 1.3229 | 0.0719 | 10000 | *** | 0.000000 | paired t | +0.1717 | *** | 0.000000 |
| 5 | Racial (White/Black) | 1.2036 | 0.0385 | 10000 | *** | 0.000000 | paired t | +0.0524 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 1.1822 | 0.0342 | 10000 |  |  |  | +0.0310 | *** | 0.000000 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.3175 | 0.0223 | 10000 | *** | 0.000000 | paired t | +0.1496 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.3057 | 0.0230 | 10000 | *** | 0.000000 | paired t | +0.1378 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.2828 | 0.0245 | 10000 | *** | 0.000000 | paired t | +0.1149 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.1895 | 0.0170 | 10000 | *** | 0.000000 | paired t | +0.0216 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.1691 | 0.0114 | 10000 | *** | 0.000000 | paired t | +0.0012 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1686 | 0.0108 | 10000 |  |  |  | +0.0007 | *** | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 0.7961 | 0.0255 | 10000 | *** | 0.000000 | paired t | +0.2977 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.7808 | 0.0266 | 10000 | *** | 0.000000 | paired t | +0.2824 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 0.7417 | 0.0305 | 10000 | *** | 0.000000 | paired t | +0.2433 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.5751 | 0.0258 | 10000 | *** | 0.000000 | paired t | +0.0767 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.5201 | 0.0184 | 10000 | *** | 0.000000 | paired t | +0.0217 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.5117 | 0.0178 | 10000 |  |  |  | +0.0133 | *** | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Color (Red/Blue) | 138.6879 | 20.0624 | 10000 | *** | 0.000000 | paired t | +129.5457 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 128.4089 | 20.2219 | 10000 | *** | 0.000000 | paired t | +119.2667 | *** | 0.000000 |
| 3 | Economic (High/Low Income) | 106.3303 | 20.3870 | 10000 | *** | 0.000000 | paired t | +97.1881 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 36.3197 | 11.1172 | 10000 | *** | 0.000000 | paired t | +27.1775 | *** | 0.000000 |
| 5 | Racial (White/Black) | 17.5336 | 6.0750 | 10000 | *** | 0.000000 | paired t | +8.3914 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 14.0982 | 5.3515 | 10000 |  |  |  | +4.9560 | *** | 0.000000 |
