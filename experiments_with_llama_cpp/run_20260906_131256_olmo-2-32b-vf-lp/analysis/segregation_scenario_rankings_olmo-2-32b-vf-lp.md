# Segregation Ranking by Scenario (olmo-2-32b-vf-lp)

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
| 1 | Color (Green/Yellow) | 0.7377 | 0.0722 | 10000 | * | 0.011506 | paired t | +0.6128 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.7350 | 0.0823 | 10000 | *** | 0.000000 | paired t | +0.6100 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.7184 | 0.0627 | 10000 | *** | 0.000000 | paired t | +0.5934 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 0.6006 | 0.0948 | 10000 | *** | 0.000000 | paired t | +0.4757 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.5710 | 0.0907 | 10000 | *** | 0.000000 | paired t | +0.4461 | *** | 0.000000 |
| 6 | Economic (High/Low Income) | 0.4583 | 0.0833 | 10000 |  |  |  | +0.3334 | *** | 0.000000 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Economic (High/Low Income) | 37.5544 | 6.2362 | 10000 | *** | 0.000000 | paired t | +58.5209 | *** | 0.000000 |
| 2 | Racial (White/Black) | 17.9342 | 4.3991 | 10000 | *** | 0.000000 | paired t | +78.1411 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 15.5707 | 4.2885 | 10000 | *** | 0.000000 | paired t | +80.5046 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 13.8563 | 3.8015 | 10000 | *** | 0.000000 | paired t | +82.2190 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 8.9443 | 2.9952 | 10000 | *** | 0.000000 | paired t | +87.1310 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 5.5108 | 1.9244 | 10000 |  |  |  | +90.5645 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Economic (High/Low Income) | 0.2883 | 0.0345 | 10000 | *** | 0.000000 | paired t | +0.2134 | *** | 0.000000 |
| 2 | Racial (White/Black) | 0.2227 | 0.0368 | 10000 | *** | 0.000000 | paired t | +0.2790 | *** | 0.000000 |
| 3 | Ethnic (Asian/Hispanic) | 0.1910 | 0.0353 | 10000 | *** | 0.000000 | paired t | +0.3107 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.0859 | 0.0244 | 10000 | *** | 0.000000 | paired t | +0.4158 | *** | 0.000000 |
| 5 | Color (Green/Yellow) | 0.0503 | 0.0185 | 10000 | *** | 0.000000 | paired t | +0.4514 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 0.0427 | 0.0168 | 10000 |  |  |  | +0.4590 | *** | 0.000000 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 4.7258 | 0.8155 | 10000 | *** | 0.000000 | paired t | +3.5747 | *** | 0.000000 |
| 2 | Color (Green/Yellow) | 4.5803 | 0.7534 | 10000 | *** | 0.000000 | paired t | +3.4291 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 3.6482 | 0.5935 | 10000 | *** | 0.000000 | paired t | +2.4970 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 3.0158 | 0.5123 | 10000 | *** | 0.000000 | paired t | +1.8646 | *** | 0.000000 |
| 5 | Racial (White/Black) | 2.5637 | 0.3797 | 10000 | *** | 0.000000 | paired t | +1.4125 | *** | 0.000000 |
| 6 | Economic (High/Low Income) | 1.8206 | 0.1621 | 10000 |  |  |  | +0.6694 | *** | 0.000000 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.4663 | 0.0104 | 10000 | *** | 0.000000 | paired t | +0.2984 | *** | 0.000000 |
| 2 | Color (Green/Yellow) | 0.4641 | 0.0108 | 10000 | *** | 0.000000 | paired t | +0.2962 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.4441 | 0.0138 | 10000 | *** | 0.000000 | paired t | +0.2762 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 0.3916 | 0.0201 | 10000 | *** | 0.000000 | paired t | +0.2237 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.3740 | 0.0213 | 10000 | *** | 0.000000 | paired t | +0.2061 | *** | 0.000000 |
| 6 | Economic (High/Low Income) | 0.3263 | 0.0209 | 10000 |  |  |  | +0.1584 | *** | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.9654 | 0.0121 | 10000 | *** | 0.000000 | paired t | +0.4670 | *** | 0.000000 |
| 2 | Color (Green/Yellow) | 0.9556 | 0.0155 | 10000 | *** | 0.000000 | paired t | +0.4572 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.9223 | 0.0214 | 10000 | *** | 0.000000 | paired t | +0.4239 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 0.8619 | 0.0249 | 10000 | *** | 0.000000 | paired t | +0.3635 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.8279 | 0.0276 | 10000 | *** | 0.000000 | paired t | +0.3296 | *** | 0.000000 |
| 6 | Economic (High/Low Income) | 0.7478 | 0.0287 | 10000 |  |  |  | +0.2494 | *** | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 278.9128 | 12.8145 | 10000 | *** | 0.000000 | paired t | +269.7706 | *** | 0.000000 |
| 2 | Color (Green/Yellow) | 269.8110 | 15.7941 | 10000 | *** | 0.000000 | paired t | +260.6688 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 238.4396 | 19.9602 | 10000 | *** | 0.000000 | paired t | +229.2974 | *** | 0.000000 |
| 4 | Ethnic (Asian/Hispanic) | 197.5521 | 21.1420 | 10000 | *** | 0.000000 | paired t | +188.4099 | *** | 0.000000 |
| 5 | Racial (White/Black) | 170.9872 | 21.8381 | 10000 | *** | 0.000000 | paired t | +161.8450 | *** | 0.000000 |
| 6 | Economic (High/Low Income) | 107.5280 | 18.9609 | 10000 |  |  |  | +98.3858 | *** | 0.000000 |
