# Segregation Ranking by Scenario (hermes-4.3-36b-vf-lp)

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
| 1 | Economic (High/Low Income) | 0.4842 | 0.1045 | 10000 | *** | 0.000009 | paired t | +0.3593 | *** | 0.000000 |
| 2 | Political (Liberal/Conservative) | 0.4788 | 0.1029 | 10000 | *** | 0.000000 | paired t | +0.3538 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.4515 | 0.0997 | 10000 | *** | 0.000000 | paired t | +0.3265 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.3886 | 0.0929 | 10000 | *** | 0.000000 | paired t | +0.2637 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.2598 | 0.0704 | 10000 | *** | 0.000000 | paired t | +0.1348 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1618 | 0.0437 | 10000 |  |  |  | +0.0368 | *** | 0.000000 |

## clusters

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 72.8409 | 9.6428 | 10000 | *** | 0.000000 | paired t | +23.2344 | *** | 0.000000 |
| 2 | Racial (White/Black) | 35.8430 | 8.1883 | 10000 | *** | 0.000000 | paired t | +60.2323 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 11.9830 | 2.9850 | 10000 | *** | 0.000000 | paired t | +84.0923 | *** | 0.000000 |
| 4 | Economic (High/Low Income) | 10.0023 | 2.5685 | 10000 | *** | 0.000000 | paired t | +86.0730 | *** | 0.000000 |
| 5 | Color (Red/Blue) | 9.2019 | 2.3006 | 10000 | *** | 0.000000 | paired t | +86.8734 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 7.8670 | 1.8717 | 10000 |  |  |  | +88.2083 | *** | 0.000000 |

## switch_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Ethnic (Asian/Hispanic) | 0.4922 | 0.0226 | 10000 | *** | 0.000000 | paired t | +0.0095 | *** | 0.000000 |
| 2 | Racial (White/Black) | 0.4101 | 0.0374 | 10000 | *** | 0.000000 | paired t | +0.0915 | *** | 0.000000 |
| 3 | Color (Green/Yellow) | 0.2365 | 0.0370 | 10000 | *** | 0.000000 | paired t | +0.2652 | *** | 0.000000 |
| 4 | Color (Red/Blue) | 0.1574 | 0.0305 | 10000 | *** | 0.000000 | paired t | +0.3443 | *** | 0.000000 |
| 5 | Economic (High/Low Income) | 0.1493 | 0.0314 | 10000 | *** | 0.000000 | paired t | +0.3524 | *** | 0.000000 |
| 6 | Political (Liberal/Conservative) | 0.1128 | 0.0261 | 10000 |  |  |  | +0.3889 | *** | 0.000000 |

## distance

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 2.8599 | 0.3685 | 10000 | *** | 0.000000 | paired t | +1.7087 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 2.8325 | 0.4257 | 10000 | *** | 0.000000 | paired t | +1.6813 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 2.6707 | 0.3478 | 10000 | *** | 0.000000 | paired t | +1.5195 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 2.3064 | 0.2723 | 10000 | *** | 0.000000 | paired t | +1.1552 | *** | 0.000000 |
| 5 | Racial (White/Black) | 1.6419 | 0.1460 | 10000 | *** | 0.000000 | paired t | +0.4907 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 1.2747 | 0.0552 | 10000 |  |  |  | +0.1235 | *** | 0.000000 |

## mix_deviation

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.4158 | 0.0152 | 10000 | *** | 0.000000 | paired t | +0.2479 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.4030 | 0.0179 | 10000 | *** | 0.000000 | paired t | +0.2351 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.3951 | 0.0173 | 10000 | *** | 0.000000 | paired t | +0.2272 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.3530 | 0.0209 | 10000 | *** | 0.000000 | paired t | +0.1851 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.2508 | 0.0244 | 10000 | *** | 0.000000 | paired t | +0.0829 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.1806 | 0.0142 | 10000 |  |  |  | +0.0127 | *** | 0.000000 |

## share

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 0.9177 | 0.0163 | 10000 | *** | 0.000000 | paired t | +0.4193 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 0.8980 | 0.0198 | 10000 | *** | 0.000000 | paired t | +0.3996 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 0.8920 | 0.0189 | 10000 | *** | 0.000000 | paired t | +0.3936 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 0.8431 | 0.0233 | 10000 | *** | 0.000000 | paired t | +0.3447 | *** | 0.000000 |
| 5 | Racial (White/Black) | 0.6921 | 0.0330 | 10000 | *** | 0.000000 | paired t | +0.1937 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 0.5530 | 0.0214 | 10000 |  |  |  | +0.0546 | *** | 0.000000 |

## ghetto_rate

| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |
|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|
| 1 | Political (Liberal/Conservative) | 225.2665 | 16.2084 | 10000 | *** | 0.000000 | paired t | +216.1243 | *** | 0.000000 |
| 2 | Economic (High/Low Income) | 213.3735 | 18.9365 | 10000 | *** | 0.000000 | paired t | +204.2313 | *** | 0.000000 |
| 3 | Color (Red/Blue) | 205.9593 | 17.9262 | 10000 | *** | 0.000000 | paired t | +196.8171 | *** | 0.000000 |
| 4 | Color (Green/Yellow) | 170.0398 | 19.9825 | 10000 | *** | 0.000000 | paired t | +160.8976 | *** | 0.000000 |
| 5 | Racial (White/Black) | 84.0861 | 18.7513 | 10000 | *** | 0.000000 | paired t | +74.9439 | *** | 0.000000 |
| 6 | Ethnic (Asian/Hispanic) | 28.7867 | 8.6497 | 10000 |  |  |  | +19.6445 | *** | 0.000000 |
