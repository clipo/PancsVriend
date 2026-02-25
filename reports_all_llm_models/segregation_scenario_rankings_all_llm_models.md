# Segregation Ranking by Scenario (All LLM Models)

Higher composite scores indicate more segregation.

## gemma3:27b

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Political (Liberal/Conservative) | 0.6527 | 0.0099 | 10 | * |
| 2 | Economic (High/Low Income) | 0.3659 | 0.0989 | 10 | * |
| 3 | Ethnic (Asian/Hispanic) | 0.2837 | 0.0205 | 75 | * |
| 4 | Racial (White/Black) | 0.2770 | 0.0156 | 100 |  |
| 5 | Color (Red/Blue) | 0.2738 | 0.0139 | 100 |  |
| 6 | Color (Green/Yellow) | 0.2650 | 0.0127 | 10 |  |

## gemma3:4b

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Color (Red/Blue) | 0.5305 | 0.1310 | 10 |  |
| 2 | Political (Liberal/Conservative) | 0.5282 | 0.1007 | 10 |  |
| 3 | Ethnic (Asian/Hispanic) | 0.4940 | 0.1240 | 10 |  |
| 4 | Racial (White/Black) | 0.4638 | 0.1053 | 10 |  |
| 5 | Color (Green/Yellow) | 0.4477 | 0.1072 | 10 |  |
| 6 | Economic (High/Low Income) | 0.4433 | 0.1407 | 10 |  |

## hermes3:latest

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Racial (White/Black) | 0.4990 | 0.1199 | 10 |  |
| 2 | Economic (High/Low Income) | 0.4745 | 0.0715 | 10 |  |
| 3 | Ethnic (Asian/Hispanic) | 0.4461 | 0.1072 | 10 |  |
| 4 | Color (Red/Blue) | 0.4377 | 0.1215 | 10 |  |
| 5 | Political (Liberal/Conservative) | 0.4360 | 0.1292 | 10 |  |
| 6 | Color (Green/Yellow) | 0.4125 | 0.1161 | 10 |  |

## llama3.3:latest

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Political (Liberal/Conservative) | 0.4250 | 0.0788 | 90 | * |
| 2 | Racial (White/Black) | 0.3992 | 0.0748 | 90 |  |
| 3 | Color (Green/Yellow) | 0.3924 | 0.0500 | 90 |  |
| 4 | Ethnic (Asian/Hispanic) | 0.3917 | 0.0591 | 90 |  |
| 5 | Economic (High/Low Income) | 0.3902 | 0.0610 | 90 |  |
| 6 | Color (Red/Blue) | 0.3880 | 0.0780 | 19 |  |

## mixtral:8x22b-instruct

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Political (Liberal/Conservative) | 0.5988 | 0.0374 | 100 | * |
| 2 | Racial (White/Black) | 0.4903 | 0.0501 | 100 | * |
| 3 | Ethnic (Asian/Hispanic) | 0.4190 | 0.0838 | 100 | * |
| 4 | Color (Red/Blue) | 0.3645 | 0.0460 | 100 | * |
| 5 | Economic (High/Low Income) | 0.3434 | 0.0395 | 100 | * |
| 6 | Color (Green/Yellow) | 0.3108 | 0.0250 | 100 |  |

## phi4:latest

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Economic (High/Low Income) | 0.5075 | 0.0760 | 10 |  |
| 2 | Ethnic (Asian/Hispanic) | 0.4532 | 0.1074 | 10 |  |
| 3 | Color (Green/Yellow) | 0.4346 | 0.0808 | 10 |  |
| 4 | Color (Red/Blue) | 0.4267 | 0.1123 | 10 |  |
| 5 | Political (Liberal/Conservative) | 0.4224 | 0.1027 | 10 |  |
| 6 | Racial (White/Black) | 0.4213 | 0.1010 | 10 |  |

## qwen2.5-coder:32B

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Economic (High/Low Income) | 0.4766 | 0.0870 | 100 | * |
| 2 | Ethnic (Asian/Hispanic) | 0.3558 | 0.0457 | 100 |  |
| 3 | Racial (White/Black) | 0.3483 | 0.0400 | 100 |  |
| 4 | Color (Green/Yellow) | 0.3431 | 0.0425 | 100 |  |
| 5 | Political (Liberal/Conservative) | 0.3416 | 0.0340 | 100 |  |
| 6 | Color (Red/Blue) | 0.3380 | 0.0369 | 100 |  |
