# Segregation Ranking by Scenario (All LLM Models)

Higher composite scores indicate more segregation.

## gemma3:27b

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Political (Liberal/Conservative) | 0.6527 | 0.0099 | 10 | * |
| 2 | Economic (High/Low Income) | 0.3659 | 0.0989 | 10 | * |
| 3 | Racial (White/Black) | 0.2770 | 0.0156 | 100 |  |
| 4 | Ethnic (Asian/Hispanic) | 0.2770 | 0.0185 | 30 |  |
| 5 | Color (Red/Blue) | 0.2738 | 0.0139 | 100 |  |

## gemma3:4b

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Color (Red/Blue) | 0.5401 | 0.1125 | 10 |  |
| 2 | Political (Liberal/Conservative) | 0.4995 | 0.1059 | 10 |  |
| 3 | Ethnic (Asian/Hispanic) | 0.4879 | 0.1048 | 10 |  |
| 4 | Racial (White/Black) | 0.4580 | 0.0909 | 10 |  |
| 5 | Economic (High/Low Income) | 0.4486 | 0.1254 | 10 |  |

## hermes3:latest

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Racial (White/Black) | 0.4997 | 0.1201 | 10 |  |
| 2 | Economic (High/Low Income) | 0.4767 | 0.0713 | 10 |  |
| 3 | Ethnic (Asian/Hispanic) | 0.4508 | 0.1118 | 10 |  |
| 4 | Color (Red/Blue) | 0.4362 | 0.1222 | 10 |  |
| 5 | Political (Liberal/Conservative) | 0.4342 | 0.1245 | 10 |  |

## llama3.3:latest

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Color (Red/Blue) | 0.4160 | 0.0853 | 86 |  |
| 2 | Racial (White/Black) | 0.3956 | 0.1241 | 10 |  |
| 3 | Ethnic (Asian/Hispanic) | 0.3931 | 0.0646 | 10 |  |
| 4 | Economic (High/Low Income) | 0.3906 | 0.0905 | 10 |  |
| 5 | Political (Liberal/Conservative) | 0.3884 | 0.0777 | 10 |  |

## mixtral:8x22b-instruct

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Political (Liberal/Conservative) | 0.5995 | 0.0375 | 100 | * |
| 2 | Racial (White/Black) | 0.4916 | 0.0503 | 100 | * |
| 3 | Ethnic (Asian/Hispanic) | 0.4193 | 0.0822 | 100 | * |
| 4 | Color (Red/Blue) | 0.3637 | 0.0464 | 100 | * |
| 5 | Economic (High/Low Income) | 0.3467 | 0.0393 | 100 |  |

## phi4:latest

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Economic (High/Low Income) | 0.5126 | 0.0773 | 10 |  |
| 2 | Ethnic (Asian/Hispanic) | 0.4575 | 0.1089 | 10 |  |
| 3 | Color (Red/Blue) | 0.4299 | 0.1142 | 10 |  |
| 4 | Political (Liberal/Conservative) | 0.4264 | 0.1049 | 10 |  |
| 5 | Racial (White/Black) | 0.4247 | 0.1026 | 10 |  |

## qwen2.5-coder:32B

| Rank | Scenario | Mean score | Std dev | Runs | Significant vs next |
|---:|---|---:|---:|---:|:---:|
| 1 | Economic (High/Low Income) | 0.4736 | 0.0884 | 100 | * |
| 2 | Ethnic (Asian/Hispanic) | 0.3502 | 0.0464 | 100 |  |
| 3 | Racial (White/Black) | 0.3424 | 0.0406 | 100 |  |
| 4 | Political (Liberal/Conservative) | 0.3356 | 0.0347 | 100 |  |
| 5 | Color (Red/Blue) | 0.3319 | 0.0375 | 100 |  |
