# Tier 3 Final Holdout Error Analysis

## Status

Phase 11A is post-final error analysis only. It inspects the already-opened `2025-26` final holdout to explain the frozen model result. It does not tune features, tune hyperparameters, tune draw thresholds, run competing models, save artifacts, or write to the database.

## Reproduced Frozen Setup

- Source table: `match_features_v3_elo`
- Training seasons: `2021-22`, `2022-23`, `2023-24`, `2024-25`
- Holdout season: `2025-26`
- Feature count: 32
- Draw threshold selected from training only: 0.24
- Model path: frozen `logistic_elo_expanding` only
- Official performance is not changed by this analysis

## Confusion Matrix

Argmax confusion matrix:

| actual | pred_H | pred_D | pred_A |
| ------ | ------ | ------ | ------ |
| H      | 130    | 0      | 32     |
| D      | 62     | 0      | 42     |
| A      | 58     | 1      | 55     |

Draw overlay confusion matrix:

| actual | pred_H | pred_D | pred_A |
| ------ | ------ | ------ | ------ |
| H      | 120    | 15     | 27     |
| D      | 56     | 8      | 40     |
| A      | 53     | 11     | 50     |

Argmax class metrics:

| class | precision | recall | f1     | support |
| ----- | --------- | ------ | ------ | ------- |
| H     | 0.5200    | 0.8025 | 0.6311 | 162     |
| D     | 0.0000    | 0.0000 | 0.0000 | 104     |
| A     | 0.4264    | 0.4825 | 0.4527 | 114     |

Draw overlay class metrics:

| class | precision | recall | f1     | support |
| ----- | --------- | ------ | ------ | ------- |
| H     | 0.5240    | 0.7407 | 0.6138 | 162     |
| D     | 0.2353    | 0.0769 | 0.1159 | 104     |
| A     | 0.4274    | 0.4386 | 0.4329 | 114     |

## Draw Failure Analysis

- Actual draw count: 104
- Argmax predicted draw count: 1
- Draw overlay predicted draw count: 34
- Argmax draw miss count: 104
- Draw overlay draw miss count: 96
- Average `P(D)` for actual draws: 0.2233
- Average `P(D)` for non-draws: 0.2206

`P(D)` distribution:

| bin       | count | actual_draw_count | actual_draw_rate | argmax_draw_count | overlay_draw_count |
| --------- | ----- | ----------------- | ---------------- | ----------------- | ------------------ |
| 0.00-0.15 | 40    | 9                 | 0.2250           | 0                 | 0                  |
| 0.15-0.20 | 99    | 29                | 0.2929           | 0                 | 0                  |
| 0.20-0.25 | 133   | 37                | 0.2782           | 0                 | 0                  |
| 0.25-0.30 | 72    | 19                | 0.2639           | 0                 | 13                 |
| 0.30-0.35 | 29    | 9                 | 0.3103           | 0                 | 15                 |
| 0.35+     | 7     | 1                 | 0.1429           | 1                 | 6                  |

## Confidence Analysis

| bin       | count | accuracy | mean_log_loss | total_log_loss | wrong_count | log_loss_share |
| --------- | ----- | -------- | ------------- | -------------- | ----------- | -------------- |
| 0.00-0.40 | 28    | 0.3929   | 1.1237        | 31.4627        | 17          | 0.0781         |
| 0.40-0.50 | 107   | 0.4393   | 1.0672        | 114.1855       | 60          | 0.2835         |
| 0.50-0.60 | 100   | 0.3900   | 1.1651        | 116.5105       | 61          | 0.2892         |
| 0.60-0.70 | 77    | 0.5584   | 1.0251        | 78.9328        | 34          | 0.1959         |
| 0.70+     | 68    | 0.6618   | 0.9080        | 61.7444        | 23          | 0.1533         |

- High-confidence wrong predictions with max probability >= 0.60: 57
- High-confidence wrong prediction rate: 0.1500

## Home/Away Bias

Actual distribution:

| result | count |
| ------ | ----- |
| H      | 162   |
| D      | 104   |
| A      | 114   |

Argmax predicted distribution:

| prediction | count |
| ---------- | ----- |
| H          | 250   |
| D          | 1     |
| A          | 129   |

Draw overlay predicted distribution:

| prediction | count |
| ---------- | ----- |
| H          | 229   |
| D          | 34    |
| A          | 117   |

Accuracy by predicted label:

| prediction | argmax_count | argmax_accuracy_when_predicted | overlay_count | overlay_accuracy_when_predicted |
| ---------- | ------------ | ------------------------------ | ------------- | ------------------------------- |
| H          | 250          | 0.5200                         | 229           | 0.5240                          |
| D          | 1            | 0.0000                         | 34            | 0.2353                          |
| A          | 129          | 0.4264                         | 117           | 0.4274                          |

- Argmax home overprediction count: 88
- Argmax home-rate gap: 0.2316
- Overlay home overprediction count: 67
- Overlay home-rate gap: 0.1763

## Elo Gap Analysis

| bin              | count | accuracy | log_loss | actual_home_rate | argmax_home_rate |
| ---------------- | ----- | -------- | -------- | ---------------- | ---------------- |
| strong away edge | 52    | 0.5769   | 1.0252   | 0.1346           | 0.0000           |
| slight away edge | 57    | 0.2807   | 1.2005   | 0.3333           | 0.2105           |
| balanced         | 58    | 0.4828   | 1.1080   | 0.4138           | 0.6379           |
| slight home edge | 71    | 0.4366   | 1.0907   | 0.4507           | 0.8592           |
| strong home edge | 142   | 0.5634   | 0.9816   | 0.5634           | 0.9859           |

- Big Elo favorite match count: 194
- Big Elo favorite wrong count: 84
- Big Elo favorite wrong rate: 0.4330

## Season-Stage Analysis

| stage  | count | accuracy | log_loss | brier_score | draw_f1 | actual_draw_rate | argmax_draw_rate | overlay_draw_rate | mean_p_draw |
| ------ | ----- | -------- | -------- | ----------- | ------- | ---------------- | ---------------- | ----------------- | ----------- |
| early  | 127   | 0.5354   | 1.0086   | 0.5960      | 0.0000  | 0.1969           | 0.0079           | 0.0630            | 0.2219      |
| middle | 127   | 0.4409   | 1.0994   | 0.6692      | 0.0000  | 0.3307           | 0.0000           | 0.1024            | 0.2268      |
| late   | 126   | 0.4841   | 1.0724   | 0.6466      | 0.0000  | 0.2937           | 0.0000           | 0.1032            | 0.2152      |

## Team-Level Error Analysis

Worst 10 teams by argmax prediction accuracy:

| team              | matches | argmax_accuracy | overlay_accuracy | draw_miss_count | mean_log_loss |
| ----------------- | ------- | --------------- | ---------------- | --------------- | ------------- |
| Everton           | 38      | 0.3421          | 0.3684           | 10              | 1.1484        |
| Sunderland        | 38      | 0.3684          | 0.3684           | 12              | 1.1762        |
| Bournemouth       | 38      | 0.3684          | 0.3947           | 18              | 1.1590        |
| Chelsea           | 38      | 0.3947          | 0.3684           | 10              | 1.1972        |
| Leeds             | 38      | 0.3947          | 0.3684           | 14              | 1.1257        |
| Crystal Palace    | 38      | 0.3947          | 0.4474           | 12              | 1.0701        |
| Brentford         | 38      | 0.4211          | 0.3684           | 11              | 1.1282        |
| Brighton          | 38      | 0.4211          | 0.3947           | 11              | 1.0998        |
| Aston Villa       | 38      | 0.4474          | 0.4211           | 8               | 1.1362        |
| Nottingham Forest | 38      | 0.4474          | 0.3684           | 11              | 1.0945        |

Best 10 teams by argmax prediction accuracy:

| team      | matches | argmax_accuracy | overlay_accuracy | draw_miss_count | mean_log_loss |
| --------- | ------- | --------------- | ---------------- | --------------- | ------------- |
| Arsenal   | 38      | 0.7105          | 0.7105           | 7               | 0.7856        |
| Burnley   | 38      | 0.6579          | 0.6053           | 10              | 0.8756        |
| West Ham  | 38      | 0.6579          | 0.5526           | 9               | 0.9757        |
| Fulham    | 38      | 0.6316          | 0.6842           | 7               | 0.8538        |
| Wolves    | 38      | 0.5789          | 0.5526           | 11              | 0.9711        |
| Liverpool | 38      | 0.5526          | 0.6053           | 9               | 1.0289        |
| Man City  | 38      | 0.5263          | 0.5263           | 9               | 0.9817        |
| Newcastle | 38      | 0.4737          | 0.4474           | 7               | 1.0537        |
| Man Utd   | 38      | 0.4737          | 0.3684           | 11              | 1.1494        |
| Tottenham | 38      | 0.4737          | 0.4474           | 11              | 1.1911        |

Teams with the most draw misses:

| team              | matches | argmax_accuracy | overlay_accuracy | draw_miss_count | mean_log_loss |
| ----------------- | ------- | --------------- | ---------------- | --------------- | ------------- |
| Bournemouth       | 38      | 0.3684          | 0.3947           | 18              | 1.1590        |
| Leeds             | 38      | 0.3947          | 0.3684           | 14              | 1.1257        |
| Crystal Palace    | 38      | 0.3947          | 0.4474           | 12              | 1.0701        |
| Sunderland        | 38      | 0.3684          | 0.3684           | 12              | 1.1762        |
| Brentford         | 38      | 0.4211          | 0.3684           | 11              | 1.1282        |
| Brighton          | 38      | 0.4211          | 0.3947           | 11              | 1.0998        |
| Man Utd           | 38      | 0.4737          | 0.3684           | 11              | 1.1494        |
| Nottingham Forest | 38      | 0.4474          | 0.3684           | 11              | 1.0945        |
| Tottenham         | 38      | 0.4737          | 0.4474           | 11              | 1.1911        |
| Wolves            | 38      | 0.5789          | 0.5526           | 11              | 0.9711        |

Teams most often overpredicted as winners:

| team           | overpredicted_as_winner_count |
| -------------- | ----------------------------- |
| Crystal Palace | 17                            |
| Chelsea        | 16                            |
| Liverpool      | 16                            |
| Newcastle      | 15                            |
| Brighton       | 14                            |
| Man City       | 14                            |
| Bournemouth    | 13                            |
| Brentford      | 12                            |
| Everton        | 12                            |
| Aston Villa    | 11                            |

## Promoted/Returning Team Analysis

Promoted or returning teams from `elo_ratings_v3`: `Burnley`, `Leeds`, `Sunderland`

| segment                             | count | accuracy | log_loss | brier_score | draw_rate |
| ----------------------------------- | ----- | -------- | -------- | ----------- | --------- |
| involving promoted_or_returning     | 108   | 0.4722   | 1.0646   | 0.6444      | 0.3241    |
| not involving promoted_or_returning | 272   | 0.4926   | 1.0583   | 0.6344      | 0.2537    |

## Biggest Log-Loss Errors

| match_id | match_date | home_team      | away_team         | actual_result | predicted_argmax | P(H)   | P(D)   | P(A)   | overlay_prediction | final_score | individual_log_loss |
| -------- | ---------- | -------------- | ----------------- | ------------- | ---------------- | ------ | ------ | ------ | ------------------ | ----------- | ------------------- |
| 1532     | 2025-08-23 | Man City       | Tottenham         | A             | H                | 0.8188 | 0.1316 | 0.0496 | H                  | 0-2         | 3.0031              |
| 1776     | 2026-02-11 | Crystal Palace | Burnley           | A             | H                | 0.8032 | 0.1336 | 0.0632 | H                  | 2-3         | 2.7617              |
| 1543     | 2025-08-30 | Sunderland     | Brentford         | H             | A                | 0.0759 | 0.1796 | 0.7445 | A                  | 2-1         | 2.5789              |
| 1896     | 2026-05-24 | Man City       | Aston Villa       | A             | H                | 0.7887 | 0.1337 | 0.0775 | H                  | 1-2         | 2.5569              |
| 1599     | 2025-10-19 | Liverpool      | Man Utd           | A             | H                | 0.6800 | 0.2293 | 0.0907 | H                  | 1-2         | 2.4005              |
| 1698     | 2025-12-27 | Chelsea        | Aston Villa       | A             | H                | 0.5419 | 0.3654 | 0.0927 | H                  | 1-2         | 2.3781              |
| 1749     | 2026-01-25 | Arsenal        | Man Utd           | A             | H                | 0.7214 | 0.1851 | 0.0935 | H                  | 2-3         | 2.3696              |
| 1784     | 2026-02-21 | Chelsea        | Burnley           | D             | H                | 0.8671 | 0.0936 | 0.0393 | H                  | 1-1         | 2.3691              |
| 1805     | 2026-03-03 | Wolves         | Liverpool         | H             | A                | 0.0947 | 0.1374 | 0.7679 | A                  | 2-1         | 2.3570              |
| 1734     | 2026-01-17 | Liverpool      | Burnley           | D             | H                | 0.8731 | 0.0960 | 0.0309 | H                  | 1-1         | 2.3436              |
| 1792     | 2026-02-27 | Wolves         | Aston Villa       | H             | A                | 0.1064 | 0.1515 | 0.7422 | A                  | 2-0         | 2.2409              |
| 1635     | 2025-11-22 | Liverpool      | Nottingham Forest | A             | H                | 0.6761 | 0.2140 | 0.1099 | H                  | 0-3         | 2.2084              |
| 1618     | 2025-11-02 | West Ham       | Newcastle         | H             | A                | 0.1147 | 0.1295 | 0.7558 | A                  | 3-1         | 2.1653              |
| 1821     | 2026-03-16 | Brentford      | Wolves            | D             | H                | 0.8247 | 0.1152 | 0.0601 | H                  | 2-2         | 2.1612              |
| 1781     | 2026-02-18 | Wolves         | Arsenal           | D             | A                | 0.0484 | 0.1220 | 0.8296 | A                  | 2-2         | 2.1034              |
| 1820     | 2026-03-15 | Liverpool      | Tottenham         | D             | H                | 0.8132 | 0.1228 | 0.0639 | H                  | 1-1         | 2.0969              |
| 1658     | 2025-12-03 | Leeds          | Chelsea           | H             | A                | 0.1229 | 0.2088 | 0.6683 | A                  | 3-1         | 2.0962              |
| 1546     | 2025-08-30 | Leeds          | Newcastle         | D             | A                | 0.0867 | 0.1274 | 0.7860 | A                  | 0-0         | 2.0606              |
| 1602     | 2025-10-25 | Chelsea        | Sunderland        | A             | H                | 0.6454 | 0.2219 | 0.1326 | H                  | 1-2         | 2.0203              |
| 1875     | 2026-05-10 | Burnley        | Aston Villa       | D             | A                | 0.1343 | 0.1365 | 0.7292 | A                  | 2-2         | 1.9915              |

## Watched Table Counts

| table                                 | before | after | changed |
| ------------------------------------- | ------ | ----- | ------- |
| historical_matches                    | 1900   | 1900  | 0       |
| historical_understat_xg               | 1900   | 1900  | 0       |
| match_features_v3_base                | 1900   | 1900  | 0       |
| elo_ratings_v3                        | 1900   | 1900  | 0       |
| match_features_v3_elo                 | 1900   | 1900  | 0       |
| standings_before_match_v3             | 3800   | 3800  | 0       |
| match_features_v3_pressure_experiment | 1900   | 1900  | 0       |
| match_features_v3_style_experiment    | 1900   | 1900  | 0       |
| match_features_v3_h2h_experiment      | 1900   | 1900  | 0       |
| players                               | 841    | 841   | 0       |
| teams                                 | 20     | 20    | 0       |
| fixtures                              | 380    | 380   | 0       |
| gameweeks                             | 38     | 38    | 0       |
| understat_xg                          | 380    | 380   | 0       |
| understat_team_history                | 760    | 760   | 0       |
| player_gameweek_history               | 29747  | 29747 | 0       |
| player_gameweek_features              | 29747  | 29747 | 0       |

## Summary Diagnosis

- Reproduced argmax accuracy was 0.4868 with log_loss 1.0601.
- Draw underprediction is the main issue: 104 actual draws, 1 argmax predicted draws, and 104 argmax draw misses (1.0000 of actual draws).
- Home bias is severe: argmax predicted 250 home wins against 162 actual home wins, a home-rate gap of 0.2316.
- Confidence was not reliable enough in the top bins: 57 wrong predictions had max probability >= 0.60.
- Big Elo favorites were not safe: 84 of 194 matches with absolute adjusted Elo gap >= 100 were wrong.
- The weakest season stage by accuracy was middle at 0.4409.
- Matches involving promoted_or_returning teams had 108 rows, 0.4722 accuracy, and 1.0646 log loss.
- Future work should focus on more seasons, better pre-match draw modeling, calibration/decision-rule research on development data only, and reliable dated team-availability sources. Any improved model needs a new untouched future holdout.

## Guardrails

- Do not claim this analysis improves official Tier 3 performance.
- Do not tune on `2025-26`.
- Any improved future model needs a new untouched future holdout.
