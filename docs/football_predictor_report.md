# Football Predictor Model - Final Project Report

**Author:** Purav Desai  
**Report date:** June 2026  
**Project type:** Local machine learning research and production pipeline  
**Domain:** Premier League match prediction and Fantasy Premier League squad optimization

> This project is not betting advice. It is a local machine learning and football analytics project built to study time-safe prediction, feature engineering, and production pipeline design.

## Table of Contents

1. Executive Summary
2. Why Football Prediction Is Hard
3. Tier 1 - Foundation
4. Tier 2 - Advanced Local ML System
5. Tier 3 - Multi-Season Research System
6. Tier 3 Experiments and Decisions
7. Final Holdout Evaluation
8. Production Pipeline
9. Streamlit Production Dashboard
10. Domain Questions, Methods, Successes, and Failures
11. Accuracy and Metrics Summary
12. Security, GitHub, and Deployment Safety
13. Original Master Reference and Extra Design Questions
14. Future Work
15. Conclusion

## 1. Executive Summary

The Football Predictor Model is a local machine learning project for predicting Premier League match outcomes and supporting Fantasy Premier League squad optimization. It started as a foundation project using the official FPL API, PostgreSQL, basic feature engineering, a match classifier, and a Streamlit app. It later grew into a multi-tier system with Understat expected goals data, an XGBoost FPL points model, strict leakage checks, multi-season Premier League training data, Elo ratings, final holdout testing, and a production-ready local pipeline.

The project evolved through three clear stages:

- **Tier 1:** proved the end-to-end idea with FPL API data, PostgreSQL, SQL feature views, basic match prediction, scoreline regression, a rule-based FPL optimizer, and a local Streamlit app.
- **Tier 2:** added Understat xG, stronger match features, a real XGBoost FPL points model, leakage-safe player gameweek features, and a more reliable local Streamlit system.
- **Tier 3:** added five seasons of Premier League data, walk-forward validation, Elo ratings, final holdout evaluation, production model training, weekly pipeline scripts, and a read-only production dashboard.

The final Tier 3 probability champion is `logistic_elo_expanding`. The production model trained after final evaluation is `production_logistic_elo_v3`, using 32 base + Elo features and seasons 2021-22 through 2025-26.

The final holdout result on 2025-26 was honest but imperfect:

| Final holdout model | Accuracy | Log loss | Brier | Draw F1 |
| --- | ---: | ---: | ---: | ---: |
| `logistic_elo_expanding` argmax | 0.4868 | 1.0601 | 0.6372 | 0.0000 |
| Draw overlay helper | 0.4684 | 1.0601 | 0.6372 | 0.1159 |

The most important lesson from the project is that football modeling is not just about adding more features. It is about avoiding leakage, respecting time order, rejecting features that fail validation, and being honest when an idea sounds good but does not improve out-of-sample performance.

## 2. Why Football Prediction Is Hard

Premier League match prediction is a noisy three-class problem: home win, draw, or away win. The sport is low-scoring, match outcomes are affected by small events, and draws are especially hard to classify. A model can look strong in training while failing in realistic future prediction if it accidentally uses future information.

The main risks identified throughout this project were:

- **Random train/test splits:** invalid for time-ordered football data because they can train on future matches and test on earlier ones.
- **H2H leakage:** unsafe with one season of data because same-season matchups can leak future results.
- **Same-gameweek FPL leakage:** dangerous in double gameweeks because a second fixture can accidentally see data from the first fixture of the same gameweek.
- **Modern strength values applied historically:** current FPL team strength or FDR values can leak present-day information into old seasons.
- **Final-table leakage:** pressure, rank, and league-table features must use only standings before the match, never end-of-season standings.
- **Overfitting through domain ideas:** style, pressure, rivalry, sentiment, and manager features sound useful, but they must survive time-safe validation.

The project's central principle became simple:

> A football feature is not accepted because it sounds right. It is accepted only if it survives chronological validation without leakage.

## 3. Tier 1 - Foundation

### Goal

Tier 1 was built to prove that the full system could work end to end. The goal was not to build the most advanced football model immediately. The goal was to collect real football and FPL data, store it properly, engineer features, train a basic model, optimize an FPL squad, and show the result in a local Streamlit app.

### Data Sources

Tier 1 used the official Fantasy Premier League API:

- `https://fantasy.premierleague.com/api/bootstrap-static/`
- `https://fantasy.premierleague.com/api/fixtures/`

### Database Foundation

The database was local PostgreSQL:

- Database: `football_db`
- Host: `localhost:5432`
- Credentials stored in local `.env`

Core tables:

- `players`
- `teams`
- `fixtures`
- `gameweeks`

SQL feature views:

- `match_results`
- `team_season_stats`
- `home_form`
- `away_form`
- `h2h_stats`
- `match_features`
- `player_fpl_features`

### Match Prediction

Tier 1 trained win/draw/loss classifiers using 12 pre-match features:

- `home_form_scored`
- `home_form_conceded`
- `home_clean_sheet_rate`
- `away_form_scored`
- `away_form_conceded`
- `away_clean_sheet_rate`
- `home_fdr`
- `away_fdr`
- `strength_overall_home`
- `strength_overall_away`
- `home_team_away_str`
- `away_team_home_str`

H2H features were intentionally excluded from training. With only one season of data, H2H created leakage and unrealistic accuracy.

Tier 1 used a time-based split instead of a random split.

### Tier 1 Metrics

| Model | Accuracy | Notes |
| --- | ---: | --- |
| Logistic Regression match classifier | 0.556 | Best Tier 1 classifier |
| XGBoost match classifier | 0.514 | Tested but weaker than logistic |

Score regression was weak:

| Target | MAE | R2 |
| --- | ---: | ---: |
| Home Goals | 1.056 | -0.295 |
| Away Goals | 0.833 | -0.160 |

The negative R2 scores showed that exact score prediction was not reliable enough to be treated as the main output.

### FPL Optimizer

Tier 1 included a rule-based Fantasy Premier League optimizer. It used PuLP linear programming to select a valid 15-player squad.

Squad constraints:

- 2 goalkeepers
- 5 defenders
- 5 midfielders
- 3 forwards
- maximum 3 players per club
- budget <= 100.0
- available players only

The optimizer also selected starters, bench players, and a captain. FPL points were estimated through a rule-based formula, not a trained ML model.

### Streamlit App

Tier 1 had a local Streamlit app with three pages:

- Match Predictor
- FPL Team Selector
- About

The app could show match probabilities, approximate scoreline predictions, optimized FPL squads, and captain recommendations.

### Tier 1 Limitations

Tier 1 proved the system could work, but it had major limitations:

- only one season of data
- no xG or xGA data
- weak exact score prediction
- rule-based FPL points
- unsafe H2H with one season
- no Elo ratings
- no Poisson score matrix
- no multi-season validation
- local-only deployment

## 4. Tier 2 - Advanced Local ML System

### Goal

Tier 2 improved Tier 1 by adding better football signal and replacing the rule-based FPL points logic with a real machine learning model. It turned the project from a basic FPL-data predictor into a stronger local football ML system.

The two main weaknesses from Tier 1 were:

- match prediction lacked chance-quality data such as expected goals
- FPL optimization used a manual formula instead of learned player point prediction

### New Data Sources

Tier 2 continued using the FPL API and added:

- FPL `element-summary/{player_id}` endpoint for player gameweek history
- Understat EPL data endpoint for xG and tactical match data

### Understat Integration

Tier 2 added:

- `understat_xg`
- `understat_team_history`

Understat coverage:

| Table | Rows |
| --- | ---: |
| `understat_xg` | 380 |
| `understat_team_history` | 760 |

The Understat integration included:

- match date
- home team
- away team
- home xG
- away xG
- home goals
- away goals
- PPDA
- deep completions
- npxG
- npxGA
- xPts

An important Tier 2 fix was changing the Understat join to use match date instead of only scoreline. Joining by scoreline is fragile because many matches can share the same score.

### Tier 2 Match Model

Tier 2 added four rolling xG/xGA features to the original 12 Tier 1 features:

- `home_xg_last5`
- `home_xga_last5`
- `away_xg_last5`
- `away_xga_last5`

The final Tier 2 match classifier used 16 features and XGBoost.

| Model | Features | Accuracy |
| --- | ---: | ---: |
| Logistic Regression | 12 baseline features | 0.570 |
| XGBoost | 12 baseline features | 0.506 |
| Logistic Regression | 16 xG features | 0.532 |
| XGBoost | 16 xG features | 0.570 |

Final Tier 2 selected classifier:

- XGBoost
- 16 features
- 0.570 test accuracy

Remembered TimeSeries CV for xG 16-feature XGBoost:

- 0.457 +/- 0.113

### Tier 2 Score Model

Score prediction improved slightly after adding xG, but remained weak:

| Target | MAE | R2 |
| --- | ---: | ---: |
| Home Goals | 0.975 | -0.103 |
| Away Goals | 0.747 | -0.025 |

The decision was to keep scoreline prediction as a supporting output, not the main model claim.

### FPL XGBoost Points Model

Tier 2 built a real FPL points model using XGBoost regression.

New tables:

- `player_gameweek_history`
- `player_gameweek_features`

FPL coverage:

| Metric | Value |
| --- | ---: |
| `player_gameweek_history` rows | 29,747 |
| `player_gameweek_features` rows | 29,747 |
| Mature rows | 27,224 |
| Players | 841 |
| Gameweeks | 38 |
| Null `total_points` rows | 0 |

FPL points model comparison:

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Baseline `points_avg_last5` | 1.007 | 2.046 | 0.194 |
| XGBoost FPL points model | 0.926 | 1.859 | 0.334 |

The XGBoost FPL points model beat the baseline on all three metrics and replaced the rule-based Tier 1 points estimate.

### Double-Gameweek Leakage Fix

Double gameweeks create a special FPL leakage risk. If a player has two fixtures in one gameweek, the second fixture can accidentally use outcome data from the first fixture of the same gameweek. Before the FPL deadline, both fixtures are still future events.

Tier 2 fixed this by using previous gameweeks only for rolling player features.

Verification:

```text
Duplicate player-gameweek groups: 409
Same-gameweek historical feature mismatch groups: 0
FPL leakage column check passed
```

### Other Tier 2 Fixes

Tier 2 also added or fixed:

- `fixtures.kickoff_time`
- `players.is_available` changed to boolean
- team name normalization between FPL and Understat
- pre-deadline player availability refresh
- model feature ordering through `model_features.json`
- same-team warning in the app
- score/result contradiction fix
- progress bar probability clipping
- cleaner native Streamlit UI after rejecting an over-customized dark UI

### Tier 2 Rejected Experiments

#### Style Clustering

KMeans style clustering used Understat features such as PPDA, deep completions, xG, xGA, npxG, npxGA, xPts, goals scored, and goals conceded. Four clusters were created:

- High Press
- Direct Attack
- Compact Defense
- Low Control

Result:

| Model | Features | Accuracy |
| --- | ---: | ---: |
| XGBoost | 16 xG features | 0.570 |
| XGBoost | 20 xG + style features | 0.506 |

Decision: rejected. The idea was useful research, but it hurt validation.

#### Position-Specific FPL Models

Separate XGBoost models were trained for GK, DEF, MID, and FWD.

| Model | MAE |
| --- | ---: |
| Single XGBoost | 0.926 |
| Position-specific combined | 0.948 |

Decision: rejected. The single global model performed better.

### Tier 2 Limitations

Tier 2 was much stronger than Tier 1, but still had limits:

- still only one completed Premier League season for match modeling
- exact score prediction still weak
- no Elo ratings
- no Poisson scoreline matrix
- no multi-season H2H
- no real manager regime features
- no sentiment or morale layer
- local-only deployment

These limitations led directly to Tier 3.

## 5. Tier 3 - Multi-Season Research System

### Goal

Tier 3 was the final advanced modeling phase. Its first priority was not flashy features. It was multi-season data.

The core rule was:

> Multi-season match data must come before H2H, manager, sentiment, style, or pressure features.

### Data Foundation

Tier 3 used football-data.co.uk Premier League CSVs and historical Understat xG data.

Seasons:

- 2021-22
- 2022-23
- 2023-24
- 2024-25
- 2025-26

Main Tier 3 tables:

| Table | Rows |
| --- | ---: |
| `historical_matches` | 1900 |
| `historical_understat_xg` | 1900 |
| `match_features_v3_base` | 1900 |
| `elo_ratings_v3` | 1900 |
| `match_features_v3_elo` | 1900 |
| `match_features_v3_h2h_experiment` | 1900 |
| `match_features_v3_style_experiment` | 1900 |
| `standings_before_match_v3` | 3800 |
| `match_features_v3_pressure_experiment` | 1900 |

Each season contains 380 matches and 20 teams.

### Validation Strategy

Tier 3 used chronological walk-forward validation only.

Development seasons:

- 2021-22
- 2022-23
- 2023-24
- 2024-25

Reserved final holdout:

- 2025-26

Walk-forward folds:

| Fold | Train seasons | Validation season |
| --- | --- | --- |
| Fold 1 | 2021-22, 2022-23 | 2023-24 |
| Fold 2 | 2021-22, 2022-23, 2023-24 | 2024-25 |

The 2025-26 season was excluded from model selection and evaluated only after the final candidate freeze.

## 6. Tier 3 Experiments and Decisions

### Baseline Models

Development validation:

| Model | Accuracy | Log loss | Brier | Draw recall |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline | 0.4342 | 1.0682 | 0.6469 | 0.0000 |
| Logistic baseline | 0.5474 | 0.9890 | 0.5875 | 0.0793 |
| XGBoost baseline | 0.5237 | 1.0176 | 0.6052 | 0.0588 |

Before Elo, logistic regression was the strongest baseline.

### Elo Ratings

Tier 3 added a time-safe Elo system.

Elo constants:

- initial Elo: 1500.0
- promoted or returning team Elo: 1400.0
- home advantage: 50.0
- K factor: 20.0

Elo ratings were updated chronologically. Only pre-match Elo values were added to match features. Post-match Elo columns were explicitly forbidden from the model feature table.

Development results:

| Model | Accuracy | Log loss | Brier | Draw recall |
| --- | ---: | ---: | ---: | ---: |
| `logistic_base` | 0.5474 | 0.9890 | 0.5875 | 0.0793 |
| `logistic_elo` | 0.5579 | 0.9705 | 0.5730 | 0.0534 |
| `xgb_base` | 0.5237 | 1.0176 | 0.6052 | 0.0588 |
| `xgb_elo` | 0.5382 | 0.9984 | 0.5901 | 0.0764 |

Decision: Elo was accepted as a core Tier 3 feature group.

### Poisson Scoreline Matrix

A Poisson scoreline model was tested as a scoreline and probability diagnostic.

Aggregate development results:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5500 |
| Log loss | 0.9775 |
| Brier | 0.5802 |
| Draw recall | 0.0000 |
| Exact score accuracy | 0.1145 |
| Home goals MAE | 0.9868 |
| Away goals MAE | 0.8882 |
| Total goals MAE | 1.4145 |

Decision: useful as a diagnostic scoreline layer, but not promoted as the W/D/L champion.

### H2H Retest

H2H was retested only after multi-season data existed. The builder used prior-only meetings where the source event time was strictly earlier than the target match time.

H2H evidence:

- rows thresholded because `h2h_matches_prior < 3`: 807
- rows with usable H2H after threshold: 713

Result: H2H improved some draw behavior but worsened log loss.

Decision: rejected / kept experimental.

### Style and Tactical Retest

Style features were rebuilt using only prior Understat xG/xGA style proxies.

Results:

| Model | Accuracy | Log loss | Brier | Draw F1 |
| --- | ---: | ---: | ---: | ---: |
| `logistic_elo` | 0.5579 | 0.9705 | 0.5730 | 0.0893 |
| `logistic_elo_style` | 0.5474 | 1.0091 | 0.5945 | 0.1526 |
| `xgb_elo` | 0.5382 | 0.9984 | 0.5901 | 0.1118 |
| `xgb_elo_style` | 0.5237 | 1.0143 | 0.5991 | 0.1263 |

Decision: rejected. Draw F1 improved, but log loss and Brier worsened too much. Some style features were also redundant with existing rolling xG/xGA features.

### Match Pressure Index

Pressure features were built from pre-match standings only. The system avoided final table rank, same-match result, future match data, rivalry, derby, and same-date result leakage.

Results:

| Model | Accuracy | Log loss | Brier | Draw F1 |
| --- | ---: | ---: | ---: | ---: |
| `logistic_elo` | 0.5579 | 0.9705 | 0.5730 | 0.0893 |
| `logistic_elo_pressure` | 0.5447 | 1.0024 | 0.5904 | 0.1643 |
| `xgb_elo` | 0.5382 | 0.9984 | 0.5901 | 0.1118 |
| `xgb_elo_pressure` | 0.5329 | 0.9918 | 0.5890 | 0.1387 |

Decision: rejected. Pressure helped draw F1 but hurt probability metrics and did not beat the logistic Elo champion.

### Draw Overlay

Many rejected feature groups improved draw behavior but damaged probability quality. The final solution was to decouple probability estimation from draw-risk labeling.

The draw overlay:

- uses the main model's probabilities
- changes hard labels only
- does not change probabilities
- does not train a binary draw classifier
- does not use H2H, style, pressure, Poisson, odds, or sentiment

Development result:

| Mode | Accuracy | Log loss | Brier | Draw recall | Draw precision | Draw F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Argmax | 0.5579 | 0.9705 | 0.5730 | 0.0534 | 0.3271 | 0.0893 |
| Overlay | 0.5618 | 0.9705 | 0.5730 | 0.1520 | 0.3740 | 0.2132 |

Decision: accepted only as an optional hard-label draw-risk helper.

### Final Tier 3 Development Champion

The final research probability champion was:

- `logistic_elo_expanding`

Development metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5579 |
| Log loss | 0.9705 |
| Brier | 0.5730 |
| Draw F1 | 0.0893 |

## 7. Final Holdout Evaluation

### Frozen Setup

The final model candidate was frozen before 2025-26 holdout evaluation.

- probability model: `logistic_elo_expanding`
- source table: `match_features_v3_elo`
- allowed feature family: base + Elo only
- feature count: 32
- training seasons: 2021-22 through 2024-25
- holdout season: 2025-26
- holdout rows: 380
- draw overlay threshold selected from training only: 0.24
- no tuning on 2025-26

### Actual Holdout Distribution

| Result | Count |
| --- | ---: |
| H | 162 |
| D | 104 |
| A | 114 |

### Argmax Final Holdout Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4868 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw recall | 0.0000 |
| Draw precision | 0.0000 |
| Draw F1 | 0.0000 |

Predicted distribution:

| Prediction | Count |
| --- | ---: |
| H | 250 |
| D | 1 |
| A | 129 |

### Draw Overlay Final Holdout Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4684 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw recall | 0.0769 |
| Draw precision | 0.2353 |
| Draw F1 | 0.1159 |

Predicted distribution:

| Prediction | Count |
| --- | ---: |
| H | 229 |
| D | 34 |
| A | 117 |

The draw overlay changed 33 labels to draw. Log loss and Brier remained unchanged because the overlay does not alter probabilities.

### Final Holdout Interpretation

Final holdout performance was weaker than development validation. The largest weakness was draw prediction:

- actual draws: 104
- argmax predicted draws: 1
- argmax missed all 104 actual draws

The model also had a clear home-win bias:

- actual home wins: 162
- predicted home wins: 250

The draw overlay improved draw F1 but reduced final holdout accuracy. Therefore, it should be described only as an optional draw-risk helper, not as an upgraded probability model.

The official final probability model remains `logistic_elo_expanding`.

## 8. Production Pipeline

After the final holdout was documented, the production model was trained on all five available seasons:

- 2021-22
- 2022-23
- 2023-24
- 2024-25
- 2025-26

Production model:

- `production_logistic_elo_v3`
- 32 features
- median imputer
- standard scaler
- logistic regression
- production draw threshold: 0.30

Production artifacts:

```text
models/saved/production_logistic_elo_v3.pkl
models/saved/production_features_v3.json
models/saved/production_draw_threshold_v3.json
models/saved/production_metadata_v3.json
```

These artifacts are local-only and ignored by Git.

Production scripts:

| Script | Purpose |
| --- | --- |
| `src/production/train_production_model.py` | train local production model and save artifacts |
| `src/production/weekly_ingest.py` | ingest FPL, football-data, and Understat source data |
| `src/production/build_upcoming_features.py` | build upcoming match features |
| `src/production/predict_production_matches.py` | generate production predictions |
| `src/production/score_predictions.py` | score predictions once actual results exist |
| `src/production/run_weekly_pipeline.py` | orchestrate ingest, feature build, prediction, and scoring |

Production tables include:

- `production_ingestion_runs`
- `production_fpl_bootstrap_snapshots`
- `production_fpl_fixture_snapshots`
- `production_data_freshness`
- `production_football_data_match_staging`
- `production_understat_xg_staging`
- `production_prediction_runs`
- `production_match_predictions`
- `production_team_name_mapping`
- `production_upcoming_match_features_v3`
- `production_model_health_log`
- `elo_current_v3`

Current production status:

- FPL ingestion works.
- football-data 2026-27 CSV may be unavailable until the source publishes it.
- Understat 2026 data may be unavailable until the source publishes it.
- Upcoming feature rows are currently zero when no unfinished fixtures exist.
- Prediction generation skips safely when there are no feature rows.
- Scoring skips safely when there are no unscored predictions.
- No fake fixtures, fake predictions, or fake scores are created.

## 9. Streamlit Production Dashboard

The project includes a read-only production dashboard:

```bash
streamlit run app/production_dashboard.py
```

Dashboard tabs:

- Overview
- Pipeline Status
- Predictions
- Model Health
- Reports
- How To Run

The dashboard checks:

- production artifact availability
- model metadata
- feature count
- draw threshold
- pipeline table state
- prediction table state
- health logs
- Tier 3 reports

If PostgreSQL is unreachable, the dashboard handles the unavailable database state cleanly instead of crashing or exposing secrets.

## 10. Domain Questions, Methods, Successes, and Failures

### Can football matches be predicted from recent form and FPL team strength?

Tier 1 answered yes, but only as a foundation. Logistic regression reached 0.556 accuracy using time-based validation. This proved the end-to-end idea but also showed that one-season data and basic form features were not enough for a final model.

### Should H2H be used?

Not with one season. H2H caused leakage risk and unrealistic behavior in Tier 1. It was excluded from Tier 1 and Tier 2 models. In Tier 3, H2H was retested only after multi-season data existed and only with prior-only filtering. It still worsened log loss and was rejected.

### Why not use random train/test splits?

Football data is chronological. Random splits can train on future fixtures and test on earlier ones. The project used time-based split, complete-gameweek split, TimeSeries CV utilities, and walk-forward validation instead.

### Did xG help?

Yes. Understat xG helped Tier 2 XGBoost match prediction. The xG-enhanced XGBoost model reached 0.570 accuracy compared with 0.506 for the 12-feature XGBoost baseline.

### Why was exact score prediction weak?

Exact football score prediction is much harder than W/D/L classification. Tier 1 score regressors had negative R2, and Tier 2 still had negative R2 even after xG. Tier 3 tested Poisson as a diagnostic layer, but W/D/L probability modeling remained the main task.

### Did tactical style help?

No. Style clustering and later Tier 3 style proxies sounded useful but failed validation. The project kept the code and analysis as research, but rejected the features from the final model.

### Did match pressure help?

Partially. Pressure improved draw F1 but worsened log loss and Brier. It was rejected because the final model prioritizes probability quality.

### Did sentiment, manager, injuries, and morale make it into the model?

No. These ideas were considered future experimental features, but they require reliable dated data. They were not included in the final Tier 3 model because the project avoided fake or weakly sourced features.

### Did FPL ML improve over rule-based scoring?

Yes. The XGBoost FPL points model improved over the last-5 average baseline:

- baseline MAE: 1.007
- XGBoost MAE: 0.926

### What was the biggest final weakness?

Draw prediction. The final holdout had 104 actual draws, but the argmax model predicted only one draw.

## 11. Successes and Failures

| Idea / Method | Tier | Result | Decision |
| --- | --- | --- | --- |
| FPL API ingestion | Tier 1 | Worked | Keep |
| PostgreSQL schema | Tier 1 | Worked | Keep |
| SQL feature views | Tier 1 | Worked | Keep |
| Time-based split | Tier 1 | Worked | Keep |
| H2H in one-season model | Tier 1 | Leakage risk | Reject |
| Rule-based FPL optimizer | Tier 1 | Worked as foundation | Replace later |
| Understat xG | Tier 2 | Improved model signal | Keep |
| xG rolling features | Tier 2 | Improved XGBoost accuracy | Keep |
| FPL XGBoost points model | Tier 2 | Beat baseline | Keep |
| Double-gameweek leakage fix | Tier 2 | Passed checks | Keep |
| Style clustering | Tier 2 | Hurt accuracy | Reject |
| Position-specific FPL models | Tier 2 | Worse MAE | Reject |
| Multi-season PL ingestion | Tier 3 | 1900 rows loaded | Keep |
| Elo ratings | Tier 3 | Improved probability metrics | Keep |
| Poisson matrix | Tier 3 | Useful diagnostic | Do not promote |
| H2H retest | Tier 3 | Worsened log loss | Reject / experimental |
| Style retest | Tier 3 | Worsened log loss and Brier | Reject |
| Pressure index | Tier 3 | Improved draw F1, hurt probabilities | Reject |
| Draw overlay | Tier 3 | Helped draw F1 without changing probabilities | Optional helper |
| Production pipeline | Production | Built and safe-skipping | Keep |
| Streamlit production dashboard | Production | Built and read-only | Keep |

## 12. Accuracy and Metrics Summary

| Tier | Model / System | Accuracy | Other Metrics | Decision |
| --- | --- | ---: | --- | --- |
| Tier 1 | Logistic Regression match classifier | 0.556 | score R2 negative | Best Tier 1 classifier |
| Tier 1 | XGBoost match classifier | 0.514 | FPL optimizer metric unknown | Rejected vs logistic |
| Tier 2 | XGBoost 16-feature xG classifier | 0.570 | TimeSeries CV 0.457 +/- 0.113 | Final Tier 2 classifier |
| Tier 2 | FPL XGBoost points model | N/A | MAE 0.926, RMSE 1.859, R2 0.334 | Accepted |
| Tier 3 development | `logistic_elo_expanding` | 0.5579 | log_loss 0.9705, Brier 0.5730 | Final research champion |
| Tier 3 final holdout | `logistic_elo_expanding` argmax | 0.4868 | log_loss 1.0601, Brier 0.6372, draw F1 0.0000 | Official final probability model |
| Tier 3 final holdout | Draw overlay | 0.4684 | draw F1 0.1159 | Optional draw-risk helper |

## 13. Security, GitHub, and Deployment Safety

The project was prepared for GitHub with safety rules:

- `.env` is not tracked.
- `.streamlit/secrets.toml` is not tracked.
- raw historical CSVs are ignored.
- production model artifacts are ignored.
- database dumps are not committed.
- production artifacts are local-only.
- no Supabase deployment is live yet.
- no APK is live yet.
- no betting advice is provided.

The repository has been merged into `main` with Tier 3 production-ready local code. Raw CSVs and production model artifacts were kept out of Git.

## 13. Original Master Reference and Extra Design Questions

Before Tier 1 was built, an early master reference document mapped out a much larger football analytics system. That first blueprint is useful because it shows how the project began: not only as a classifier, but as a broader sports analytics architecture with match prediction, FPL optimization, league simulation, deployment planning, and long-term research ideas.

This section records those original ideas honestly. Some became part of the final project. Some were tested and rejected. Some remain future work. They should not be confused with implemented Tier 3 production features.

### Original Three-System Blueprint

The first master reference described three connected systems sharing one PostgreSQL database, one Python pipeline, and one app layer:

| Planned system | Original purpose | Final project status |
| --- | --- | --- |
| Score / match predictor | predict PL match scoreline or W/D/L outcome | implemented as W/D/L model; exact score kept as supporting/diagnostic |
| League winner simulator | estimate title-race probabilities with Monte Carlo | not implemented in the final Tier 3 system |
| FPL optimizer | choose best 15-player FPL squad and captain | implemented locally; improved in Tier 2 with XGBoost points model |

The final project focused mainly on the match predictor and FPL optimizer. The league winner simulator remains a possible future extension.

### Original Architecture Decisions

The early design chose PostgreSQL over MySQL because the project depends heavily on analytical SQL:

- rolling averages
- `LAG` / `LEAD`
- `OVER`
- `PARTITION BY`
- multi-table joins
- feature views
- future JSON/JSONB storage for raw API responses

That decision held up through Tier 3. PostgreSQL became the central storage layer for FPL data, fixtures, Understat xG, historical matches, feature tables, production snapshots, prediction runs, and health logs.

The early design also recommended Git tags and branches instead of separate Tier folders. That also held up. The project used versioning rather than duplicated folders:

- `tier-1-complete`
- `tier-2-complete`
- `tier-3-dev`
- `v1.0`
- `v2.0`
- `main`

This preserved each tier without breaking imports or duplicating most of the codebase.

### Leakage Discovery: The 100% Accuracy Warning

One of the earliest and most important questions was why an initial model could show near-perfect accuracy. The answer was data leakage, not model quality.

H2H features were being calculated from the same season as the matches being predicted. With only one season of data, that can accidentally give the model information it would not know before kickoff. The fix was to remove H2H from Tier 1 and Tier 2 training and replace random splitting with time-based validation.

That decision became one of the project's core engineering rules:

> If a feature creates unrealistic accuracy, treat it as a bug until proven otherwise.

### The Arsenal Problem and Style Bias

One early domain question was whether a model based only on goals and xG could underrate teams whose value is defensive control rather than shot volume. The example used in the first master reference was Arsenal: a team can look less explosive by raw shot/xG volume while still controlling matches through pressing, structure, and transition prevention.

The proposed technical answer was tactical style modeling using metrics such as PPDA, deep completions, possession patterns, set-piece ratios, and clustering.

What happened:

- Tier 2 built KMeans style clustering and tested it.
- Tier 3 retested narrower prior-only style proxies using historical Understat xG/xGA.
- Both attempts failed validation.

Decision:

- The domain insight was valuable.
- The implemented style features did not improve the model.
- Style remains rejected / experimental, not part of the final production feature set.

This is a good example of the project philosophy: domain knowledge can propose features, but validation decides whether they stay.

### Morale, Sentiment, and Internal Conflict

Another early question was whether team morale, player confidence, dressing-room conflict, or news sentiment could affect match outcomes. The original design proposed:

- `team_morale_score`
- `internal_conflict_flag`
- `player_confidence_score`
- NewsAPI headlines
- HuggingFace sentiment models
- manual app overrides as a short-term option

Final status:

- Not implemented in Tier 3.
- No sentiment feature is part of the production model.
- No morale score is used by the current pipeline.

Reason:

Sentiment is noisy and easy to overfit. It also requires reliable timestamped news data. A sentiment feature must be known before the match and must improve chronological validation before it can be accepted.

### Psychological Blocks and Tournament Context

The early master reference also discussed psychological matchup patterns: for example, clubs repeatedly underperforming against certain opponents in high-stakes competitions, or teams performing differently in Europe than in domestic leagues.

Proposed future features included:

- psychological block score
- tournament elevation score
- competition context ratio
- UCL-specific model

Final status:

- Not implemented in Tier 3.
- H2H was retested in the Premier League only, using prior-only logic, and rejected because it worsened log loss.
- UCL prediction was deliberately left out of the project.

Reason:

Cup and UCL modeling needs a different dataset and different validation plan. Tournament data is sparse, format changes matter, and one-off events have high variance. This remains future research, not a current claim.

### Manager Regimes

Another domain question was how to handle teams that change managers. A three-year team history can mix multiple tactical regimes, making old data less useful.

The original plan suggested:

- manager tenure
- regime-change flag
- exponential decay weighting after a manager change
- lower weights for matches under old managers

Final status:

- Manager features were discussed and table infrastructure was explored.
- Real manager data was not completed.
- No manager feature is used in the final Tier 3 model.

Reason:

The project refused to invent manager dates or names. Manager features require reliable dated source data before they can be used safely.

### Player Availability and Key Player Dependency

The user raised another football-specific point: a team is not equally strong when a key player is unavailable. The early planning connected this to FPL availability fields and player-level history.

Implemented pieces:

- Tier 2 added player availability refresh.
- FPL fields such as status, price, ownership, and chance of playing were handled as current pre-deadline signals.
- The FPL optimizer avoids unavailable or low-minutes players.

Not implemented:

- no full key-player dependency model
- no `home_key_player_out` / `away_key_player_out` feature in the final Tier 3 match model
- no NLP injury extractor

Reason:

Team availability can be valuable, but it needs reliable pre-match injury and lineup data. It remains one of the most promising future additions.

### FPL Chip Strategy

The original master reference included more advanced chip logic:

- Wildcard
- Bench Boost
- Triple Captain
- Free Hit
- transfer-hit strategy
- season-long chip timing

Implemented pieces:

- Tier 1 built a valid 15-player FPL optimizer.
- Tier 2 replaced rule-based player points with XGBoost predicted points.
- Bench Boost objective logic was improved conceptually by treating all 15 players as important.
- Captain recommendation was included.

Future pieces:

- season-long reinforcement learning for chip timing
- transfer planning across multiple gameweeks
- Free Hit / Wildcard state management
- Triple Captain upside modeling

The final report should describe those as future FPL strategy work, not as completed production features.

### Hardware and Compute Strategy

The original master reference included hardware planning for an i5-12500H, 16GB DDR5, and RTX 3050 4GB. The final project confirmed the core idea: for the tabular datasets used in Tiers 1-3, CPU-based scikit-learn and XGBoost are appropriate. The GPU is not necessary for the current model.

GPU compute becomes relevant only for future neural or NLP work:

- LSTM sequence models
- DistilBERT fine-tuning
- larger text/sentiment models
- graph neural networks

The final Tier 3 production model does not require GPU acceleration.

### Future Research Ideas from the Original Blueprint

The early blueprint included several ideas beyond the final Tier 3 scope:

| Future idea | Status |
| --- | --- |
| UCL predictor | future only |
| Reinforcement learning FPL season manager | future only |
| Graph neural network for player/team relationships | future only |
| Football-BERT over match event sequences | future only |
| F1 prediction system using FastF1 | separate future project |
| News sentiment and morale layer | future only |
| manager regime weighting | future only |
| key-player dependency modeling | future only |

These ideas are valuable as roadmap items, but they are not part of the completed Tier 3 production model.

### What the Report Must Not Claim

Because the original master reference was ambitious, it is important to separate planning from implementation. The final PDF must not claim that:

- the project is a betting tool
- betting advice is provided
- Supabase is live
- an APK is built
- 2026-27 live predictions exist
- sentiment/morale features are implemented
- odds are part of the current production model
- UCL prediction is implemented
- F1 prediction is implemented
- graph neural networks or Football-BERT are implemented
- H2H is in the current final model

## 14. Future Work

The strongest future direction is not to tune the 2025-26 holdout after seeing it. That holdout is already spent. Future improvement should use new 2026-27 data and a new untouched future evaluation window.

Future work includes:

- add 2026-27 data once sources publish it
- run the weekly pipeline during the season
- improve draw modeling with a new future holdout
- add reliable injury and team availability data
- add manager regime features only with real dated manager data
- add sentiment or morale features only with reliable dated news inputs
- compare with odds as analysis, not betting advice
- build a private APK later
- use Node/Express backend + Supabase Postgres later
- trigger weekly updates through a secure backend button, not by running Python inside the APK
- consider multi-league only after the Premier League production model is stable

## 15. Conclusion

The Football Predictor Model grew from a simple local prototype into a full football machine learning research system. Tier 1 proved the basic pipeline. Tier 2 added xG and a real FPL points model. Tier 3 added multi-season data, walk-forward validation, Elo, final holdout evaluation, a production model, a weekly pipeline, and a read-only production dashboard.

The model is not perfect. The final holdout showed clear weakness on draws and a home-win bias. But the project is honest about those weaknesses. It rejected features that failed validation, avoided random splits, prevented same-gameweek leakage, kept production artifacts out of Git, and documented the full path from idea to final local production pipeline.

The final result is a useful and extensible Premier League modeling system. Its biggest strength is not only the final accuracy number, but the discipline of building, testing, rejecting, and documenting each modeling idea under time-safe rules.
