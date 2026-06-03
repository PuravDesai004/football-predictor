# Tier 3 Experiment Summary

## Status

Tier 3 is in local development. The results below are development validation results, not final test results.

The `2025-26` season is reserved as the final untouched test season. It has not been evaluated for model selection or reporting. No deployment is complete yet.

## Data Foundation

Current Tier 3 local data tables:

- `historical_matches`: 1900 rows
- `historical_understat_xg`: 1900 rows
- `match_features_v3_base`: 1900 rows
- `elo_ratings_v3`: 1900 rows
- `match_features_v3_elo`: 1900 rows

Seasons present:

- `2021-22`
- `2022-23`
- `2023-24`
- `2024-25`
- `2025-26`

Each season has 380 matches and 20 teams. Understat join coverage is complete at 1900 joined rows. Tier 2 tables were untouched while building and validating the Tier 3 local foundation.

## Validation Strategy

Tier 3 does not use a random train/test split. Validation uses chronological walk-forward folds only.

Development seasons:

- `2021-22`
- `2022-23`
- `2023-24`
- `2024-25`

Final holdout:

- `2025-26`

Walk-forward folds:

- Fold 1:
  - Train: `2021-22`, `2022-23`
  - Validate: `2023-24`
- Fold 2:
  - Train: `2021-22`, `2022-23`, `2023-24`
  - Validate: `2024-25`

The `2025-26` season has not been evaluated.

## Phase Results

## Phase 2B: Baseline Models

`majority_baseline`:

- accuracy: 0.4342
- log_loss: 1.0682
- brier: 0.6469
- draw_recall: 0.0000

`logistic_baseline`:

- accuracy: 0.5474
- log_loss: 0.9890
- brier: 0.5875
- draw_recall: 0.0793

`xgb_imputed_baseline`:

- accuracy: 0.5237
- log_loss: 1.0176
- brier: 0.6052
- draw_recall: 0.0588

`xgb_native_nan_baseline`:

- accuracy: 0.5158
- log_loss: 1.0186
- brier: 0.6057
- draw_recall: 0.0635

Decision:

- Keep imputed XGBoost as the safer XGBoost baseline.
- Logistic baseline was the strongest model before Elo.

## Phase 3C: Elo Feature Test

`logistic_base`:

- accuracy: 0.5474
- log_loss: 0.9890
- brier: 0.5875
- draw_recall: 0.0793

`logistic_elo`:

- accuracy: 0.5579
- log_loss: 0.9705
- brier: 0.5730
- draw_recall: 0.0534

`xgb_base`:

- accuracy: 0.5237
- log_loss: 1.0176
- brier: 0.6052
- draw_recall: 0.0588

`xgb_elo`:

- accuracy: 0.5382
- log_loss: 0.9984
- brier: 0.5901
- draw_recall: 0.0764

Decision:

- Elo clearly improved XGBoost across aggregate metrics.
- Elo improved Logistic probability metrics and accuracy but reduced draw recall.
- Keep Elo as a Tier 3 candidate feature group.
- Do not declare a final model yet.

## Phase 4A/4A.1: Poisson Scoreline Pathway

Aggregate development metrics:

- accuracy: 0.5500
- log_loss: 0.9775
- brier: 0.5802
- draw_recall: 0.0000
- exact_score_accuracy: 0.1145
- home_goals_mae: 0.9868
- away_goals_mae: 0.8882
- total_goals_mae: 1.4145

Draw diagnostics:

- mean draw probability: 0.2181
- median draw probability: 0.2205
- max draw probability: 0.2913
- draw-highest matches: 0
- matches draw probability >= 0.25: 35
- matches draw probability >= 0.30: 0
- actual draw rate: 0.2303

Decision:

- Poisson should not replace the W/D/L classifier yet.
- Keep Poisson as a scoreline/probability diagnostic layer.
- Do not feed Poisson outputs into XGBoost yet.

## Current Best Development Models

The best overall development classifier so far is `logistic_elo` by accuracy, log loss, and Brier score.

The best XGBoost candidate is `xgb_elo`.

Poisson is useful for scoreline diagnostics, not as the final W/D/L classifier. No final test has been run.

## Feature Decisions So Far

Accepted or candidate:

- Multi-season data foundation
- Rolling form/xG baseline features
- Elo as a candidate feature group

Not accepted as the main W/D/L model:

- Poisson hard-class W/D/L

Deferred:

- H2H retest
- Style/tactical retest
- Manager features
- Sentiment/morale
- Injuries/team availability
- Betting odds
- FPL multi-gameweek planning
- Deployment/Supabase/private backend

## Leakage Rules Still Active

- No random split
- `2025-26` untouched
- No post-match Elo columns
- No H2H until safe retest
- No current FPL strength/FDR in historical model
- All imputation/scaling inside training folds only
- No feature accepted without walk-forward validation

## Next Recommended Steps

1. Review/commit current stable Tier 3 local foundation.
2. Phase 5A: H2H retest only using prior meetings across multi-season data.
3. Or Phase 5A alternative: model calibration / draw handling before H2H.
4. Final `2025-26` test only after model choice is frozen.
