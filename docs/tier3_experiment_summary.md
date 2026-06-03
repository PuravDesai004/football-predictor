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
- `match_features_v3_h2h_experiment`: 1900 rows

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

## Phase 5A: H2H Feature Engineering

Phase 5A added `match_features_v3_h2h_experiment` as a local experiment table with 1900 rows.

The H2H feature builder uses leakage-safe prior-only logic. For each fixture, only previous meetings before that match date are eligible. The table includes feature rows for `2025-26`, but those rows were not used for development model selection.

Key table result:

- `match_features_v3_h2h_experiment`: 1900 rows
- Development seasons available for experiments: `2021-22` through `2024-25`
- Reserved final holdout still untouched for model metrics: `2025-26`

## Phase 5B: H2H Model Experiment

The H2H model experiment used 1520 development rows only.

H2H evidence threshold:

- Rows thresholded because `h2h_matches_prior < 3`: 807
- Rows with usable H2H after thresholding: 713

Result:

- H2H improved some draw behavior.
- H2H worsened log loss materially.
- Verdict: reject / keep experimental.

H2H is not promoted into the core Tier 3 feature set.

## Phase 6A: Calibration and Draw-Rule Experiment

Phase 6A tested time-safe fit/calibrate/validate folds:

- Fold 1:
  - Fit: `2021-22`
  - Calibrate: `2022-23`
  - Validate: `2023-24`
- Fold 2:
  - Fit: `2021-22`, `2022-23`
  - Calibrate: `2023-24`
  - Validate: `2024-25`

The initial calibration setup looked promising within its smaller fit/calibrate/validate design, especially on probability metrics. The logistic draw rule became an experimental hard-label candidate. The XGBoost draw rule was rejected.

No `2025-26` rows were loaded, tuned, calibrated, or evaluated for model metrics.

## Phase 6B: Development Model-Selection Audit

Phase 6B compared the original expanding-window Elo models against the time-safe calibrated models in one repeatable leaderboard.

Final development leaderboard:

| Model | Accuracy | Log loss | Brier | Draw F1 |
| --- | ---: | ---: | ---: | ---: |
| `logistic_elo_expanding` | .5579 | .9705 | .5730 | .0893 |
| `logistic_elo_calibrated_draw_rule` | .5395 | .9732 | .5773 | .1910 |
| `logistic_elo_calibrated` | .5382 | .9732 | .5773 | .0424 |
| `logistic_base_expanding` | .5474 | .9890 | .5875 | .1193 |
| `xgb_elo_expanding` | .5382 | .9984 | .5901 | .1118 |
| `xgb_elo_calibrated` | .5368 | 1.0035 | .5966 | .0503 |
| `xgb_elo_calibrated_draw_rule` | .5132 | 1.0035 | .5966 | .0845 |
| `xgb_base_expanding` | .5237 | 1.0176 | .6052 | .0895 |

Model-selection audit result:

- Current development probability champion: `logistic_elo_expanding`
- Calibration is not promoted because it did not beat expanding `logistic_elo` on the promotion gate.
- Logistic draw rule remains an experimental hard-label helper only.
- XGBoost draw rule is not promoted.

## Current Model Decision

Current development probability champion:

- `logistic_elo_expanding`

Current status:

- Elo remains a core candidate feature group.
- The draw rule remains an experimental hard-label helper only.
- No final model has been frozen.
- No final `2025-26` holdout evaluation has been run.
- `2025-26` remains the untouched final holdout.

## Active Rejection List

Not promoted:

- H2H
- Calibration
- XGBoost draw rule
- Poisson for W/D/L classification

Poisson remains useful as a scoreline/probability diagnostic pathway, not as the promoted W/D/L model.

## Feature Decisions So Far

Accepted or candidate:

- Multi-season data foundation
- Rolling form/xG baseline features
- Elo as a candidate feature group

Experimental only:

- Logistic draw rule as a hard-label helper
- Poisson scoreline diagnostics

Not accepted as core W/D/L model inputs:

- H2H
- Poisson outputs
- Betting odds

Deferred:

- Style/tactical features
- Manager features
- Sentiment/morale
- Injuries/team availability
- FPL multi-gameweek planning
- Deployment/Supabase/private backend

## Leakage Rules Still Active

- No random split
- `2025-26` untouched
- No post-match Elo columns
- No final holdout metrics before model freeze
- No current FPL strength/FDR in historical model
- All imputation/scaling inside training folds only
- No feature accepted without walk-forward validation

## Next Recommended Phases

1. Phase 7A: source audit for style/tactical features before building anything.
2. Phase 7B only if source quality is valid: build a time-safe style/tactical experiment.
3. Manager, sentiment, and injury features remain experimental unless reliable dated sources exist.
4. Deployment comes only after final model freeze.
