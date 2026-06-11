# Tier 3 Experiment Summary

## Status

Tier 3 is in local development. The results below include development validation results and the official final `2025-26` holdout result recorded after the Phase 10A freeze audit.

The `2025-26` season was reserved as the final untouched test season and was evaluated once after the final candidate freeze. It was not used for model selection, feature tuning, hyperparameter tuning, or threshold tuning. No deployment is complete yet.

## Data Foundation

Current Tier 3 local data tables:

- `historical_matches`: 1900 rows
- `historical_understat_xg`: 1900 rows
- `match_features_v3_base`: 1900 rows
- `elo_ratings_v3`: 1900 rows
- `match_features_v3_elo`: 1900 rows
- `match_features_v3_h2h_experiment`: 1900 rows
- `match_features_v3_style_experiment`: 1900 rows
- `standings_before_match_v3`: 3800 rows
- `match_features_v3_pressure_experiment`: 1900 rows

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

The `2025-26` season was excluded from walk-forward model selection and evaluated only after the final candidate freeze.

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

## Phase 7: Style/Tactical Source Audit and Experiment

Phase 7A inspected local tactical/style sources before feature building. The audit recommendation allowed only a narrow Understat xG/xGA style-proxy experiment from `historical_understat_xg`. It rejected direct use of PPDA, deep completions, style clusters, FPL strength/FDR, final-season averages, current/final table positions, manually labeled style, and same-match post-match stats.

Phase 7B created the local experiment table:

- `match_features_v3_style_experiment`: 1900 rows

The style builder used prior-match xG/xGA only and the leakage audit passed. The `2025-26` rows exist in the table for future holdout use, but they were not used for model selection or reported model metrics.

Phase 7C development results:

| Model | Accuracy | Log loss | Brier | Draw F1 |
| --- | ---: | ---: | ---: | ---: |
| `logistic_elo` | .5579 | .9705 | .5730 | .0893 |
| `logistic_elo_style` | .5474 | 1.0091 | .5945 | .1526 |
| `xgb_elo` | .5382 | .9984 | .5901 | .1118 |
| `xgb_elo_style` | .5237 | 1.0143 | .5991 | .1263 |

Verdict:

- `REJECT_STYLE_EXPERIMENT`

Reasons:

- Draw F1 improved, but log loss and Brier worsened too much.
- Four style features had exact `1.0000` correlation with existing rolling xG/xGA features.
- Style remains rejected / experimental archive, not a promoted core W/D/L model input.

## Phase 8: Match Pressure Experiment

Phase 8A created pre-match standings and pressure experiment tables:

- `standings_before_match_v3`: 3800 rows
- `match_features_v3_pressure_experiment`: 1900 rows

Pressure was built only from pre-match standings. No final table, final rank, future match, same-match result, same-date result, derby, or rivalry feature was used.

Pressure coverage in development rows:

- Both pressure indexes non-null: 1198
- Only one pressure index non-null: 4
- Both pressure indexes null: 318

Phase 8B development results:

| Model | Accuracy | Log loss | Brier | Draw F1 |
| --- | ---: | ---: | ---: | ---: |
| `logistic_elo` | .5579 | .9705 | .5730 | .0893 |
| `logistic_elo_pressure` | .5447 | 1.0024 | .5904 | .1643 |
| `xgb_elo` | .5382 | .9984 | .5901 | .1118 |
| `xgb_elo_pressure` | .5329 | .9918 | .5890 | .1387 |

Verdict:

- `REJECT_PRESSURE_EXPERIMENT`

Reasons:

- `logistic_elo_pressure` worsened log loss by +0.0319 and Brier by +0.0174.
- Pressure improved draw F1 but hurt probability quality.
- XGB pressure improved XGB log loss slightly, but did not beat the logistic champion and did not improve both folds.
- Pressure remains rejected / experimental archive, not a promoted core W/D/L model input.

## Phase 9: Final Hard-Label Draw Overlay

Phase 9A tested one final hard-label draw overlay on top of the current probability champion, `logistic_elo_expanding`.

Design:

- Main model remains `logistic_elo_expanding`.
- Overlay changes hard labels only.
- Probabilities are not adjusted.
- No binary draw classifier was trained.
- No stacked model was trained.
- No H2H, style, pressure, or Poisson inputs were used.
- No validation tuning was used.
- No `2025-26` rows were loaded, tuned, evaluated, or reported for model metrics.

Overlay rule:

- Start from the normal argmax prediction.
- Change the hard label to draw only if:
  - `P(D)` is above the training-derived threshold
  - Draw is the second-highest class
  - Dominant class probability is below `0.50`
- Probabilities remain untouched.

Thresholds:

- Fold 1 threshold: `0.34`
- Fold 2 threshold: `0.28`
- Difference: `0.06`

Threshold stability passed, but narrowly.

Fold 1:

| Mode | Accuracy | Log loss | Brier | Draw F1 | Changed to draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Argmax | .5921 | .9289 | .5436 | .1386 | 0 |
| Overlay | .5947 | .9289 | .5436 | .2764 | 22 |

Fold 2:

| Mode | Accuracy | Log loss | Brier | Draw F1 | Changed to draw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Argmax | .5237 | 1.0121 | .6024 | .0400 | 0 |
| Overlay | .5289 | 1.0121 | .6024 | .1500 | 20 |

Aggregate:

| Mode | Accuracy | Log loss | Brier | Draw recall | Draw precision | Draw F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Argmax | .5579 | .9705 | .5730 | .0534 | .3271 | .0893 |
| Overlay | .5618 | .9705 | .5730 | .1520 | .3740 | .2132 |

Verdict:

- `ACCEPT_DRAW_OVERLAY_EXPERIMENTAL_SERVING_HELPER`

Clarification:

- The probability champion remains `logistic_elo_expanding`.
- The overlay is accepted only for hard-label serving.
- It must not be described as improving probability calibration.
- It must not alter probabilities.
- It should be included in final holdout evaluation as a separate hard-label metric line.

## Phase 10A: Final Model Freeze Audit

Phase 10A freezes the final Tier 3 candidate definition before any final holdout evaluation.

Final candidate definition:

- Probability model: `logistic_elo_expanding`
- Source table: `match_features_v3_elo`
- Allowed feature family: base + Elo features only
- Serving helper: draw overlay hard-label rule only
- Overlay status: accepted experimental serving helper
- Overlay probability behavior: probabilities must not be altered

Rejected or excluded feature families:

- H2H: rejected / experimental archive
- Style: rejected / experimental archive
- Pressure: rejected / experimental archive
- Calibration variants: not promoted
- XGB variants: not champion
- Poisson: diagnostic only
- Betting odds, manager, sentiment, injury, rivalry, and derby: rejected for Tier 3

At the time of the Phase 10A checkpoint, the final `2025-26` holdout remained untouched. No final `2025-26` holdout evaluation had been run.

Next phase:

- Final holdout evaluation only after the Phase 10A freeze audit passes.

## Phase 10C: Official Final 2025-26 Holdout Evaluation

Phase 10C records the official final holdout result exactly as observed after the Phase 10A freeze audit. No tuning, competing final models, feature changes, hyperparameter changes, draw overlay threshold changes, or alternate model-selection experiments were run after seeing the final holdout.

Setup:

- Official final probability model: `logistic_elo_expanding`
- Training seasons: `2021-22`, `2022-23`, `2023-24`, `2024-25`
- Training rows: 1520
- Holdout season: `2025-26`
- Holdout rows: 380
- Source table: `match_features_v3_elo`
- Feature count: 32
- Selected draw threshold from training only: 0.24
- No tuning on `2025-26`

Actual holdout distribution:

| Result | Count |
| --- | ---: |
| H | 162 |
| D | 104 |
| A | 114 |

Argmax final holdout metrics:

| Metric | Value |
| --- | ---: |
| accuracy | 0.4868 |
| log_loss | 1.0601 |
| Brier | 0.6372 |
| draw recall | 0.0000 |
| draw precision | 0.0000 |
| draw F1 | 0.0000 |

Argmax predicted distribution:

| Prediction | Count |
| --- | ---: |
| H | 250 |
| D | 1 |
| A | 129 |

Draw overlay final holdout metrics:

| Metric | Value |
| --- | ---: |
| accuracy | 0.4684 |
| log_loss | 1.0601 |
| Brier | 0.6372 |
| draw recall | 0.0769 |
| draw precision | 0.2353 |
| draw F1 | 0.1159 |

Draw overlay predicted distribution:

| Prediction | Count |
| --- | ---: |
| H | 229 |
| D | 34 |
| A | 117 |

Draw overlay checks:

- Labels changed to draw: 33
- Probability metrics unchanged assertion: passed

Interpretation:

- Final holdout performance is weaker than development validation.
- The largest weakness is draw prediction.
- The draw overlay improved draw F1 but reduced final holdout accuracy.
- The draw overlay must not be presented as improving the final probability model.
- The draw overlay may be documented only as an optional draw-risk helper, not as the default final prediction rule.
- Do not tune after the final holdout.

## Current Tier 3 Decision List

- Elo: core candidate
- `logistic_elo_expanding`: official final probability model
- Draw overlay: accepted experimental serving helper
- H2H: rejected / experimental archive
- Style: rejected / experimental archive
- Pressure: rejected / experimental archive
- Calibration: not promoted
- XGB calibration/draw rule: not promoted
- Poisson: diagnostic only
- Rivalry/derby: rejected for Tier 3

## Pattern Noticed

H2H, style, and pressure all improved draw behavior in some way. All hurt or failed to improve probability quality enough.

Therefore the main probability model should remain `logistic_elo_expanding`.

## Current Model Decision

Current development probability champion:

- `logistic_elo_expanding`

Development champion metrics:

- accuracy: 0.5579
- log_loss: 0.9705
- Brier: 0.5730
- draw F1: 0.0893

Current status:

- Elo remains a core candidate feature group.
- The official final probability model is `logistic_elo_expanding`.
- The draw overlay is documented as an optional draw-risk helper only.
- The probability model remains `logistic_elo_expanding`; probabilities are unchanged by the overlay.
- H2H, style, and pressure are rejected / experimental archive.
- Poisson remains diagnostic only.
- Calibration is not promoted.
- The final model candidate was frozen before final holdout evaluation.
- The final `2025-26` holdout evaluation has been recorded.
- No tuning should occur after the final holdout result.

## Active Rejection List

Not promoted:

- H2H
- Style
- Pressure
- Calibration
- XGBoost draw rule
- Poisson for W/D/L classification
- Rivalry/derby

Poisson remains useful as a scoreline/probability diagnostic pathway, not as the promoted W/D/L model.

## Feature Decisions So Far

Accepted or candidate:

- Multi-season data foundation
- Rolling form/xG baseline features
- Elo as a candidate feature group

Experimental only:

- Draw overlay as a hard-label serving helper
- Poisson scoreline diagnostics

Not accepted as core W/D/L model inputs:

- H2H
- Style
- Pressure
- Poisson outputs
- Betting odds
- Rivalry/derby

Deferred:

- Manager features
- Sentiment/morale
- Injuries/team availability
- FPL multi-gameweek planning
- Deployment/Supabase/private backend

## Leakage Rules Still Active

- No random split
- `2025-26` excluded from model selection and evaluated only after freeze
- No post-match Elo columns
- No final holdout metrics before model freeze
- No tuning after final holdout evaluation
- No current FPL strength/FDR in historical model
- All imputation/scaling inside training folds only
- No feature accepted without walk-forward validation

## Next Recommended Phases

1. Keep `logistic_elo_expanding` as the official final probability model.
2. Document the draw overlay only as an optional draw-risk helper, without changing probabilities.
3. Preserve H2H, style, and pressure as rejected / experimental archive work so the same failed experiments are not repeated.
4. Manager, sentiment, and injury features remain experimental unless reliable dated sources exist.
5. Deployment comes only after the final holdout result is documented and accepted without post-holdout tuning.
