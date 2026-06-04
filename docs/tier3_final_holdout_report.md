# Tier 3 Final Holdout Report

## Status

This report records the official `2025-26` final holdout evaluation for Tier 3 after the Phase 10A final candidate freeze audit.

No model-selection experiments, feature tuning, hyperparameter tuning, draw overlay threshold tuning, competing final models, or interpretation-based metric changes were run after seeing the final holdout.

## Frozen Setup

- Official final probability model: `logistic_elo_expanding`
- Training seasons: `2021-22`, `2022-23`, `2023-24`, `2024-25`
- Training rows: 1520
- Holdout season: `2025-26`
- Holdout rows: 380
- Source table: `match_features_v3_elo`
- Feature count: 32
- Selected draw threshold from training only: 0.24
- No tuning on `2025-26`

## Actual Holdout Distribution

| Result | Count |
| --- | ---: |
| H | 162 |
| D | 104 |
| A | 114 |

## Argmax Final Holdout Metrics

| Metric | Value |
| --- | ---: |
| accuracy | 0.4868 |
| log_loss | 1.0601 |
| Brier | 0.6372 |
| draw recall | 0.0000 |
| draw precision | 0.0000 |
| draw F1 | 0.0000 |

Predicted distribution:

| Prediction | Count |
| --- | ---: |
| H | 250 |
| D | 1 |
| A | 129 |

## Draw Overlay Final Holdout Metrics

| Metric | Value |
| --- | ---: |
| accuracy | 0.4684 |
| log_loss | 1.0601 |
| Brier | 0.6372 |
| draw recall | 0.0769 |
| draw precision | 0.2353 |
| draw F1 | 0.1159 |

Predicted distribution:

| Prediction | Count |
| --- | ---: |
| H | 229 |
| D | 34 |
| A | 117 |

Draw overlay checks:

- Labels changed to draw: 33
- Probability metrics unchanged assertion: passed

## Interpretation

The official final probability model is `logistic_elo_expanding`.

Final holdout performance is weaker than development validation. The largest weakness is draw prediction. The draw overlay improved draw F1 but reduced final holdout accuracy, while log loss and Brier remained unchanged because the overlay does not alter probabilities.

Therefore, the draw overlay must not be presented as improving the final probability model. It may be documented only as an optional draw-risk helper, not as the default final prediction rule.

No tuning should occur after the final holdout.

## Guardrails

- Do not tune features after this result.
- Do not tune hyperparameters after this result.
- Do not change draw overlay threshold logic after this result.
- Do not run competing final models after this result.
- Do not add H2H, style, pressure, Poisson, odds, manager, sentiment, injury, rivalry, or derby features in response to this result.
- Do not overwrite these final holdout metrics based on interpretation.
