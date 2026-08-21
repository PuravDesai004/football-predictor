# Tier 3 Freeze Report

**Freeze date:** 2026-08-21  
**Frozen Git commit:** `ba59083`  
**Branch:** `main`  
**Repository:** `PuravDesai004/football-predictor`

## Freeze decision

Tier 3 is frozen as the current research and production baseline. No further Tier 3 feature engineering, model tuning, retraining, threshold changes, schema changes, or production-artifact replacement should occur unless the owner explicitly reopens Tier 3.

The next development effort should be a separate standalone FPL squad-optimization project. Tier 3 remains available as a reference system and historical benchmark.

## Frozen match model

- Model: `production_logistic_elo_v3`
- Artifact: `models/saved/production_logistic_elo_v3.pkl`
- Feature contract: `models/saved/production_features_v3.json`
- Features: 32 base + Elo features
- Training seasons: 2021-22 through 2025-26 for production refit
- Official final holdout: 2025-26, evaluated before the production refit

### Official final holdout metrics

| Metric | Argmax | Draw overlay |
| --- | ---: | ---: |
| Accuracy | 0.4868 | 0.4684 |
| Log loss | 1.0601 | 1.0601 |
| Brier score | 0.6372 | 0.6372 |
| Draw recall | 0.0000 | 0.0769 |
| Draw precision | 0.0000 | 0.2353 |
| Draw F1 | 0.0000 | 0.1159 |

The 2025-26 holdout is spent and must not be reused for tuning or model selection.

## Frozen FPL v3 candidate

- Artifact: `data/production_artifacts/fpl_points_v3_candidate.pkl`
- Feature contract: `data/production_artifacts/fpl_points_v3_candidate_features.json`
- Metadata: `data/production_artifacts/fpl_points_v3_candidate_metadata.json`
- Status: candidate-only, not a promoted production model
- Validation season: 2024-25
- Final holdout season: 2025-26
- Validation MAE: 1.0873
- Validation Spearman: 0.6992
- Validation NDCG@10: 0.4614
- Validation top-10 points captured: 40.3%

## Database verification at freeze

The local PostgreSQL database was not modified during the freeze verification. Key rows were present:

- `production_match_predictions`: 380
- `production_upcoming_match_features_v3`: 380
- `fpl_player_predictions_v3`: 587
- `fpl_optimizer_outputs_v3`: 1

## Verification

- Project-local Python: 3.14.7
- scikit-learn: 1.8.0
- XGBoost: 3.4.1
- Regression tests: 3 passed
- Key Tier 3 modules: compiled successfully
- Git working tree: clean at freeze commit
- No live pipeline was run
- No model was retrained
- No database writes were performed

## Artifact SHA-256 hashes

| Artifact | SHA-256 |
| --- | --- |
| `models/saved/production_logistic_elo_v3.pkl` | `2AC34BB8964B981E96E1D04368AF653901DF819EB475663EDC96F7155877E2E5` |
| `models/saved/production_features_v3.json` | `00341E167030EAE3A326D997242D19953730CA64337D9B20B9A8B5EF983C6C67` |
| `models/saved/production_metadata_v3.json` | `ADF18C9AE849F1D2523AF726C767B8032EEC8D284F687D1FBB8E1AA1ED89C855` |
| `models/saved/production_draw_threshold_v3.json` | `9821528AAD08B981E5D534D240B993D25D2D746EA8D4685EB90EE29C455D6289` |
| `data/production_artifacts/fpl_points_v3_candidate.pkl` | `E9B450FE814A235B8FE6B4AB64CF9D1351B48625626725614A1D84BC7C5AA51F` |
| `data/production_artifacts/fpl_points_v3_candidate_features.json` | `E879432F2217E63313DB72C200D066315E00590E5FBC000B25D24E2CFB9183FC` |
| `data/production_artifacts/fpl_points_v3_candidate_metadata.json` | `115755D4D8C27F844327AF89AA2F96014B65F88EB7A8E030FB0F65F48F13E6C5` |

## Reopening criteria

Tier 3 should only be reopened if the owner explicitly requests one of the following:

1. A controlled development experiment.
2. A new future holdout evaluation.
3. A production bug fix affecting Tier 3 behavior.
4. A documented model-artifact migration.

Accuracy improvement work should happen in the new standalone FPL project first. Its primary objectives should be legal squad quality, starting-XI points, captain and vice-captain points, bench value, and realized gameweek points.
