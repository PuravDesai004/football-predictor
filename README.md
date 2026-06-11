# Football Predictor Model

A local machine learning project for Premier League match prediction and Fantasy Premier League analysis.

The project grew through three stages:

- Tier 1: data foundation, PostgreSQL tables, baseline match prediction, and a rule-based FPL optimizer.
- Tier 2: full local system with Understat xG, FPL XGBoost points modeling, and the original Streamlit app.
- Tier 3: multi-season Premier League research, walk-forward validation, final holdout evaluation, and a production-transition pipeline for 2026-27.

This repository is intended for code review and local reproduction. Raw data, secrets, and production model artifacts are intentionally excluded.

## Current Status

- Tier 3 research is complete.
- The production model has been trained locally on 2021-22 through 2025-26.
- A read-only production Streamlit dashboard is available.
- The 2026-27 production pipeline exists and waits for real upcoming fixtures/results.
- Current 2026-27 state may legitimately be empty if football-data CSVs, Understat 2026 data, or unfinished FPL fixtures are unavailable.

## Main Production Model

The current production model is:

```text
production_logistic_elo_v3
```

It is a logistic regression model using the Tier 3 base feature set plus Elo features. The production artifact was trained locally on:

```text
2021-22, 2022-23, 2023-24, 2024-25, 2025-26
```

The production feature artifact contains 32 features. The production draw threshold artifact is `0.30`.

Production artifacts are local-only and ignored by Git:

```text
models/saved/production_logistic_elo_v3.pkl
models/saved/production_features_v3.json
models/saved/production_draw_threshold_v3.json
models/saved/production_metadata_v3.json
```

## Final Tier 3 Holdout Result

The official final holdout season was 2025-26. It was evaluated once after the model candidate was frozen.

Argmax final holdout metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4868 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw F1 | 0.0000 |

Draw overlay final holdout metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4684 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw F1 | 0.1159 |

Main weakness:

- Draw underprediction.
- Home prediction bias.
- Some high-confidence wrong predictions.

The draw overlay is documented as a draw-risk helper only. It is not the default final probability model and should not be presented as improving probability quality.

## Production Dashboard

Run the production dashboard locally:

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

The dashboard is read-only by default. It checks artifact availability, summarizes production table state, renders Tier 3 reports, and handles PostgreSQL unavailability with a clean UI state.

The original Tier 2 app remains available:

```bash
streamlit run app/streamlit_app.py
```

## Weekly Production Pipeline

Run the full local production workflow:

```bash
python src/production/run_weekly_pipeline.py --target-season 2026-27
```

Individual production steps:

```bash
python src/production/weekly_ingest.py --target-season 2026-27
python src/production/build_upcoming_features.py --target-season 2026-27
python src/production/predict_production_matches.py --target-season 2026-27
python src/production/score_predictions.py --target-season 2026-27
```

The pipeline is designed to skip safely when real source data is unavailable. It does not create fake fixtures, fake predictions, or fake scores.

## Important Caveats

- Requires a local PostgreSQL database named `football_db`.
- Requires local environment variables in `.env`; secrets are not included in this repository.
- Raw football-data CSVs are ignored by Git.
- Production model artifacts are ignored by Git.
- Some Tier 2 model artifacts are intentionally tracked because the original local Streamlit app depends on them.
- This project is not betting advice.
- No odds, sentiment/NLP, multi-league production support, Supabase deployment, or Android APK work is included in the current production transition.

## Directory Overview

```text
Football Predictor Model/
|-- app/
|   |-- streamlit_app.py              # Tier 2 local app
|   `-- production_dashboard.py       # Production status dashboard
|-- data/
|   `-- historical/                   # Local football-data CSVs ignored except .gitkeep
|-- docs/
|   |-- tier3_experiment_summary.md
|   |-- tier3_final_holdout_report.md
|   |-- tier3_final_error_analysis.md
|   `-- tier3_style_source_audit.md
|-- models/
|   `-- saved/                        # Production artifacts ignored; selected Tier 2 artifacts tracked
|-- sql/
|   |-- schema.sql
|   |-- feature_queries.sql
|   |-- fpl_feature_queries.sql
|   `-- tier3_schema.sql
|-- src/
|   |-- production/
|   |   |-- train_production_model.py
|   |   |-- weekly_ingest.py
|   |   |-- build_upcoming_features.py
|   |   |-- predict_production_matches.py
|   |   |-- score_predictions.py
|   |   `-- run_weekly_pipeline.py
|   |-- tier3_validation.py
|   |-- tier3_freeze_audit.py
|   |-- tier3_final_holdout_eval.py
|   |-- tier3_final_error_analysis.py
|   `-- ... Tier 1/Tier 2/Tier 3 pipeline and training scripts
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Setup Notes

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a local `.env` file. Do not commit it.

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=football_db
DB_USER=your_username
DB_PASS=your_password
```

For hosted PostgreSQL, the code also supports `DATABASE_URL`. Do not commit real credentials.

## Validation Commands

Useful local checks:

```bash
python -m py_compile app/production_dashboard.py
python -m py_compile src/production/run_weekly_pipeline.py
python src/tier3_validation.py
streamlit run app/production_dashboard.py
python src/production/run_weekly_pipeline.py --target-season 2026-27
```

If PostgreSQL is unreachable, `src/tier3_validation.py` should fail clearly within the configured timeout instead of hanging.

## GitHub Safety

The repository is prepared so sensitive and heavy local files stay out of GitHub:

- `.env` is ignored.
- `.streamlit/secrets.toml` is ignored.
- `data/historical/*.csv` is ignored.
- Production model artifacts under `models/saved/production_*` are ignored.
- Python caches, virtual environments, and local generated data folders are ignored.

Before uploading, check:

```bash
git status --short
git check-ignore -v data/historical/E0_2021-22.csv
git check-ignore -v models/saved/production_logistic_elo_v3.pkl
git check-ignore -v models/saved/production_features_v3.json
git check-ignore -v models/saved/production_draw_threshold_v3.json
git check-ignore -v models/saved/production_metadata_v3.json
```

Do not upload raw data, database dumps, secrets, or local-only production model artifacts.

## Data Sources

- Fantasy Premier League API
- football-data.co.uk local CSVs
- Understat historical xG data

All research and production steps are designed around time-safe validation and pre-match feature availability.

## Design Principles

- No random train/test splits on time-ordered match data.
- No final-holdout tuning after seeing 2025-26.
- No fake production fixtures, predictions, or scores.
- No same-gameweek leakage in FPL features.
- No H2H, style, pressure, Poisson, odds, manager, sentiment, injury, or rivalry features in the final production model.
- Probability quality matters; hard-label accuracy alone is not enough.

---

Purav Desai
