# Football Predictor Model

A local machine learning project for predicting Premier League match outcomes and optimizing Fantasy Premier League squads. Not deployed anywhere. Not a betting tool. Just a personal research project that started small and grew into a three-tier pipeline with a production model trained on five seasons of data.

The project went through three stages:

- **Tier 1:** data foundation, PostgreSQL tables, baseline match prediction, and a rule-based FPL optimizer.
- **Tier 2:** full local system with Understat xG data, FPL XGBoost points modeling, and the original Streamlit app.
- **Tier 3:** multi-season research, walk-forward validation, final holdout evaluation, and a production-ready local pipeline for 2026-27.

## Current Status

Tier 3 research is complete. The production model is trained locally on 2021-22 through 2025-26. A read-only Streamlit dashboard runs locally.

The 2026-27 pipeline is built and idle. It is waiting for real upcoming fixtures and results. Running it now and getting no predictions back is expected behavior. The pipeline does not fabricate fixtures, predictions, or scores when source data is not available.

## Production Model

The current production model is `production_logistic_elo_v3`: logistic regression with Elo features added on top of the Tier 3 base feature set. The research champion was `logistic_elo_expanding`.

- **Trained on:** 2021-22 through 2025-26
- **Features:** 32
- **Production draw threshold:** 0.30

Production artifacts are local-only and gitignored:

```text
models/saved/production_logistic_elo_v3.pkl
models/saved/production_features_v3.json
models/saved/production_draw_threshold_v3.json
models/saved/production_metadata_v3.json
```

## Final Holdout Results

The holdout season was 2025-26. The model candidate was frozen before any 2025-26 data was used, then evaluated once.

**Argmax predictions:**

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4868 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw F1 | 0.0000 |

**Draw overlay, final holdout threshold 0.24:**

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4684 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw F1 | 0.1159 |

Draw F1 of 0.0000 under argmax is the honest number. Draws are systematically underpredicted, and the model leans toward home wins more than it should. The draw overlay surfaces some draw risk, but it does not improve probability quality. It trades home/away accuracy for marginal draw recall, so it is documented as a draw-risk helper rather than an upgrade to the default model.

48% accuracy on three-class Premier League prediction without odds data is not shocking, but it leaves clear room for improvement. For this kind of model, log loss, Brier score, and calibration matter as much as hard-label accuracy.

## Production Dashboard

```bash
streamlit run app/production_dashboard.py
```

Tabs:

- Overview
- Pipeline Status
- Predictions
- Model Health
- Reports
- How To Run

The dashboard is read-only. It checks artifact availability, summarizes production table state, and renders Tier 3 reports. If PostgreSQL is unreachable, it handles that cleanly instead of crashing.

The original Tier 2 app still runs:

```bash
streamlit run app/streamlit_app.py
```

## Weekly Production Pipeline

Full pipeline:

```bash
python src/production/run_weekly_pipeline.py --target-season 2026-27
```

Individual steps, if you want to run them separately:

```bash
python src/production/weekly_ingest.py --target-season 2026-27
python src/production/build_upcoming_features.py --target-season 2026-27
python src/production/predict_production_matches.py --target-season 2026-27
python src/production/score_predictions.py --target-season 2026-27
```

The pipeline skips safely when source data is not available. No fake fixtures, no fake predictions, no fake scores.

## Setup

Requires a local PostgreSQL database named `football_db` and a `.env` file.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` and do not commit it:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=football_db
DB_USER=your_username
DB_PASS=your_password
```

The code also reads `DATABASE_URL` if you are pointing at a hosted PostgreSQL instance.

## Validation

```bash
python -m py_compile app/production_dashboard.py
python -m py_compile src/production/run_weekly_pipeline.py
python src/tier3_validation.py
streamlit run app/production_dashboard.py
python src/production/run_weekly_pipeline.py --target-season 2026-27
```

If PostgreSQL is unreachable, `tier3_validation.py` fails cleanly within the configured timeout instead of hanging indefinitely.

## Directory

```text
Football Predictor Model/
|-- app/
|   |-- streamlit_app.py              # Tier 2 local app
|   `-- production_dashboard.py       # Production status dashboard
|-- data/
|   `-- historical/                   # Local football-data CSVs, gitignored
|-- docs/
|   |-- tier3_experiment_summary.md
|   |-- tier3_final_holdout_report.md
|   |-- tier3_final_error_analysis.md
|   `-- tier3_style_source_audit.md
|-- models/
|   `-- saved/                        # Production artifacts gitignored; some Tier 2 artifacts tracked
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

## Data Sources

- Fantasy Premier League API
- football-data.co.uk CSVs
- Understat historical xG data

All features are pre-match only. No same-gameweek FPL leakage. No random splits on time-ordered data. The holdout was evaluated once after the model was frozen.

The production model skips H2H, style, pressure, Poisson, odds, manager, sentiment, injury, and rivalry features. Those ideas are not automatically bad, but this version prioritizes a feature set that is auditable, stable, and leakage-free.

## GitHub Safety Check

Before pushing:

```bash
git status --short
git check-ignore -v data/historical/E0_2021-22.csv
git check-ignore -v models/saved/production_logistic_elo_v3.pkl
git check-ignore -v models/saved/production_features_v3.json
git check-ignore -v models/saved/production_draw_threshold_v3.json
git check-ignore -v models/saved/production_metadata_v3.json
```

Raw data, database dumps, secrets, and production model artifacts should not be in this repository.

---

*This project is not betting advice.*

---

Purav Desai
