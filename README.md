# Football Predictor Model

Football Predictor Model is a local machine learning project for Premier League match prediction and Fantasy Premier League analysis. It is a research and portfolio project, not betting advice.

## Current Status

The project is Tier 3 production-ready locally:

- Tier 1 foundation is complete.
- Tier 2 full local system is complete.
- Tier 3 multi-season research is complete.
- Production Premier League model artifacts were trained locally and are intentionally ignored by Git.
- Production Streamlit dashboard is available locally.
- The 2026-27 production pipeline exists, but live 2026-27 predictions are not claimed until real upcoming fixtures and completed results are available.

No Supabase deployment is live. No Android APK is built. Odds, sentiment, morale, and injury features are not part of the current model.

## Tier 1 Summary

Tier 1 built the first local PostgreSQL and Streamlit foundation:

- FPL API ingestion for players, teams, fixtures, and gameweeks.
- Baseline Premier League match predictor.
- Rule-based FPL optimizer.
- Original Streamlit app at `app/streamlit_app.py`.

## Tier 2 Summary

Tier 2 added the full local xG + FPL ML system:

- Understat xG ingestion.
- xG/xGA features for match prediction.
- FPL player gameweek history and feature engineering.
- XGBoost FPL points model.
- PuLP squad optimizer backed by model predictions.

Some Tier 2 model artifacts are intentionally tracked because the original app depends on them.

## Tier 3 Summary

Tier 3 added multi-season Premier League research and production-transition tooling:

- Historical football-data.co.uk CSV ingestion for 2021-22 through 2025-26.
- Historical Understat xG alignment.
- Time-safe rolling base features.
- Elo feature engineering.
- Walk-forward validation.
- Final untouched 2025-26 holdout evaluation.
- Production model training script.
- Weekly production ingestion, upcoming feature, prediction, and scoring pipeline.
- Production Streamlit dashboard.

Rejected experiment scripts are kept as research archive rather than deleted.

## Production Model

- Production model: `production_logistic_elo_v3`
- Research champion: `logistic_elo_expanding`
- Model family: logistic regression with Tier 3 base + Elo features
- Training seasons: `2021-22` through `2025-26`
- Feature count: 32
- Production draw threshold: 0.30

Production artifacts are local-only and ignored by Git:

```text
models/saved/production_logistic_elo_v3.pkl
models/saved/production_features_v3.json
models/saved/production_draw_threshold_v3.json
models/saved/production_metadata_v3.json
```

## Final Holdout Result

Final holdout season: `2025-26`

Argmax final holdout metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4868 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw F1 | 0.0000 |

Draw overlay final holdout threshold: 0.24

| Metric | Value |
| --- | ---: |
| Accuracy | 0.4684 |
| Log loss | 1.0601 |
| Brier | 0.6372 |
| Draw F1 | 0.1159 |

Main weakness:

- Draw underprediction.
- Home-win bias.
- Some high-confidence wrong predictions.

The draw overlay is documented as a draw-risk helper, not as an improvement to the probability model.

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

The dashboard is read-only by default and handles PostgreSQL unavailability cleanly.

## Weekly Production Pipeline

Run the full local workflow:

```bash
python src/production/run_weekly_pipeline.py --target-season 2026-27
```

Individual steps:

```bash
python src/production/weekly_ingest.py --target-season 2026-27
python src/production/build_upcoming_features.py --target-season 2026-27
python src/production/predict_production_matches.py --target-season 2026-27
python src/production/score_predictions.py --target-season 2026-27
```

The pipeline skips safely when real source data is unavailable. It does not create fake fixtures, predictions, or scores.

## Data Sources

- Fantasy Premier League API
- football-data.co.uk CSVs
- Understat xG

Raw CSVs are local-only and ignored by Git.

## Rejected Experiments

These experiments were built and evaluated, then rejected or kept experimental:

- H2H features
- Style/tactical features
- Pressure index features
- Poisson as a W/D/L replacement
- Calibration variants

The scripts remain in the repository as research archive.

## Reports

Final report source:

```text
docs/football_predictor_report.md
```

PDF generation utility:

```text
scripts/generate_report_pdf.py
```

Generated HTML and PDF outputs are ignored:

```text
docs/Football_Predictor_Report.html
docs/*.pdf
```

## Setup

Install dependencies:

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

## Validation Commands

```bash
python -m py_compile app/production_dashboard.py
python -m py_compile src/tier3_validation.py
python -m py_compile src/production/run_weekly_pipeline.py
python -m py_compile src/production/train_production_model.py
python -m py_compile scripts/generate_report_pdf.py
python src/tier3_validation.py
```

If PostgreSQL is unavailable, `src/tier3_validation.py` should fail clearly within the configured timeout instead of hanging.

## GitHub Safety

The repository is prepared so sensitive and heavy local files stay out of GitHub:

- `.env` is ignored.
- `.streamlit/secrets.toml` is ignored.
- `data/historical/*.csv` is ignored.
- Production model artifacts are ignored.
- Generated report HTML/PDF files are ignored.
- Python cache folders are ignored.

Do not upload raw data, secrets, database dumps, or local production model artifacts.

## Project Layout

```text
Football Predictor Model/
|-- app/
|   |-- streamlit_app.py
|   `-- production_dashboard.py
|-- data/
|   `-- historical/
|-- docs/
|   |-- football_predictor_report.md
|   |-- tier3_experiment_summary.md
|   |-- tier3_final_holdout_report.md
|   `-- tier3_final_error_analysis.md
|-- scripts/
|   `-- generate_report_pdf.py
|-- sql/
|   `-- tier3_schema.sql
|-- src/
|   |-- production/
|   `-- tier3_validation.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Safety Principles

- No random train/test splits on time-ordered match data.
- No post-holdout tuning after seeing 2025-26.
- No fake production fixtures, predictions, or scores.
- No betting claims or bookmaker-beating claims.
- No live Supabase, APK, sentiment, morale, injury, or odds layer in the current model.

---

Purav Desai
