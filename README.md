# Football Predictor Model

A machine learning system for Premier League match prediction and Fantasy Premier League squad optimization.

**Purav Desai** | B.Tech IT Semester 6, SCET Surat  
[GitHub](https://github.com/PuravDesai004/football-predictor)

---

## Current Status

The project is at **Tier 2.5**, running locally. It predicts Premier League match outcomes (Win/Draw/Loss + approximate scoreline), optimizes a 15-player FPL squad using a trained XGBoost points model, and refreshes player availability data before gameweek deadlines.

Cloud deployment is pending Supabase migration - the current database is local PostgreSQL.

---

## Why This Project Exists

Football prediction is hard for specific reasons. One season gives you 380 Premier League matches. Scores are low and variance is high. Head-to-head features leak future results when you only have a single season of data. And FPL points depend on minutes, availability, role, and fixture difficulty in ways a scoring formula can't fully capture.

The goal here was to build a realistic ML pipeline - time-safe splits, explicit leakage checks, features that actually improve held-out validation - rather than a model that looks impressive in training and fails in practice.

---

## Tier 1

Tier 1 built the data pipeline and the core prediction system.

**Data sources:**
- `https://fantasy.premierleague.com/api/bootstrap-static/`
- `https://fantasy.premierleague.com/api/fixtures/`

**Core PostgreSQL tables:** `players`, `teams`, `fixtures`, `gameweeks`

**SQL views:** `match_results`, `team_season_stats`, `home_form`, `away_form`, `h2h_stats`, `match_features`, `player_fpl_features`

The match model used 12 features. H2H was intentionally excluded - with one season of data it causes leakage and inflates accuracy artificially.

```
home_form_scored        away_form_scored
home_form_conceded      away_form_conceded
home_clean_sheet_rate   away_clean_sheet_rate
home_fdr                away_fdr
strength_overall_home   strength_overall_away
home_team_away_str      away_team_home_str
```

Tier 1 also delivered the Streamlit app, a rule-based FPL optimizer, and a clean project structure ready for GitHub.

---

## Tier 2

Tier 2 added Understat expected goals data to improve match features.

**New data source:** `https://understat.com/getLeagueData/EPL/2025/`

**New tables:** `understat_xg`, `understat_team_history`

**Four new rolling features:**

```
home_xg_last5    away_xg_last5
home_xga_last5   away_xga_last5
```

The final match model uses 16 features (12 original + 4 xG/xGA) with a complete-gameweek time split: **Train GW3-GW30 / Test GW31-GW38.** No gameweek overlap between train and test.

---

## Match Model Results

**Final classifier: XGBoost**

| Model | Features | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 12 (baseline) | 0.570 |
| XGBoost | 12 (baseline) | 0.506 |
| Logistic Regression | 16 (xG) | 0.532 |
| XGBoost | 16 (xG) | **0.570** |

Score prediction is weaker but improved after adding xG:

| Target | MAE | R2 |
|--------|-----|----|
| Home Goals | 0.975 | -0.103 |
| Away Goals | 0.747 | -0.025 |

Scorelines are a supporting output. Win/Draw/Loss probabilities are the reliable prediction - negative R2 on the score model is expected at this data volume and improves in Tier 3 with multi-season history.

---

## Tier 2.5

Tier 2.5 replaced the rule-based FPL optimizer with a proper ML model.

The original optimizer ran a scoring formula. It worked well enough for Tier 1, but it wasn't learning from data - it was just arithmetic. Tier 2.5 trained a real XGBoost regressor on player gameweek history.

**New data source:** `https://fantasy.premierleague.com/api/element-summary/{player_id}/`

**New table:** `player_gameweek_history`

**Dataset:**

| Metric | Value |
|--------|-------|
| Total rows | 29,747 |
| Players | 841 |
| Gameweeks | GW1-GW38 |
| Null `total_points` rows | 0 |

**Feature table:** `player_gameweek_features` - 29,747 rows, 49 columns, 27,224 mature rows.

The model trains only on mature rows where a player has enough prior history for rolling features to be meaningful.

**Features used:** previous points, minutes, starts, rolling xG/xA, rolling ICT index (influence, creativity, threat), transfers in/out, ownership percentage, price, opponent, home/away.

---

## FPL Leakage Prevention

Double gameweeks create a specific leakage risk: if the second fixture in a double gameweek uses stats from the first fixture of the same gameweek, the model has seen data that wouldn't exist before the deadline.

All rolling features here are built from previous gameweeks only.

**Verification output:**

```
Duplicate player-gameweek groups: 409
Same-gameweek historical feature mismatch groups: 0
FPL leakage column check passed
```

Two fixtures in the same gameweek share identical historical features - which is the correct pre-deadline behavior.

---

## FPL Points Model Results

**Final model: Single XGBoost Regressor**

Train: GW4-GW31 / Test: GW32-GW38. No gameweek overlap.

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| Baseline (`points_avg_last5`) | 1.007 | 2.046 | 0.194 |
| XGBoost | **0.926** | **1.859** | **0.334** |

The XGBoost model beats the rolling average baseline on all three metrics. Saved to `models/saved/fpl_points_xgb.pkl`.

---

## FPL Optimizer

The optimizer uses PuLP linear programming to select a valid 15-player squad.

**Constraints:**
- 2 GK / 5 DEF / 5 MID / 3 FWD
- Max 3 players per club
- Budget <= 100.0
- Available players only

**Outputs:** full squad selection, starter/bench split, captain recommendation, XGBoost points prediction. Falls back to rule-based scoring if the model file is missing.

Sanity guards prevent selecting unavailable players or players with very low expected minutes.

---

## Pre-Deadline Refresh

The Streamlit app can pull current FPL player data without retraining anything.

**Fields updated on refresh:** `chance_of_playing_this_round`, `chance_of_playing_next_round`, `status`, `price`, `form`, `selected_by_percent`

**Not updated:** fixtures, `player_gameweek_history`, `player_gameweek_features`, or saved models. The lightweight `player_fpl_features` view can be refreshed for the current app state.

---

## Experiments That Were Rejected

### KMeans Style Clustering

A tactical clustering system was built using Understat team-history features: PPDA, PPDA allowed, deep completions, deep allowed, xG, xGA, npxG, npxGA, xPts, goals scored, goals conceded. Four clusters: High Press, Direct Attack, Compact Defense, Low Control.

Adding style features to the match model made it worse:

| Model | Features | Accuracy |
|-------|----------|----------|
| XGBoost | 16 (xG) | 0.570 |
| XGBoost | 20 (xG + style) | 0.506 |

Style clustering is not in the final model. The code and cluster analysis are kept in `src/clustering.py` as a starting point for Tier 3 research.

### Position-Specific FPL Models

Separate XGBoost regressors were trained per position (GK, DEF, MID, FWD) and combined at prediction time.

| Model | MAE |
|-------|-----|
| Single XGBoost | 0.926 |
| Position-specific combined | 0.948 |

The single model performed better on the combined test metrics, so it remains the final FPL model. Position-specific code stays in `src/train_fpl_position_models.py` but is not used in production.

---

## Streamlit App

Three pages: **Match Predictor**, **FPL Team Selector**, **About**.

The app predicts win/draw/loss probabilities and a predicted score, builds an optimized FPL squad with starter/bench/captain assignments, and runs the pre-deadline player refresh. Active model type is displayed on each prediction page.

Running locally at `http://localhost:8501`. The UI uses native Streamlit components - no custom frontend.

---

## Saved Models

```
models/saved/xgb_classifier.pkl
models/saved/logistic_classifier.pkl
models/saved/scaler.pkl
models/saved/label_encoder.pkl
models/saved/xgb_home_goals.pkl
models/saved/xgb_away_goals.pkl
models/saved/fpl_points_xgb.pkl
models/saved/model_features.json
models/saved/fpl_points_features.json
```

---

## Project Structure

```
Football Predictor Model/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   `-- raw/
|-- models/
|   `-- saved/
|-- sql/
|   |-- schema.sql
|   |-- feature_queries.sql
|   `-- fpl_feature_queries.sql
|-- src/
|   |-- data_pipeline.py
|   |-- feature_engineering.py
|   |-- train_model.py
|   |-- fpl_optimizer.py
|   |-- understat_scraper.py
|   |-- clustering.py
|   |-- manager_features.py
|   |-- model_validation.py
|   |-- fpl_history_pipeline.py
|   |-- fpl_feature_engineering.py
|   |-- train_fpl_model.py
|   `-- train_fpl_position_models.py
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Local Setup

```bash
git clone https://github.com/PuravDesai004/football-predictor.git
cd "football-predictor"

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:
Create a local `.env` file. Do not commit it to GitHub.

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=football_db
DB_USER=your_username
DB_PASS=your_password
```

```bash
streamlit run app/streamlit_app.py
```

---

## Pipeline Commands

```bash
python src/data_pipeline.py            # FPL data pipeline
python src/understat_scraper.py        # Understat xG data
python src/feature_engineering.py      # match features
python src/train_model.py              # train match model
python src/fpl_history_pipeline.py     # player gameweek history
python src/fpl_feature_engineering.py  # player features
python src/train_fpl_model.py          # train FPL model
streamlit run app/streamlit_app.py     # launch app
```

---

## Data Sources

- Official FPL API (`fantasy.premierleague.com`)
- Understat (`understat.com`)

`football-data.org` is not currently used.

---

## Deployment

The code is ready for GitHub upload after final review.

Cloud deployment requires a hosted PostgreSQL database. Recommended path:

1. Push Tier 2.5 to GitHub, tag `v2.5`, create branch `tier-2.5-complete`
2. Migrate local PostgreSQL to Supabase
3. Add `DATABASE_URL` to Streamlit Cloud secrets with `sslmode=require`
4. Deploy via Streamlit Cloud

The app already handles `DATABASE_URL`, Streamlit secrets, `sslmode=require`, and falls back to `.env` for local use.

---

## What Is Not Done Yet

- Supabase migration
- Streamlit Cloud deployment
- Weekly automated data refresh
- Multi-season match data
- Elo ratings
- Poisson scoreline matrix
- Sentiment and morale layer
- Managerial regime-change features
- Psychological block score
- Competition context features

---

## Tier 3

Tier 3 focuses on the data volume problem. One season of 380 matches is the main constraint - the fix is more data, not more compute.

Planned work:
- Multi-season Premier League data
- Elo ratings
- Poisson scoreline probability matrix
- Time-decay weighting for older seasons
- H2H features reintroduced only after multi-season data is available
- Weekly automated refresh pipeline

---

## Tier 3.5

Tier 3.5 is the research-heavy phase. Possible additions:

- Multi-league data
- Champions League context and competition-specific performance ratios
- Team morale and news sentiment (NewsAPI + HuggingFace transformer)
- Injury and availability intelligence from match-week news
- Managerial regime-change detection with exponential decay weighting
- Psychological block scores for recurring H2H matchups
- Tactical matchup features
- Advanced FPL transfer planning

The same standard applies throughout: any feature that improves training accuracy but fails time-safe cross-validation gets cut.

---

## Design Principles

- No random train/test splits on time-ordered data
- No same-gameweek leakage in FPL features
- No H2H features in single-season model training
- No feature kept because it sounds useful - it has to improve held-out validation
- Score prediction is a supporting output, not the primary metric
- FPL model uses only prior-gameweek history, never same-gameweek outcome data

---

**Purav Desai** | B.Tech IT Semester 6, SCET Surat
