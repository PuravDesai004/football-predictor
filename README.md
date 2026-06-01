# Football Predictor Model

A machine learning system for Premier League match prediction and Fantasy Premier League squad optimization.

**Purav Desai** | B.Tech IT Semester 6, SCET Surat
[GitHub](https://github.com/PuravDesai004/football-predictor)

---

## Current Status

**Tier 2 - running locally.** The system predicts Premier League match outcomes
(Win/Draw/Loss + approximate scoreline), builds an optimized 15-player FPL squad
using a trained XGBoost points model, and can refresh player availability before
gameweek deadlines.

Cloud deployment is pending Supabase migration - the current database is local PostgreSQL.

---

## Why This Project Exists

Football prediction is hard for specific, non-obvious reasons. One season gives you
380 Premier League matches. Scores are low and variance is high. Head-to-head features
leak future results when only a single season of data is available. FPL points depend
on minutes, availability, role, and fixture difficulty in ways no scoring formula can
fully replicate.

The goal was a realistic ML pipeline - time-safe splits, explicit leakage checks,
features that hold up in validation - not a model that looks good in training and
breaks against real fixtures.

---

## Tier 1

Tier 1 built the data pipeline and the core prediction system.

**Data sources:**
- `https://fantasy.premierleague.com/api/bootstrap-static/`
- `https://fantasy.premierleague.com/api/fixtures/`

**Core PostgreSQL tables:** `players`, `teams`, `fixtures`, `gameweeks`

**SQL views:** `match_results`, `team_season_stats`, `home_form`, `away_form`,
`h2h_stats`, `match_features`, `player_fpl_features`

The match model trained on 12 features. H2H was intentionally excluded - with one
season of data it causes leakage and inflates accuracy to unrealistic levels.

```
home_form_scored        away_form_scored
home_form_conceded      away_form_conceded
home_clean_sheet_rate   away_clean_sheet_rate
home_fdr                away_fdr
strength_overall_home   strength_overall_away
home_team_away_str      away_team_home_str
```

Tier 1 also delivered the Streamlit app, a rule-based FPL optimizer, and the full
project structure.

---

## Tier 2

Tier 2 added Understat expected goals data and replaced the rule-based FPL optimizer
with a proper XGBoost model trained on player gameweek history.

**New data source:** `https://understat.com/getLeagueData/EPL/2025/`

**New tables:** `understat_xg`, `understat_team_history`

**Four new rolling features:**

```
home_xg_last5    away_xg_last5
home_xga_last5   away_xga_last5
```

The final match model uses 16 features (12 original + 4 xG/xGA) with a
complete-gameweek time split: **Train GW3-GW30 / Test GW31-GW38.**
No overlap between train and test.

---

## Match Model Results

**Final classifier: XGBoost**

| Model | Features | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 12 (baseline) | 0.570 |
| XGBoost | 12 (baseline) | 0.506 |
| Logistic Regression | 16 (xG) | 0.532 |
| XGBoost | 16 (xG) | **0.570** |

Score prediction improved after adding xG but is still weak:

| Target | MAE | R2 |
|--------|-----|----|
| Home Goals | 0.975 | -0.103 |
| Away Goals | 0.747 | -0.025 |

Win/Draw/Loss probabilities are the reliable output. Scorelines are a supporting
feature - negative R2 at this data volume is expected and improves once multi-season
history is added in Tier 3.

---

## FPL XGBoost Model

The Tier 1 optimizer scored players with a fixed formula. That had a clear ceiling -
it wasn't learning from data, it was just arithmetic. Tier 2 trained a real XGBoost
regressor on player gameweek history instead.

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

Training uses only mature rows - players with enough prior history for rolling
features to mean something.

**Features used:** previous points, minutes, starts, rolling xG/xA, rolling ICT
index (influence, creativity, threat), transfers in/out, ownership percentage,
price, opponent, home/away.

---

## FPL Leakage Prevention

Double gameweeks create a specific leakage risk. If the second fixture in a double
gameweek uses stats from the first fixture of the same week, the model has seen
information that wouldn't exist before the deadline.

All rolling features are built from previous gameweeks only.

**Verification output:**

```
Duplicate player-gameweek groups: 409
Same-gameweek historical feature mismatch groups: 0
FPL leakage column check passed
```

Two fixtures in the same gameweek share identical historical features - which is
the correct pre-deadline behavior.

---

## FPL Points Model Results

**Final model: Single XGBoost Regressor**

Train: GW4-GW31 / Test: GW32-GW38. No gameweek overlap.

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| Baseline (`points_avg_last5`) | 1.007 | 2.046 | 0.194 |
| XGBoost | **0.926** | **1.859** | **0.334** |

The XGBoost model beats the last-5 average baseline on all three metrics.
Saved to `models/saved/fpl_points_xgb.pkl`.

---

## FPL Optimizer

The optimizer uses PuLP linear programming to select a valid 15-player squad.

**Constraints:**
- 2 GK / 5 DEF / 5 MID / 3 FWD
- Max 3 players per club
- Budget <= 100.0
- Available players only

**Outputs:** full squad selection, starter/bench split, captain recommendation,
XGBoost points prediction. Falls back to rule-based scoring if the model file
is missing.

Sanity guards prevent selecting unavailable players or players with very low
expected minutes.

---

## Pre-Deadline Refresh

The Streamlit app can pull current FPL player data without retraining anything.

**Fields updated on refresh:** `chance_of_playing_this_round`,
`chance_of_playing_next_round`, `status`, `price`, `form`, `selected_by_percent`

**Not updated:** fixtures, `player_gameweek_history`, `player_gameweek_features`,
or saved models. The lightweight `player_fpl_features` view can be refreshed for
the current app state.

---

## Experiments That Were Rejected

### KMeans Style Clustering

A tactical clustering system was built using Understat team-history features:
PPDA, PPDA allowed, deep completions, deep allowed, xG, xGA, npxG, npxGA, xPts,
goals scored, goals conceded. Four clusters: High Press, Direct Attack, Compact
Defense, Low Control.

Adding the style features made the match model worse:

| Model | Features | Accuracy |
|-------|----------|----------|
| XGBoost | 16 (xG) | 0.570 |
| XGBoost | 20 (xG + style) | 0.506 |

Style clustering is not in the final model. The code and cluster analysis are in
`src/clustering.py` as a starting point for Tier 3 research.

### Position-Specific FPL Models

Separate XGBoost regressors were trained per position (GK, DEF, MID, FWD) and
combined at prediction time.

| Model | MAE |
|-------|-----|
| Single XGBoost | 0.926 |
| Position-specific combined | 0.948 |

The single model performed better on the combined test metrics, so it remains the
final FPL model. Position-specific code stays in `src/train_fpl_position_models.py`
but is not used in production.

---

## Streamlit App

Three pages: **Match Predictor**, **FPL Team Selector**, **About**.

The Match Predictor page shows win/draw/loss probabilities and a predicted scoreline
for any two PL teams. The FPL Team Selector builds an optimized squad with
starter/bench/captain assignments. Both pages show the active model type.
The pre-deadline refresh is also available from the app.

Running locally at `http://localhost:8501`. UI uses native Streamlit components.

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

Create a local `.env` file from `.env.example`. Do not commit it to GitHub.

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

Tier 2 is already versioned on GitHub:

- `main` contains the latest Tier 2 code
- `tier-2-complete` preserves the Tier 2 branch
- `v2.0` preserves the Tier 2 release snapshot

Cloud deployment still requires a hosted PostgreSQL database. Recommended path:

1. Migrate local PostgreSQL to Supabase
2. Add `DATABASE_URL` to Streamlit Cloud secrets with `sslmode=require`
3. Deploy via Streamlit Cloud

The app already handles `DATABASE_URL`, Streamlit secrets, `sslmode=require`,
and falls back to `.env` for local use.

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

## Roadmap

- **Tier 1** - foundation (complete)
- **Tier 2** - current xG + FPL XGBoost system (complete, local)
- **Tier 3** - final advanced model (planned)

---

## Tier 3: Final Advanced Model

The main constraint right now is data volume - 380 matches from one season.
The fix is more data, not more compute.

Planned additions:
- Multi-season Premier League data
- Elo ratings
- Poisson scoreline probability matrix
- Time-decay weighting for older seasons
- H2H features (only once multi-season data exists)
- Weekly automated refresh pipeline
- Multi-league data
- Champions League context and competition-specific performance ratios
- Team morale and news sentiment (NewsAPI + HuggingFace transformer)
- Injury and availability intelligence from match-week news
- Managerial regime-change detection with exponential decay weighting
- Psychological block scores for recurring H2H matchups
- Tactical matchup features
- Advanced FPL transfer planning

The standard throughout: any feature that improves training accuracy but fails
time-safe cross-validation gets cut.

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
