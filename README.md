# Football Predictor

Premier League match prediction and FPL squad optimization, built with XGBoost and real football data.

**Live app:** https://footballpredictort2.streamlit.app/
**GitHub:** https://github.com/PuravDesai004/football-predictor
**Branch:** `tier-2-complete`

---

## What it does

Two tools in one Streamlit app.

**Match Predictor** predicts Win/Draw/Loss probabilities for Premier League fixtures using an XGBoost classifier trained on form, strength ratings, and Understat xG data. It also outputs a predicted scoreline, though that is the weaker output (see limitations).

**FPL Squad Optimizer** picks a valid 15-player FPL squad using an XGBoost points regressor and PuLP linear programming. Respects budget, position limits, and the 3-player-per-club rule. Accounts for player availability before gameweek deadlines.

---

## Tech stack

| Layer | Tool |
|-------|------|
| ML models | XGBoost, Logistic Regression, scikit-learn |
| Optimization | PuLP (linear programming) |
| Data | FPL API, Understat xG |
| Database | Supabase (PostgreSQL) |
| App | Streamlit Community Cloud |
| Language | Python |

---

## How it works

The app loads pre-trained model files from the repository (`models/saved/`) and reads live player data from Supabase. No retraining happens on the server.

```
GitHub repo  →  Streamlit Cloud  →  loads model .pkl files
                      ↓
               Supabase DB  →  live player availability, form, fixtures
```

When you open the app, it fetches current player data from Supabase, runs predictions through the saved XGBoost models, and the optimizer solves the squad selection using PuLP constraints. A pre-deadline refresh button updates player availability without retraining anything.

---

## Match model

The final match model uses 16 features in two groups: 12 form and strength features from the FPL API, and 4 rolling xG/xGA features from Understat.

**Train/test split:** GW3 to GW30 (train) / GW31 to GW38 (test). No gameweek overlap.

| Model | Features | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 12 | 0.570 |
| XGBoost | 12 | 0.506 |
| Logistic Regression | 16 (xG) | 0.532 |
| **XGBoost** | **16 (xG)** | **0.570** |

H2H features are excluded. With one season of data they cause leakage and inflate training accuracy without helping on the held-out test set.

Score prediction is weaker but improved after adding xG:

| Target | MAE | R² |
|--------|-----|----|
| Home Goals | 0.975 | -0.103 |
| Away Goals | 0.747 | -0.025 |

Negative R² on score prediction is expected at this data volume. Scorelines are a supporting output. Win/Draw/Loss probabilities are the reliable one.

---

## FPL points model

Trained on 29,747 player-gameweek rows across 841 players (GW1 to GW38). Only rows where a player has enough prior history for rolling features to be meaningful are used in training.

**Train/test split:** GW4 to GW31 (train) / GW32 to GW38 (test).

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Baseline (rolling avg last 5) | 1.007 | 2.046 | 0.194 |
| **XGBoost** | **0.926** | **1.859** | **0.334** |

Features: previous points, minutes, starts, rolling xG/xA, ICT index, transfers in/out, ownership %, price, opponent difficulty, home/away.

**Leakage check:** double gameweek fixtures share identical historical features, which is the correct pre-deadline behaviour.

```
Duplicate player-gameweek groups: 409
Same-gameweek historical feature mismatch groups: 0
FPL leakage column check passed
```

---

## FPL optimizer

PuLP linear programming selects a valid 15-player squad under these constraints:

- 2 GK / 5 DEF / 5 MID / 3 FWD
- Max 3 players per club
- Budget ≤ 100.0
- Available players only

Outputs: full squad, starter/bench split, captain pick, XGBoost points predictions. Falls back to rule-based scoring if the model file is missing.

---

## Experiments that did not make the final model

**KMeans style clustering:** built a tactical clustering layer using Understat PPDA, deep completions, xG, and xGA. Four clusters: High Press, Direct Attack, Compact Defense, Low Control. Adding style features to the match model dropped accuracy from 0.570 to 0.506. Cut.

**Position-specific FPL models:** trained separate XGBoost regressors per position. Combined MAE came out at 0.948 vs 0.926 for the single model. Single model stays.

Code for both is in `src/` as a starting point for Tier 3.

---

## Current limitations

Football is noisy. One season of Premier League data is 380 matches, which is not much to train on.

- **Scorelines are estimates.** Win/Draw/Loss probabilities are the reliable output. Treat predicted scores as rough anchors, not exact forecasts.
- **Single season of data.** Multi-season history is the main constraint on model quality. That is the first thing Tier 3 fixes.
- **No H2H features.** Intentionally excluded to prevent leakage with one season of data. Reintroduced in Tier 3.
- **Not a betting tool.** Predictions are based on historical form and public data. This is an analytics and ML project, not a financial tool.

---

## Project status

| Tier | Status | What it covers |
|------|--------|----------------|
| Tier 1 | Complete | FPL API pipeline, PostgreSQL, baseline models, Streamlit app, rule-based optimizer |
| **Tier 2** | **Live** | XGBoost models, Understat xG integration, Supabase, Streamlit Cloud deployment |
| Tier 3 | In progress | Multi-season data, Elo features, walk-forward validation, production dashboard |

Tier 3 is waiting on the 2026/27 Premier League season to run with fresh weekly data.

---

## Local setup

```bash
git clone https://github.com/PuravDesai004/football-predictor.git
cd football-predictor
git checkout tier-2-complete

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file (do not commit this):

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=football_db
DB_USER=your_username
DB_PASS=your_password
```

For Supabase, add `DATABASE_URL` to your `.env` with `sslmode=require`. The app handles both local PostgreSQL and Supabase automatically.

```bash
streamlit run app/streamlit_app.py
```

---

## Pipeline commands

```bash
python src/data_pipeline.py            # FPL data
python src/understat_scraper.py        # Understat xG
python src/feature_engineering.py      # match features
python src/train_model.py              # train match model
python src/fpl_history_pipeline.py     # player gameweek history
python src/fpl_feature_engineering.py  # player features
python src/train_fpl_model.py          # FPL points model
streamlit run app/streamlit_app.py     # launch app
```

---

## Data sources

- FPL API (`fantasy.premierleague.com`)
- Understat (`understat.com`)

---

## Design principles

- No random train/test splits on time-ordered data
- No same-gameweek leakage in FPL features
- No H2H features in a single-season model
- No feature kept unless it improves held-out validation
- Score prediction is a supporting output, not the primary metric

---

## Project structure

```
Football Predictor Model/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
├── models/
│   └── saved/
├── sql/
│   ├── schema.sql
│   ├── feature_queries.sql
│   └── fpl_feature_queries.sql
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── fpl_optimizer.py
│   ├── understat_scraper.py
│   ├── clustering.py
│   ├── model_validation.py
│   ├── fpl_history_pipeline.py
│   ├── fpl_feature_engineering.py
│   ├── train_fpl_model.py
│   └── train_fpl_position_models.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

**Purav Desai** | B.Tech IT, SCET Surat | [GitHub](https://github.com/PuravDesai004/football-predictor)
