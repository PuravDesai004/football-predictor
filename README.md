# ⚽ Football ML Prediction System

> A multi-model Premier League match predictor and FPL squad optimizer built from scratch using Python, PostgreSQL, XGBoost, and Streamlit.

**Built by Purav Desai** | B.Tech IT Sem 6, SCET Surat  
[GitHub](https://github.com/PuravDesai004) · [LinkedIn](https://linkedin.com/in/puravdesai41)

---

## 🔴 Live Demo
*Deploying in Tier 2 — multi-season data + xG/xGA features (July 2026)*

---

## 📸 App Screenshots

### ⚽ Match Predictor
> Select any two PL teams → Win/Draw/Loss probabilities + predicted scoreline

![Match Predictor](screenshots/match_predictor.png)

### 🏆 FPL Squad Selection
> Mathematically optimal 15-man squad under £100m using PuLP linear programming

![FPL Squad](screenshots/fpl_squad1.png)

### ⭐ FPL Captain Recommendation
> Ceiling-based captain pick — estimated points + 0.4 × form score

![FPL Captain](screenshots/fpl_squad2.png)

---

## 🧠 What This Builds

Three prediction systems sharing one PostgreSQL database and one Streamlit frontend:

| System | Predicts | Algorithm |
|--------|----------|-----------|
| **Match Predictor** | Win/Draw/Loss + scoreline | Logistic Regression + XGBoost |
| **League Winner** | Title race probability | Monte Carlo simulation |
| **FPL Optimizer** | Optimal 15-man fantasy squad | XGBoost + PuLP linear programming |

---

## 🏗️ Architecture

```
FPL Official API + football-data.org
           ↓
   Python (requests + pandas)
   Light cleaning + type fixing
           ↓
   PostgreSQL
   7 SQL views — rolling form,
   clean sheet rate, H2H, FDR
           ↓
   scikit-learn + XGBoost
   Time-based train/test split
   5-fold cross-validation
           ↓
   PuLP linear optimizer
   FPL squad selection
           ↓
   Streamlit app (3 pages)
```

---

## 📊 Tier 1 Model Results

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Logistic Regression | 48.6% ± 2.5% | 55.6% |
| XGBoost Classifier | 41.3% ± 3.3% | 51.4% |

| Score Model | MAE | R² |
|-------------|-----|----|
| Home Goals | 1.04 | -0.29 |
| Away Goals | 0.97 | -0.16 |

> ⚠️ One season of data. Score R² is negative — improves in Tier 2 with
> multi-season data + xG/xGA. Win/Draw/Loss probabilities are the reliable output.

---

## 🔍 Domain Problems Identified

These problems were identified from football domain knowledge before writing any code.
Each has a direct technical solution built into the architecture.

---

### 1. Style Bias — The Arsenal Problem
**Problem identified by author:**
Arsenal finishes 2nd for 3 consecutive seasons yet has low xG and shot volume.
A model trained purely on goals and xG classifies them as a weak attacking team.
Their real value is in what does NOT happen — no transitions allowed, opponents
forced into low-percentage shots from distance. Completely invisible in standard stats.

**Technical solution:**
PPDA (Passes Per Defensive Action) metric + KMeans style clustering groups teams
into tactical archetypes — Arsenal correctly classified as Defensive Block, not Weak Attack.
Style cluster label becomes a feature in XGBoost. *(Tier 2)*

---

### 2. Human Behaviour — The Morale Layer
**Problem identified by author:**
Real-world events like internal team conflicts (e.g. RMA players fighting each other),
player personal problems, or beef between teammates create measurable performance impact.
Pure statistical models ignore this. A team with active internal conflict playing a big
match is fundamentally different from the same team at full harmony — yet their xG
and rolling form numbers look identical.

**Technical solution:**
Morale/Sentiment Layer: team_morale_score (-1 to +1), internal_conflict_flag,
player_confidence_score, rivalry_intensity, psychological_block_score.
Short term: manual Streamlit sliders. Medium term: NewsAPI + HuggingFace transformer. *(Tier 2)*

---

### 3. Psychological Blocks in H2H Matchups
**Problem identified by author:**
Man City repeatedly collapsing against Real Madrid in UCL knockouts despite being
statistically superior. PSG choking in UCL for years before finally winning in 2024-25.
Pattern is too consistent to be random variance — it is systematic and should be modeled.

**Technical solution:**
Psychological Block Score: compare win rate in high-stakes matches (QF/SF/Final)
vs overall win rate. Reverse logic gives Tournament Elevation Score for clubs like
Real Madrid who consistently overperform their league form in Europe. *(Tier 2)*

---

### 4. Managerial Regime Change Breaking Historical Data
**Problem identified by author:**
Using 3 years of Chelsea data across 4 different managers averages across completely
incompatible tactical systems — signal becomes noise. Historical data is not equally
valuable when a team has changed its entire playing philosophy multiple times.

**Technical solution:**
Manager tenure feature, regime change flag, exponential decay weighting —
recent matches under current manager weighted 1.0, older matches under previous
managers decay toward 0.1. Player-level features remain valid across changes. *(Tier 2)*

---

### 5. UCL Performance Does Not Reflect League Performance
**Problem identified by author:**
PSG and Bayern win their leagues every year but fail in UCL at QF/SF stage.
Domestic dominance creates patterns tuned to beating weak opposition — pressing
works against Ligue 1 defenses but gets punished by Dortmund or Real Madrid.
League pressure and UCL pressure are fundamentally different environments.

**Technical solution:**
Competition Context Feature: per-club ratio of domestic vs European performance.
UCL predictor deliberately deferred — insufficient data with one season, format
changed in 2024-25, tournament variance too high. *(Tier 3)*

---

### 6. Data Leakage — Caught and Fixed
**Problem caught before deployment:**
Original model achieved 100% accuracy. Investigation revealed H2H features were
calculated from the same season's matches being predicted — the model already knew
the answers. In production with future fixtures these features would not exist,
making the model useless despite perfect training metrics.

**Fix applied:**
H2H features removed until multi-season data available in Tier 3.
Random split replaced with time-based split — train GW3-GW31, test GW31-GW38.
Honest accuracy: 55.6%.

---

### 7. Small Dataset Limiting Generalization
**Problem identified proactively:**
One PL season = 360 usable rows after cleaning. GPU acceleration does not help —
this is a data problem not a compute problem.

**Solution:**
Player-level rows: 3800 matches → 83,000+ rows. Multi-league in Tier 3:
5 leagues → 400,000+ rows. FPL data: 38 GWs × 500 players × 5 seasons = 95,000 rows. *(Tier 3)*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| Database | PostgreSQL (local) → Supabase (production) |
| ML Models | scikit-learn, XGBoost |
| Optimizer | PuLP (linear programming) |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud + Supabase *(Tier 2)* |
| Model saving | joblib |
| Data sources | FPL Official API, football-data.org |

---

## 📁 Project Structure

```
Football Predictor Model/
├── data/
│   ├── raw/                    ← raw JSON from APIs (gitignored)
│   ├── processed/
│   └── exports/
├── src/
│   ├── data_pipeline.py        ← API fetch + PostgreSQL storage
│   ├── feature_engineering.py  ← 7 SQL views + feature loading
│   ├── train_model.py          ← model training + match prediction
│   └── fpl_optimizer.py        ← PuLP optimizer + captain logic
├── models/saved/               ← .pkl files (gitignored)
├── app/
│   └── streamlit_app.py        ← Streamlit 3-page frontend
├── sql/
│   ├── schema.sql              ← PostgreSQL table definitions
│   └── feature_queries.sql     ← 7 feature engineering views
├── screenshots/                ← app screenshots
├── .env.example                ← credential template
└── requirements.txt
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/PuravDesai004/football-predictor.git
cd football-predictor
pip install -r requirements.txt

# 1. Create PostgreSQL database called football_db
# 2. Copy .env.example to .env and fill in your credentials

python src/data_pipeline.py        # fetch + store data
python src/feature_engineering.py  # create SQL views
python src/train_model.py          # train + save models
streamlit run app/streamlit_app.py # launch app
```

---

## 📚 What I Learned Building This

**Already knew before this project:**
Python (pandas, numpy), SQL basics, MongoDB, Power BI, ML theory
(regression, classification, clustering, overfitting/underfitting, bias-variance tradeoff)

**Learned during Tier 1 (via YouTube + documentation):**
- PostgreSQL window functions: LAG, OVER, PARTITION BY for rolling averages
- SQLAlchemy as Python ↔ PostgreSQL bridge
- XGBoost in practice on real tabular data
- PuLP linear programming for constrained optimization
- Streamlit for building ML apps
- joblib for model serialization
- Time-based train/test splits for time-ordered data
- Data leakage detection and prevention
- Feature importance analysis
- Git version control

---

## 🗺️ Full Roadmap

- [x] **Tier 1** — PL predictor + FPL optimizer + Streamlit app *(complete)*
- [ ] **Tier 2** — xG/xGA + sentiment layer + style clustering *(July 2026)*
- [ ] **Tier 3** — Multi-league + Elo ratings + ensemble stacking *(Aug 2026)*
- [ ] **Tier 4** — PyTorch LSTM + DistilBERT fine-tuning *(Sep 2026)*
- [ ] **Tier 5** — UCL predictor + RL chip strategy *(End 2026)*
- [ ] **F1 System** — FastF1 telemetry + pit stop RL agent *(2027)*

---

## ⚠️ Known Limitations (Tier 1)

- One season of data only — predictions favor historically strong teams
- Score model has negative R² — use win probabilities not exact scores
- No xG/xGA yet — chance quality not captured
- No morale or sentiment layer yet
- FPL points are rule-based estimates not ML predictions
- Database is local — Supabase migration in Tier 2

---

*Part of a larger sports analytics system. F1 prediction module planned using FastF1 library after Tier 3.*