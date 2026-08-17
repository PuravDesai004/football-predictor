# Restart Context Prompt

Copy the text below into a new Codex chat after restoring this project on a reset computer.

```text
You are continuing the Football Predictor Model project directly. Work in the restored project folder and inspect the actual files before making assumptions.

PROJECT
-------
Project name: Football Predictor Model
This is a local Premier League match-prediction and Fantasy Premier League analysis system. It is a research and portfolio project, not betting advice. The project uses local Python, PostgreSQL, Streamlit, football-data.co.uk CSV data, Understat xG data, and the Fantasy Premier League API.

FIRST ACTIONS AFTER RESTORE
---------------------------
1. Read README.md, docs/RESET_SETUP_GUIDE.md, and this prompt.
2. Check git status, but do not reset, delete, or overwrite user changes.
3. Confirm Python, the new .venv, PostgreSQL, and the dependencies from requirements.txt.
4. Confirm that .env exists locally without printing its values. Never expose DB_PASS or service credentials.
5. Confirm the PostgreSQL football_db database is restored from postgresql_football_db.dump before diagnosing missing database data.
6. Use targeted checks. Do not repeat a complete historical audit unless a targeted integrity check finds a real problem.

NON-NEGOTIABLE RULES
--------------------
- Work directly in the project.
- Do not use another coding agent.
- Do not commit, push, tag, merge, rebase, or reset unless the user explicitly asks for that exact Git operation.
- Do not delete existing user changes.
- Do not modify Tier 2 Streamlit code, Tier 2 tables, historical tables, database schemas, or model artifacts unless the user explicitly scopes a request to do so.
- Do not retrain models or run tuning during dashboard, pipeline, or verification work.
- Do not create fake fixtures, fake results, fake xG, fake FPL history, or fake predictions.
- Do not evaluate or tune on 2025-26 as a development shortcut.
- Do not add H2H, sentiment, odds, injury, manager, rivalry, style, or pressure features.
- Do not expose passwords, database URLs, service credentials, or private local configuration.

PROJECT TIERS
-------------
Tier 1 is the original local PostgreSQL/FPL foundation, baseline match predictor, rule-based optimizer, and original Streamlit app at app/streamlit_app.py.

Tier 2 is the full local xG + FPL ML system. Preserve these Tier 2 tables and their counts:
- players: 841
- teams: 20
- fixtures: 380
- gameweeks: 38
- player_gameweek_history: 29747
- player_gameweek_features: 29747

Tier 3 is the multi-season research and production-transition layer. It includes historical football-data CSV ingestion, Understat xG alignment, time-safe rolling features, Elo, walk-forward validation, production ingestion, upcoming-feature construction, frozen prediction, scoring, FPL v3 prediction, optimization, and the production Streamlit dashboard.

PRODUCTION MATCH MODEL
----------------------
Official model: production_logistic_elo_v3
Feature artifact: models/saved/production_features_v3.json
The frozen match model has 32 production features. Keep its feature contract exact. Validate numeric columns, feature count, forbidden features, duplicate fixture IDs, same-team fixtures, Elo formulas, expected-score sums, and count ranges.

For genuinely new current-season FPL teams, the production upcoming-feature builder uses explicit cold start:
- exact FPL team-name self-mapping in production_team_name_mapping;
- source such as fpl_bootstrap_run_11:new_team_cold_start;
- warning output identifying the team;
- neutral in-memory Elo of 1400.0;
- zero prior match counts and zero rolling goals, xG, xGA, points, clean-sheet, and related state;
- no invented historical rows and no writes to historical state.

FPL V3 MODEL
------------
The local candidate artifact is under data/production_artifacts/ and contains:
- fpl_points_v3_candidate.pkl
- fpl_points_v3_candidate_features.json
- fpl_points_v3_candidate_metadata.json

The FPL v3 model has 20 prior-only features. Training seasons are 2020-21 through 2023-24. Validation is 2024-25. The final holdout is 2025-26 and must remain untouched for tuning/model selection. Do not retrain while restoring or verifying the project.

CURRENT PRODUCTION DATA CONTEXT
-------------------------------
The latest live FPL bootstrap used target season 2026-27 and produced 587 current FPL players and 380 fixtures. The 2026-27 fixtures were unfinished at the last known state, so safe gameweek ingestion skipped with SKIPPED_NO_COMPLETED_GAMEWEEK. Do not manufacture completed results.

Known non-blocking source warnings:
- football-data can fail to resolve a team name such as Altrincham;
- Understat can report that the 2026 response has no dates data.

Do not create fake data to hide either warning. Report source availability clearly.

DATABASE TABLES
---------------
Production tables include fpl_player_gameweek_history_v3, fpl_player_features_v3, fpl_player_identity_map_v3, fpl_model_training_runs_v3, fpl_player_predictions_v3, fpl_optimizer_outputs_v3, production_fpl_gameweek_snapshots_v3, production_upcoming_match_features_v3, production_match_predictions, production_ingestion_runs, production_data_freshness, and production_fpl_bootstrap_snapshots.

The last known counts were:
- fpl_player_gameweek_history_v3: 253890
- fpl_player_features_v3: 244737
- fpl_player_identity_map_v3: 7358
- fpl_model_training_runs_v3: 2
- fpl_player_predictions_v3: 0 before the FPL prediction runner was executed
- fpl_optimizer_outputs_v3: 0 before the optimizer was executed
- production_fpl_gameweek_snapshots_v3: 0 before GW1 completed

The database is separate from the project folder. The PostgreSQL custom dump postgresql_football_db.dump must be restored before expecting database rows. Keep the dump private.

PRODUCTION DASHBOARD
--------------------
app/production_dashboard.py is the Tier 3 production dashboard. Preserve existing tabs and styling. The dashboard is read-only for database data and should handle PostgreSQL unavailability, stale data, partial ingestion, missing snapshots, null captain/vice-captain values, and empty prediction/optimizer tables safely.

The current dashboard includes FPL Predictions and operator controls:
- Refresh Dashboard only reruns current database queries;
- Run Weekly Pipeline is not automatic;
- it checks the latest FPL deadline in production_fpl_bootstrap_snapshots;
- it blocks when the deadline is missing, less than four hours away, the session already has an active run, or the snapshot is clearly stale;
- when allowed it invokes src/production/run_weekly_pipeline.py with an argument list and a 600-second timeout, never shell=True;
- it displays running, success, partial, and failure states, exit code, concise output, and expandable full output;
- after completion it refreshes match predictions, FPL predictions, optimizer output, pipeline status, freshness, and warnings.

EXPECTED DASHBOARD DATA WHEN THE DATABASE IS CURRENT
-----------------------------------------------------
- approximately 587 FPL player prediction rows;
- one optimizer output;
- a 15-player optimized squad;
- an 11-player starting XI;
- visible captain and vice-captain;
- approximately 380 upcoming/match prediction rows when the current production run has processed all fixtures.

SAFE WORKFLOW FOR FUTURE TASKS
------------------------------
1. State the exact files in scope.
2. Inspect the current implementation and database state.
3. Make the smallest targeted change.
4. Run focused compilation/tests and report exact results.
5. Check Git status and artifact hashes if relevant.
6. Clearly report warnings, skipped stages, stale data, and PostgreSQL-unavailable behavior.

Never claim that a model was trained, a pipeline completed, or a table was populated unless the command output and database counts prove it.
```
