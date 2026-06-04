# Tier 3 Style/Tactical Source Audit

## Status

Phase 7A is a source audit only. No style or tactical feature tables were created, no model training was run, and no database writes were performed.

The `2025-26` season remains the untouched final holdout. It must not be used for model selection, tuning, calibration, or reported model metrics.

## Sources Inspected

Inspected local files and folders:

- `src/`
- `sql/tier3_schema.sql`
- `docs/tier3_experiment_summary.md`
- `data/`
- `src/tier3_understat_pipeline.py`
- `src/understat_scraper.py`
- `sql/feature_queries.sql`
- `src/clustering.py`
- `src/build_tier3_features.py`
- `src/build_tier3_elo_features.py`

Inspected database tables and views:

- `historical_matches`
- `historical_understat_xg`
- `match_features_v3_elo`
- `match_features_v3_base`
- `match_features_v3_h2h_experiment`
- `understat_team_history`
- `understat_xg`
- `team_style_clusters`
- `team_xg_stats`
- `team_tactical_match_stats`
- `team_style_form`
- `home_xg_form`
- `away_xg_form`
- `match_features`
- `fixtures`
- `teams`
- `players`
- `player_gameweek_history`
- `player_gameweek_features`

All database inspection was read-only.

## Current Available Local Sources

### `historical_matches`

- Source name: local football-data.co.uk historical match ingestion
- Row count: 1900
- Seasons covered: `2021-22`, `2022-23`, `2023-24`, `2024-25`, `2025-26`, 380 rows each
- Granularity: match-level
- Important columns: `match_id`, `season_id`, `match_date`, `kickoff_time`, `home_team`, `away_team`, `home_goals`, `away_goals`, `result`, `source_file`
- Reliable match date: yes
- Join to `historical_matches`: source table
- Pre-match safe: targets and same-match result fields are not pre-match safe; prior rolling transforms from earlier rows are safe
- Leakage risk rating: medium
- Recommendation: usable only after transformation

### `historical_understat_xg`

- Source name: Tier 3 Understat match xG import
- Row count: 1900
- Seasons covered: `2021-22`, `2022-23`, `2023-24`, `2024-25`, `2025-26`, 380 rows each
- Granularity: match-level
- Important columns: `understat_match_id`, `season_id`, `match_date`, `home_team`, `away_team`, `home_xg`, `away_xg`, `home_goals`, `away_goals`
- Reliable match date: yes
- Join to `historical_matches`: yes, by `season_id`, `match_date`, `home_team`, `away_team`; current coverage is complete at 1900 joined rows
- Pre-match safe: same-match `home_xg` and `away_xg` are post-match stats; prior rolling xG/xGA built with `event_time < current_event_time` is pre-match safe
- Leakage risk rating: low after prior-match transformation, high if same-match xG is used directly
- Recommendation: usable now for prior rolling xG/xGA only

### `match_features_v3_base`

- Source name: Tier 3 derived rolling form/xG feature table
- Row count: 1900
- Seasons covered: `2021-22`, `2022-23`, `2023-24`, `2024-25`, `2025-26`, 380 rows each
- Granularity: match-level feature rows
- Important columns: rolling goals, points, xG, xGA, home/away split counts, result target columns
- Reliable match date: yes
- Join to `historical_matches`: yes, by `match_id`
- Pre-match safe: existing rolling features were built from prior team rows using `event_time < current_event_time`; target columns are not features
- Leakage risk rating: low for existing feature columns, high for target/result columns
- Recommendation: usable now as reference/baseline; do not treat as a new style source

### `match_features_v3_elo`

- Source name: Tier 3 derived base plus pre-match Elo feature table
- Row count: 1900
- Seasons covered: `2021-22`, `2022-23`, `2023-24`, `2024-25`, `2025-26`, 380 rows each
- Granularity: match-level feature rows
- Important columns: all `match_features_v3_base` features plus `home_elo_before`, `away_elo_before`, `elo_diff_before`, `elo_diff_home_adjusted`, `expected_home_score`, `expected_away_score`
- Reliable match date: yes
- Join to `historical_matches`: yes, by `match_id`
- Pre-match safe: pre-match Elo columns are safe; result/goal columns are targets
- Leakage risk rating: low for feature columns, high for target/result columns
- Recommendation: usable now as the current model-selection reference, not as a tactical source

### Local `data/historical/E0_*.csv`

- Source name: local football-data.co.uk Premier League CSV files
- Row count: 5 files present, expected 380 completed rows per season after parsing
- Seasons covered: `2021-22`, `2022-23`, `2023-24`, `2024-25`, `2025-26`
- Granularity: match-level
- Important columns observed: `Date`, `Time`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, cards/fouls, odds columns
- Reliable match date: yes after parser normalization
- Join to `historical_matches`: yes after team-name normalization, by season/date/home/away; current ingestion stores only score/result fields
- Pre-match safe: same-match shots, corners, cards, fouls, and odds are not pre-match safe; prior rolling shot/corner transforms can be safe if parsed into team-match rows and filtered by `event_time < current_event_time`
- Leakage risk rating: medium
- Recommendation: usable only after transformation; do not use odds columns

### `understat_xg`

- Source name: older Tier 2 Understat xG table
- Row count: 380
- Seasons covered: season `2025` only
- Granularity: match-level
- Important columns: `match_date`, `home_team`, `away_team`, `home_xg`, `away_xg`, `home_goals`, `away_goals`, `season`
- Reliable match date: yes
- Join to `historical_matches`: possible by date/home/away after name normalization, but only overlaps the reserved `2025-26` season
- Pre-match safe: same-match xG is post-match; prior rolling rows would be safe only inside the 2025 season, which is the reserved final holdout
- Leakage risk rating: high for Tier 3 development, because it covers only the final holdout season
- Recommendation: reject for Tier 3 model selection

### `understat_team_history`

- Source name: older Tier 2 Understat team-history table
- Row count: 760
- Seasons covered: season `2025` only
- Granularity: team-match-level, two rows per match
- Important columns: `team_name`, `match_date`, `venue`, `xg`, `xga`, `npxg`, `npxga`, `npxgd`, `ppda`, `ppda_allowed`, `deep`, `deep_allowed`, `scored`, `missed`, `xpts`, `pts`
- Reliable match date: yes
- Join to `historical_matches`: possible through team/date/opponent after deriving both team perspectives, but only for `2025-26`
- Pre-match safe: safe only after prior-match rolling transformation; not safe as same-match stats
- Leakage risk rating: high for Tier 3 development, because it covers only the final holdout season
- Recommendation: reject for current Tier 3 model selection; usable only if equivalent historical seasons are imported later

### `team_style_clusters`

- Source name: older style clustering output table
- Row count: 760
- Seasons covered: `2025-26` current FPL/Understat season only
- Granularity: team-match-level
- Important columns: `fixture_id`, `gameweek`, `match_date`, `team_name`, `opponent_name`, `venue`, `style_matches_last5`, `style_cluster`, `style_label`, rolling PPDA/deep/xG/xGA/npxG columns
- Reliable match date: yes
- Join to `historical_matches`: possible only for `2025-26`; team names include older FPL naming such as `Spurs`, so normalization risk exists
- Pre-match safe: the rolling stats appear prior-match based, but clusters were learned by a separate script that can save artifacts and write DB tables
- Leakage risk rating: high for Tier 3 model selection, because it is current/final-holdout only and model-artifact generating
- Recommendation: reject for Tier 3 Phase 7B; do not reuse directly

### Views: `team_xg_stats`, `team_tactical_match_stats`, `team_style_form`, `home_xg_form`, `away_xg_form`, `match_features`

- Source name: older SQL feature views
- Row count: `team_xg_stats` 760, `team_tactical_match_stats` 760, `team_style_form` 760, `home_xg_form` 380, `away_xg_form` 380, `match_features` 380
- Seasons covered: season `2025` only
- Granularity: match-level or team-match-level depending on view
- Important columns: rolling xG/xGA, PPDA, deep/deep_allowed, style cluster ids, fixture strength/FDR
- Reliable match date: yes
- Join to `historical_matches`: possible only for `2025-26`; older views are built around Tier 2/FPL `fixtures` and `teams`
- Pre-match safe: some rolling view windows use previous rows, but the views are final-holdout-only and include same-season current FPL strength/FDR concepts that are not allowed in historical Tier 3
- Leakage risk rating: high
- Recommendation: reject for direct Tier 3 model selection; use only as design reference

### `fixtures`, `teams`, `players`, `player_gameweek_history`, `player_gameweek_features`

- Source name: Tier 2/FPL tables
- Row counts: `fixtures` 380, `teams` 20, `players` 841, `player_gameweek_history` 29747, `player_gameweek_features` 29747
- Seasons covered: current FPL season only for fixture/team context; player history is FPL-specific
- Granularity: fixture-level, team season/current state, player-level, player-gameweek-level
- Important columns: FPL fixture difficulty, current team strength, player expected stats, player rolling features
- Reliable match date: `fixtures` and player gameweek rows have kickoff dates; `teams` is current-state only
- Join to `historical_matches`: not safely across all Tier 3 seasons; mainly current season
- Pre-match safe: not for historical Tier 3 model selection unless reconstructed historically by event time
- Leakage risk rating: high
- Recommendation: reject for Tier 3 style/tactical features

## Understat Tactical/Style Potential

Only recommend features that can be built from prior matches with:

```text
source_event_time < current_match_event_time
```

### Rolling Team xG For

- Available now: yes, from `historical_understat_xg`
- Safe rule: convert each match into team-perspective rows, then average prior `xg_for` rows where `event_time < current_event_time`
- Current status: already partly represented in `match_features_v3_base` as rolling xG columns
- Recommendation: usable now, but Phase 7B should avoid duplicating existing columns unless testing a clearly different window or interaction

### Rolling Team xG Against

- Available now: yes, from `historical_understat_xg`
- Safe rule: convert each match into team-perspective rows, then average prior `xg_against` rows where `event_time < current_event_time`
- Current status: already partly represented in `match_features_v3_base` as rolling xGA columns
- Recommendation: usable now, but incremental value may be limited

### Rolling Shot Volume

- Available in current Understat Tier 3 table: no
- Available locally elsewhere: yes, football-data CSVs include `HS`, `AS`, `HST`, `AST`
- Safe rule if used later: parse raw CSV shot counts into team-match rows and roll only prior matches by event time
- Recommendation: usable only after transformation from local CSVs, not from current Understat tables

### Rolling Deep Completions

- Available in current Tier 3 Understat table: no
- Available in older `understat_team_history`: yes, `deep` and `deep_allowed`, but only season `2025`
- Safe rule if imported later: use multi-season Understat team history and roll prior team-match rows only
- Recommendation: reject now for Tier 3 model selection; usable only after multi-season historical import

### Rolling PPDA

- Available in current Tier 3 Understat table: no
- Available in older `understat_team_history`: yes, `ppda` and `ppda_allowed`, but only season `2025`
- Safe rule if imported later: use multi-season Understat team history and roll prior team-match rows only
- Recommendation: reject now for Tier 3 model selection; usable only after multi-season historical import

### Rolling Pressure/Pressing Proxy

- Available now: partial only
- Candidate columns: `ppda`, `ppda_allowed`, possibly opponent deep completions allowed
- Blocking issue: PPDA/deep columns are only in final-holdout-season Tier 2 Understat tables, not multi-season Tier 3 history
- Recommendation: reject now for Tier 3 model selection

### Rolling Attacking Tempo Proxy

- Available now: partial only
- Candidate columns: xG rate from `historical_understat_xg`; shot/corner volume from football-data CSVs after parsing; `deep` only if multi-season Understat team history is added later
- Recommendation: usable only after transformation, and only for prior rolling rows

### Rolling Defensive Concession Proxy

- Available now: yes for xGA from `historical_understat_xg`
- Candidate columns: rolling xGA, rolling goals conceded, rolling opponent xG allowed
- Blocking issue for richer style: no multi-season PPDA/deep_allowed in current Tier 3 tables
- Recommendation: usable now for xGA-only proxies; richer style version needs better source

## Style/Tactical Features Not Allowed Yet

Explicitly rejected:

- Full-season averages known only after the season
- Current league-table position unless reconstructed match by match
- End-of-season possession or PPDA
- Post-match stats from the same match
- Manually labeled team style without dated evidence
- Any feature that requires future matches
- Current FPL team strength or FDR in historical Tier 3 models
- `team_style_clusters` as currently stored, because it is final-holdout-season only and was produced by an artifact-writing clustering pipeline
- `understat_team_history` PPDA/deep features for model selection until equivalent historical development seasons exist

## Candidate Phase 7B Feature Groups

Phase 7B is allowed only if it stays small, prior-only, and development-only. Do not evaluate `2025-26`.

### Rolling Attacking Volume

- Source table: `historical_understat_xg`
- Required columns: `season_id`, `match_date`, `home_team`, `away_team`, `home_xg`, `away_xg`
- Exact time-safe rule: build team-perspective rows and calculate rolling means/counts from rows with `event_time < current_event_time`
- Join key: `season_id`, `match_date`, `home_team`, `away_team` to `historical_matches`; final feature row by `match_id`
- Expected null behavior: first team match has no prior rows; early season rows should remain null or be imputed inside model folds
- Why it might help: captures attacking chance volume independent of actual goals
- Leakage risk: low if prior-only, high if same-match xG is joined directly

### Rolling Defensive Concession Volume

- Source table: `historical_understat_xg`
- Required columns: `season_id`, `match_date`, `home_team`, `away_team`, `home_xg`, `away_xg`
- Exact time-safe rule: build team-perspective rows and roll prior `xg_against` only
- Join key: `match_id` after joining Understat rows to `historical_matches`
- Expected null behavior: early rows have null prior xGA; do not fill from future rows
- Why it might help: captures defensive concession quality better than goals conceded alone
- Leakage risk: low if prior-only

### Rolling Tempo/Shot Pressure Proxy

- Source table: local `data/historical/E0_*.csv`, not current Tier 3 DB tables
- Required columns: `Date`, `Time`, `HomeTeam`, `AwayTeam`, `HS`, `AS`, `HST`, `AST`, optionally `HC`, `AC`
- Exact time-safe rule: parse into team-match rows, normalize teams, join to `historical_matches`, and roll prior shot/corner counts with `event_time < current_event_time`
- Join key: season plus normalized date/home/away to `historical_matches`
- Expected null behavior: early team rows null until prior matches exist
- Why it might help: approximates attacking tempo and territorial pressure when Understat shot counts are unavailable
- Leakage risk: medium, because source columns are post-match and must never be used from the current fixture

### Home/Away Style Imbalance

- Source table: `historical_understat_xg`, optionally parsed football-data shot/corner team rows
- Required columns: venue-specific prior `xg_for`, `xg_against`, and optionally shots/corners
- Exact time-safe rule: compare home team's prior home profile against away team's prior away profile, using only rows before current event time
- Join key: feature row by `match_id`
- Expected null behavior: promoted teams and early season rows will have sparse venue-specific history; nulls should be imputed inside folds
- Why it might help: captures style mismatch between home attacking pressure and away concession profile
- Leakage risk: low to medium depending on whether CSV shot/corner parsing is added correctly

### xG-Based Style Matchup Deltas

- Source table: `historical_understat_xg`
- Required columns: prior rolling `xg_for`, `xg_against`, venue split counts
- Exact time-safe rule: compute deltas such as home prior xG for minus away prior xGA, and away prior xG for minus home prior xGA, only from prior rows
- Join key: `match_id`
- Expected null behavior: null when either side lacks prior history; impute inside fold
- Why it might help: represents matchup pressure more directly than standalone team form
- Leakage risk: low if derived after prior-only rolling values are built

## Recommendation

`PROCEED_TO_PHASE_7B_WITH_UNDERSTAT_STYLE_EXPERIMENT`

Scope restriction:

- Proceed only with a narrow Understat xG/xGA style-proxy experiment based on `historical_understat_xg`.
- Do not use `understat_team_history`, `team_style_clusters`, PPDA, deep completions, or current FPL style views for Tier 3 model selection.
- If shot/corner tempo proxies are desired, first parse local football-data CSV shot/corner columns into a prior-only team-match source in a separate clearly scoped step.

## Count Stability

Database inspection was read-only. Inspected row counts are expected to remain unchanged before and after Phase 7A:

- `historical_matches`: 1900
- `historical_understat_xg`: 1900
- `match_features_v3_base`: 1900
- `match_features_v3_elo`: 1900
- `match_features_v3_h2h_experiment`: 1900
- `understat_xg`: 380
- `understat_team_history`: 760
- `team_style_clusters`: 760
- `fixtures`: 380
- `teams`: 20
- `players`: 841
- `player_gameweek_history`: 29747
- `player_gameweek_features`: 29747
