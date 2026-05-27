-- ── QUERY 1: Match results from fixtures ──────────────────────────
-- Creates a clean match results table with home and away team stats.
-- We separate home and away because teams perform differently in each.

CREATE OR REPLACE VIEW match_results AS
SELECT
    fixture_id,
    gameweek,
    team_h                          AS home_team,
    team_a                          AS away_team,
    team_h_score                    AS home_goals,
    team_a_score                    AS away_goals,
    team_h_difficulty               AS home_fdr,
    team_a_difficulty               AS away_fdr,
    CASE
        WHEN team_h_score > team_a_score THEN 'H'
        WHEN team_h_score < team_a_score THEN 'A'
        ELSE 'D'
    END                             AS result,
    CASE
        WHEN team_h_score > team_a_score THEN 1 ELSE 0
    END                             AS home_win,
    CASE
        WHEN team_h_score = team_a_score THEN 1 ELSE 0
    END                             AS is_draw,
    CASE
        WHEN team_h_score < team_a_score THEN 1 ELSE 0
    END                             AS away_win
FROM fixtures
WHERE finished = TRUE;


-- ── QUERY 2: Team season summary ──────────────────────────────────
-- Aggregates each team's full season stats from match results.
-- Used as a baseline feature for team strength.

CREATE OR REPLACE VIEW team_season_stats AS
SELECT
    home_team                           AS team_id,
    COUNT(*)                            AS total_matches,
    SUM(home_goals)                     AS total_goals_scored,
    SUM(away_goals)                     AS total_goals_conceded,
    SUM(home_win)                       AS total_wins,
    SUM(is_draw)                        AS total_draws,
    SUM(CASE WHEN away_goals = 0 THEN 1 ELSE 0 END) AS clean_sheets,
    ROUND(AVG(home_goals)::NUMERIC, 2)  AS avg_goals_scored,
    ROUND(AVG(away_goals)::NUMERIC, 2)  AS avg_goals_conceded
FROM match_results
GROUP BY home_team

UNION ALL

SELECT
    away_team                           AS team_id,
    COUNT(*)                            AS total_matches,
    SUM(away_goals)                     AS total_goals_scored,
    SUM(home_goals)                     AS total_goals_conceded,
    SUM(away_win)                       AS total_wins,
    SUM(is_draw)                        AS total_draws,
    SUM(CASE WHEN home_goals = 0 THEN 1 ELSE 0 END) AS clean_sheets,
    ROUND(AVG(away_goals)::NUMERIC, 2)  AS avg_goals_scored,
    ROUND(AVG(home_goals)::NUMERIC, 2)  AS avg_goals_conceded
FROM match_results
GROUP BY away_team;


-- ── QUERY 3: Rolling form — last 5 home matches ───────────────────
-- For each match, calculates the average goals scored and conceded
-- in the previous 5 HOME games for the home team.
-- This is more predictive than season averages.

CREATE OR REPLACE VIEW home_form AS
SELECT
    fixture_id,
    gameweek,
    home_team,
    home_goals,
    away_goals,
    ROUND(AVG(home_goals) OVER (
        PARTITION BY home_team
        ORDER BY gameweek, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS home_form_scored,
    ROUND(AVG(away_goals) OVER (
        PARTITION BY home_team
        ORDER BY gameweek, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS home_form_conceded,
    ROUND(AVG(CASE WHEN away_goals = 0 THEN 1.0 ELSE 0.0 END) OVER (
        PARTITION BY home_team
        ORDER BY gameweek, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS home_clean_sheet_rate
FROM match_results;


-- ── QUERY 4: Rolling form — last 5 away matches ───────────────────
-- Same as above but for the away team's last 5 away games.

CREATE OR REPLACE VIEW away_form AS
SELECT
    fixture_id,
    gameweek,
    away_team,
    away_goals,
    home_goals,
    ROUND(AVG(away_goals) OVER (
        PARTITION BY away_team
        ORDER BY gameweek, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS away_form_scored,
    ROUND(AVG(home_goals) OVER (
        PARTITION BY away_team
        ORDER BY gameweek, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS away_form_conceded,
    ROUND(AVG(CASE WHEN home_goals = 0 THEN 1.0 ELSE 0.0 END) OVER (
        PARTITION BY away_team
        ORDER BY gameweek, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS away_clean_sheet_rate
FROM match_results;


-- ── QUERY 5: H2H (Head to Head) win rate ──────────────────────────
-- For every pair of teams that have faced each other,
-- calculates the home team historical win rate.
-- Captures the psychological block feature we designed.

CREATE OR REPLACE VIEW h2h_stats AS
SELECT
    home_team,
    away_team,
    COUNT(*)                                        AS h2h_matches,
    SUM(home_win)                                   AS h2h_home_wins,
    SUM(away_win)                                   AS h2h_away_wins,
    SUM(is_draw)                                    AS h2h_draws,
    ROUND(AVG(home_win)::NUMERIC, 2)                AS h2h_home_win_rate,
    ROUND(AVG(home_goals)::NUMERIC, 2)              AS h2h_avg_home_goals,
    ROUND(AVG(away_goals)::NUMERIC, 2)              AS h2h_avg_away_goals
FROM match_results
GROUP BY home_team, away_team;


-- ── QUERY 6: Master feature table ────────────────────────────────
-- This is the final table your ML model will train on.
-- Joins everything: match results + home form + away form +
-- H2H stats + team names + fixture difficulty.
-- One row per finished match with all features ready.

CREATE OR REPLACE VIEW match_features AS
SELECT
    mr.fixture_id,
    mr.gameweek,

    -- Team identifiers
    mr.home_team,
    mr.away_team,
    ht.name                             AS home_team_name,
    at.name                             AS away_team_name,

    -- Target variables (what we want to predict)
    mr.home_goals,
    mr.away_goals,
    mr.result,
    mr.home_win,
    mr.is_draw,
    mr.away_win,

    -- Home team rolling form features
    hf.home_form_scored,
    hf.home_form_conceded,
    hf.home_clean_sheet_rate,

    -- Away team rolling form features
    af.away_form_scored,
    af.away_form_conceded,
    af.away_clean_sheet_rate,

    -- Fixture difficulty ratings
    mr.home_fdr,
    mr.away_fdr,

    -- Team strength ratings from teams table
    ht.strength_overall_home,
    ht.strength_overall_away            AS home_team_away_str,
    at.strength_overall_home            AS away_team_home_str,
    at.strength_overall_away,

    -- H2H historical stats
    h2h.h2h_matches,
    h2h.h2h_home_win_rate,
    h2h.h2h_avg_home_goals,
    h2h.h2h_avg_away_goals

FROM match_results mr

LEFT JOIN home_form hf
    ON mr.fixture_id = hf.fixture_id

LEFT JOIN away_form af
    ON mr.fixture_id = af.fixture_id

LEFT JOIN teams ht
    ON mr.home_team = ht.team_id

LEFT JOIN teams at
    ON mr.away_team = at.team_id

LEFT JOIN h2h_stats h2h
    ON mr.home_team = h2h.home_team
    AND mr.away_team = h2h.away_team;


-- ── QUERY 7: FPL player value features ───────────────────────────
-- Features specifically for the FPL points predictor.
-- Combines player stats with their team's upcoming fixture difficulty.

CREATE OR REPLACE VIEW player_fpl_features AS
SELECT
    p.player_id,
    p.first_name,
    p.second_name,
    p.team,
    t.name                              AS team_name,
    p.position,
    p.price,
    p.total_points,
    p.minutes,
    p.goals_scored,
    p.assists,
    p.clean_sheets,
    p.goals_conceded,
    p.bonus,
    p.form,
    p.selected_by_percent,
    p.is_available,
    p.expected_goals,
    p.expected_assists,
    p.expected_goal_involvements,
    p.expected_goals_conceded,
    p.starts,
    p.influence,
    p.creativity,
    p.threat,
    p.ict_index,
    -- Value metric: points per million spent
    ROUND((p.total_points::FLOAT / NULLIF(p.price, 0))::NUMERIC, 2) AS points_per_million,
    -- Minutes ratio: how often does this player actually play
    ROUND((p.minutes::FLOAT / NULLIF(38 * 90, 0))::NUMERIC, 2)      AS minutes_ratio,
    -- Team strength as context for player predictions
    t.strength_overall_home,
    t.strength_overall_away
FROM players p
LEFT JOIN teams t ON p.team = t.team_id;
