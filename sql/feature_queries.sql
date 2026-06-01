-- QUERY 1: Match results from fixtures
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


-- QUERY 2: Team season summary
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


-- QUERY 3: Rolling form - last 5 home matches
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


-- QUERY 4: Rolling form - last 5 away matches
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


-- QUERY 5: H2H (Head to Head) win rate
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


-- QUERY 6: Team xG stats from Understat
-- Joins Understat xG rows to FPL fixtures and outputs one row from
-- each team's perspective, so every fixture has a home and away row.

CREATE OR REPLACE VIEW team_xg_stats AS
SELECT
    f.fixture_id,
    f.gameweek,
    ux.match_date,
    ht.name                         AS team_name,
    at.name                         AS opponent_name,
    'H'                             AS venue,
    ux.home_xg                      AS team_xg,
    ux.away_xg                      AS opponent_xg,
    ux.home_goals                   AS goals_for,
    ux.away_goals                   AS goals_against,
    ux.season
FROM understat_xg ux
JOIN teams ht
    ON ux.home_team = ht.name
JOIN teams at
    ON ux.away_team = at.name
JOIN fixtures f
    ON f.team_h = ht.team_id
    AND f.team_a = at.team_id
    AND DATE(f.kickoff_time) = ux.match_date
    AND f.finished = TRUE

UNION ALL

SELECT
    f.fixture_id,
    f.gameweek,
    ux.match_date,
    at.name                         AS team_name,
    ht.name                         AS opponent_name,
    'A'                             AS venue,
    ux.away_xg                      AS team_xg,
    ux.home_xg                      AS opponent_xg,
    ux.away_goals                   AS goals_for,
    ux.home_goals                   AS goals_against,
    ux.season
FROM understat_xg ux
JOIN teams ht
    ON ux.home_team = ht.name
JOIN teams at
    ON ux.away_team = at.name
JOIN fixtures f
    ON f.team_h = ht.team_id
    AND f.team_a = at.team_id
    AND DATE(f.kickoff_time) = ux.match_date
    AND f.finished = TRUE;


-- QUERY 7: Tactical match stats from Understat team history
-- Joins raw post-match tactical rows to fixture ids. These values are not
-- used directly for prediction. They feed later previous-match rolling views.

CREATE OR REPLACE VIEW team_tactical_match_stats AS
SELECT
    txs.fixture_id,
    txs.gameweek,
    uth.match_date,
    uth.team_name,
    txs.opponent_name,
    txs.venue,
    uth.xg,
    uth.xga,
    uth.npxg,
    uth.npxga,
    uth.npxgd,
    uth.ppda,
    uth.ppda_allowed,
    uth.deep,
    uth.deep_allowed,
    uth.scored,
    uth.missed,
    uth.xpts,
    uth.pts
FROM understat_team_history uth
JOIN team_xg_stats txs
    ON uth.team_name = txs.team_name
    AND uth.match_date = txs.match_date
    AND LOWER(uth.venue) = LOWER(txs.venue);


-- QUERY 8: Previous-5 tactical style form
-- Creates leakage-safe rolling style features using only each team's previous
-- 5 matches. Current-match tactical stats are excluded by the window frame.

DROP VIEW IF EXISTS team_style_form;

CREATE OR REPLACE VIEW team_style_form AS
SELECT
    fixture_id,
    gameweek,
    match_date,
    team_name,
    opponent_name,
    venue,
    COUNT(*) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )                               AS style_matches_last5,
    ROUND(AVG(ppda) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS ppda_last5,
    ROUND(AVG(ppda_allowed) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS ppda_allowed_last5,
    ROUND(AVG(deep) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS deep_last5,
    ROUND(AVG(deep_allowed) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS deep_allowed_last5,
    ROUND(AVG(xg) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS xg_last5,
    ROUND(AVG(xga) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS xga_last5,
    ROUND(AVG(npxg) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS npxg_last5,
    ROUND(AVG(npxga) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS npxga_last5,
    ROUND(AVG(npxgd) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS npxgd_last5,
    ROUND(AVG(scored) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS goals_for_last5,
    ROUND(AVG(missed) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS goals_against_last5,
    ROUND(AVG(xpts) OVER (
        PARTITION BY team_name
        ORDER BY match_date, fixture_id
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    )::NUMERIC, 2)                  AS xpts_last5
FROM team_tactical_match_stats;


-- QUERY 9: Home rolling xG form
-- For each home fixture, calculates the team's previous 5-match xG
-- and xGA averages. The current fixture is excluded to prevent leakage.

CREATE OR REPLACE VIEW home_xg_form AS
WITH rolling_xg AS (
    SELECT
        fixture_id,
        team_name,
        venue,
        ROUND(AVG(team_xg) OVER (
            PARTITION BY team_name
            ORDER BY match_date, fixture_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        )::NUMERIC, 2)              AS xg_last5,
        ROUND(AVG(opponent_xg) OVER (
            PARTITION BY team_name
            ORDER BY match_date, fixture_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        )::NUMERIC, 2)              AS xga_last5
    FROM team_xg_stats
)
SELECT
    fixture_id,
    team_name                       AS home_team,
    xg_last5                        AS home_xg_last5,
    xga_last5                       AS home_xga_last5
FROM rolling_xg
WHERE venue = 'H';


-- QUERY 10: Away rolling xG form
-- Same leakage-safe previous-5 rolling logic for each away fixture.

CREATE OR REPLACE VIEW away_xg_form AS
WITH rolling_xg AS (
    SELECT
        fixture_id,
        team_name,
        venue,
        ROUND(AVG(team_xg) OVER (
            PARTITION BY team_name
            ORDER BY match_date, fixture_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        )::NUMERIC, 2)              AS xg_last5,
        ROUND(AVG(opponent_xg) OVER (
            PARTITION BY team_name
            ORDER BY match_date, fixture_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        )::NUMERIC, 2)              AS xga_last5
    FROM team_xg_stats
)
SELECT
    fixture_id,
    team_name                       AS away_team,
    xg_last5                        AS away_xg_last5,
    xga_last5                       AS away_xga_last5
FROM rolling_xg
WHERE venue = 'A';


-- QUERY 11: Master feature table
-- This is the final table your ML model will train on.
-- Joins everything: match results + form + rolling xG/xGA +
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
    h2h.h2h_avg_away_goals,

    -- Leakage-safe rolling xG features from previous matches only
    COALESCE(hxf.home_xg_last5, 0.0)     AS home_xg_last5,
    COALESCE(hxf.home_xga_last5, 0.0)    AS home_xga_last5,
    COALESCE(axf.away_xg_last5, 0.0)     AS away_xg_last5,
    COALESCE(axf.away_xga_last5, 0.0)    AS away_xga_last5,

    -- Leakage-safe style clusters built from previous-match tactical form
    COALESCE(hsc.style_cluster, -1)       AS home_style_cluster,
    COALESCE(ascs.style_cluster, -1)      AS away_style_cluster,
    COALESCE(hsc.style_matches_last5, 0)  AS home_style_matches_last5,
    COALESCE(ascs.style_matches_last5, 0) AS away_style_matches_last5

FROM match_results mr

LEFT JOIN home_form hf
    ON mr.fixture_id = hf.fixture_id

LEFT JOIN away_form af
    ON mr.fixture_id = af.fixture_id

LEFT JOIN home_xg_form hxf
    ON mr.fixture_id = hxf.fixture_id

LEFT JOIN away_xg_form axf
    ON mr.fixture_id = axf.fixture_id

LEFT JOIN teams ht
    ON mr.home_team = ht.team_id

LEFT JOIN teams at
    ON mr.away_team = at.team_id

LEFT JOIN team_style_clusters hsc
    ON mr.fixture_id = hsc.fixture_id
    AND ht.name = hsc.team_name
    AND LOWER(hsc.venue) = 'h'

LEFT JOIN team_style_clusters ascs
    ON mr.fixture_id = ascs.fixture_id
    AND at.name = ascs.team_name
    AND LOWER(ascs.venue) = 'a'

LEFT JOIN h2h_stats h2h
    ON mr.home_team = h2h.home_team
    AND mr.away_team = h2h.away_team;


-- QUERY 12: FPL player value features
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
