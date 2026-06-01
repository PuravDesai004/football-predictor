-- Creates a refreshed leakage-safe player gameweek feature table.
-- The current row's total_points is kept only as the prediction target.
-- Historical inputs are built from prior gameweek aggregates, so double
-- gameweek fixtures cannot see outcomes from the same gameweek.

DROP TABLE IF EXISTS player_gameweek_features;

CREATE TABLE player_gameweek_features AS
WITH base AS (
    SELECT
        pgh.player_id,
        pgh.gameweek,
        pgh.fixture,
        pgh.kickoff_time,
        pgh.opponent_team,
        pgh.was_home,
        p.first_name,
        p.second_name,
        p.team AS team_id,
        p.position,
        pgh.value::FLOAT AS value,
        pgh.selected::FLOAT AS selected,
        pgh.total_points::FLOAT AS total_points,
        pgh.minutes::FLOAT AS minutes,
        pgh.goals_scored::FLOAT AS goals_scored,
        pgh.assists::FLOAT AS assists,
        pgh.clean_sheets::FLOAT AS clean_sheets,
        pgh.goals_conceded::FLOAT AS goals_conceded,
        pgh.yellow_cards::FLOAT AS yellow_cards,
        pgh.red_cards::FLOAT AS red_cards,
        pgh.saves::FLOAT AS saves,
        pgh.bonus::FLOAT AS bonus,
        pgh.bps::FLOAT AS bps,
        pgh.influence::FLOAT AS influence,
        pgh.creativity::FLOAT AS creativity,
        pgh.threat::FLOAT AS threat,
        pgh.ict_index::FLOAT AS ict_index,
        pgh.starts::FLOAT AS starts,
        pgh.expected_goals::FLOAT AS expected_goals,
        pgh.expected_assists::FLOAT AS expected_assists,
        pgh.expected_goal_involvements::FLOAT AS expected_goal_involvements,
        pgh.expected_goals_conceded::FLOAT AS expected_goals_conceded,
        pgh.transfers_balance::FLOAT AS transfers_balance,
        pgh.transfers_in::FLOAT AS transfers_in,
        pgh.transfers_out::FLOAT AS transfers_out
    FROM player_gameweek_history pgh
    LEFT JOIN players p
        ON pgh.player_id = p.player_id
),
gw_stats AS (
    SELECT
        player_id,
        gameweek,
        SUM(total_points) AS gw_total_points,
        SUM(minutes) AS gw_minutes,
        SUM(starts) AS gw_starts,
        SUM(expected_goals) AS gw_xg,
        SUM(expected_assists) AS gw_xa,
        SUM(expected_goal_involvements) AS gw_xgi,
        SUM(expected_goals_conceded) AS gw_xgc,
        AVG(influence) AS gw_influence,
        AVG(creativity) AS gw_creativity,
        AVG(threat) AS gw_threat,
        AVG(ict_index) AS gw_ict,
        SUM(bps) AS gw_bps,
        SUM(bonus) AS gw_bonus,
        SUM(goals_scored) AS gw_goals,
        SUM(assists) AS gw_assists,
        SUM(clean_sheets) AS gw_clean_sheets,
        SUM(goals_conceded) AS gw_goals_conceded,
        SUM(saves) AS gw_saves,
        SUM(yellow_cards) AS gw_yellow_cards,
        SUM(red_cards) AS gw_red_cards,
        SUM(transfers_balance) AS gw_transfers_balance,
        SUM(transfers_in) AS gw_transfers_in,
        SUM(transfers_out) AS gw_transfers_out,
        AVG(value) AS gw_value,
        AVG(selected) AS gw_selected
    FROM base
    GROUP BY player_id, gameweek
),
gw_rolling AS (
    SELECT
        player_id,
        gameweek,

        LAG(gw_total_points) OVER player_order AS points_prev1,
        LAG(gw_minutes) OVER player_order AS minutes_prev1,
        LAG(gw_starts) OVER player_order AS starts_prev1,
        LAG(gw_xg) OVER player_order AS xg_prev1,
        LAG(gw_xa) OVER player_order AS xa_prev1,
        LAG(gw_xgi) OVER player_order AS xgi_prev1,
        LAG(gw_xgc) OVER player_order AS xgc_prev1,
        LAG(gw_ict) OVER player_order AS ict_prev1,
        LAG(gw_value) OVER player_order AS value_prev1,
        LAG(gw_selected) OVER player_order AS selected_prev1,

        COUNT(*) OVER player_last5 AS history_matches_last5,
        AVG(gw_total_points) OVER player_last3 AS points_avg_last3,
        AVG(gw_total_points) OVER player_last5 AS points_avg_last5,
        AVG(gw_minutes) OVER player_last3 AS minutes_avg_last3,
        AVG(gw_minutes) OVER player_last5 AS minutes_avg_last5,
        AVG(gw_starts) OVER player_last5 AS starts_avg_last5,
        AVG(gw_xg) OVER player_last5 AS xg_avg_last5,
        AVG(gw_xa) OVER player_last5 AS xa_avg_last5,
        AVG(gw_xgi) OVER player_last5 AS xgi_avg_last5,
        AVG(gw_xgc) OVER player_last5 AS xgc_avg_last5,
        AVG(gw_influence) OVER player_last5 AS influence_avg_last5,
        AVG(gw_creativity) OVER player_last5 AS creativity_avg_last5,
        AVG(gw_threat) OVER player_last5 AS threat_avg_last5,
        AVG(gw_ict) OVER player_last5 AS ict_avg_last5,
        AVG(gw_bps) OVER player_last5 AS bps_avg_last5,
        AVG(gw_bonus) OVER player_last5 AS bonus_avg_last5,
        AVG(gw_goals) OVER player_last5 AS goals_avg_last5,
        AVG(gw_assists) OVER player_last5 AS assists_avg_last5,
        AVG(gw_clean_sheets) OVER player_last5 AS clean_sheets_avg_last5,
        AVG(gw_goals_conceded) OVER player_last5 AS goals_conceded_avg_last5,
        AVG(gw_saves) OVER player_last5 AS saves_avg_last5,
        AVG(gw_yellow_cards) OVER player_last5 AS yellow_cards_avg_last5,
        AVG(gw_red_cards) OVER player_last5 AS red_cards_avg_last5,
        AVG(gw_transfers_balance) OVER player_last5 AS transfers_balance_avg_last5,
        AVG(gw_transfers_in) OVER player_last5 AS transfers_in_avg_last5,
        AVG(gw_transfers_out) OVER player_last5 AS transfers_out_avg_last5
    FROM gw_stats
    WINDOW
        player_order AS (
            PARTITION BY player_id
            ORDER BY gameweek
        ),
        player_last3 AS (
            PARTITION BY player_id
            ORDER BY gameweek
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ),
        player_last5 AS (
            PARTITION BY player_id
            ORDER BY gameweek
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        )
)
SELECT
    base.player_id,
    base.gameweek,
    base.fixture,
    base.kickoff_time,
    base.opponent_team,
    base.was_home,
    base.first_name,
    base.second_name,
    base.team_id,
    base.position,
    base.value,
    base.selected,
    base.total_points AS target_total_points,

    gw_rolling.points_prev1,
    gw_rolling.minutes_prev1,
    gw_rolling.starts_prev1,
    gw_rolling.xg_prev1,
    gw_rolling.xa_prev1,
    gw_rolling.xgi_prev1,
    gw_rolling.xgc_prev1,
    gw_rolling.ict_prev1,
    gw_rolling.value_prev1,
    gw_rolling.selected_prev1,

    COALESCE(gw_rolling.history_matches_last5, 0) AS history_matches_last5,
    gw_rolling.points_avg_last3,
    gw_rolling.points_avg_last5,
    gw_rolling.minutes_avg_last3,
    gw_rolling.minutes_avg_last5,
    gw_rolling.starts_avg_last5,
    gw_rolling.xg_avg_last5,
    gw_rolling.xa_avg_last5,
    gw_rolling.xgi_avg_last5,
    gw_rolling.xgc_avg_last5,
    gw_rolling.influence_avg_last5,
    gw_rolling.creativity_avg_last5,
    gw_rolling.threat_avg_last5,
    gw_rolling.ict_avg_last5,
    gw_rolling.bps_avg_last5,
    gw_rolling.bonus_avg_last5,
    gw_rolling.goals_avg_last5,
    gw_rolling.assists_avg_last5,
    gw_rolling.clean_sheets_avg_last5,
    gw_rolling.goals_conceded_avg_last5,
    gw_rolling.saves_avg_last5,
    gw_rolling.yellow_cards_avg_last5,
    gw_rolling.red_cards_avg_last5,
    gw_rolling.transfers_balance_avg_last5,
    gw_rolling.transfers_in_avg_last5,
    gw_rolling.transfers_out_avg_last5
FROM base
LEFT JOIN gw_rolling
    ON base.player_id = gw_rolling.player_id
    AND base.gameweek = gw_rolling.gameweek;

CREATE INDEX idx_player_gameweek_features_player_id
ON player_gameweek_features (player_id);

CREATE INDEX idx_player_gameweek_features_gameweek
ON player_gameweek_features (gameweek);

CREATE INDEX idx_player_gameweek_features_fixture
ON player_gameweek_features (fixture);

CREATE INDEX idx_player_gameweek_features_kickoff_time
ON player_gameweek_features (kickoff_time);

CREATE INDEX idx_player_gameweek_features_player_gameweek
ON player_gameweek_features (player_id, gameweek);
