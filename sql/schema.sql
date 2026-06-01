CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    first_name VARCHAR(100),
    second_name VARCHAR(100),
    team INTEGER,
    position INTEGER,
    price FLOAT,
    total_points INTEGER,
    minutes INTEGER,
    goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    bonus INTEGER,
    form FLOAT,
    selected_by_percent FLOAT,
    is_available BOOLEAN,
    status VARCHAR(10),
    news TEXT,
    influence FLOAT,
    creativity FLOAT,
    threat FLOAT,
    ict_index FLOAT,
    expected_goals FLOAT,
    expected_assists FLOAT,
    expected_goal_involvements FLOAT,
    expected_goals_conceded FLOAT,
    starts INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    short_name VARCHAR(10),
    strength INTEGER,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    played INTEGER,
    win INTEGER,
    draw INTEGER,
    loss INTEGER,
    points INTEGER
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id INTEGER PRIMARY KEY,
    gameweek INTEGER,
    team_h INTEGER,
    team_a INTEGER,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    finished BOOLEAN,
    kickoff_time TIMESTAMP,
    team_h_score INTEGER,
    team_a_score INTEGER
);

CREATE TABLE IF NOT EXISTS gameweeks (
    gw_id INTEGER PRIMARY KEY,
    name VARCHAR(50),
    deadline_time VARCHAR(50),
    finished BOOLEAN,
    is_current BOOLEAN,
    is_next BOOLEAN,
    is_previous BOOLEAN,
    average_entry_score INTEGER,
    highest_score INTEGER
);
