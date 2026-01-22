# %% sports betting year long baseball
# %% Thoughts: 
    """
# ============================================================
# PROJECT: Batting Average Projection Model
# GOAL: Estimate probability a player hits >= target AVG in a season
# ============================================================

# ------------------------------------------------------------
# 1. DATABASE STRUCTURE
# ------------------------------------------------------------

# --- HITTING DATA (per season + rolling windows) ---
# Plate Appearances (PA)
# At Bats (AB)
# Hits (H)
# Strikeout Rate (K%)
# Walk Rate (BB%)
# Contact %
# Zone Contact %
# Chase Rate (O-Swing %)
# BABIP
# Pull %, Oppo %, Center %
# Ground Ball %, Fly Ball %, Line Drive %
# Hard-Hit %
# Infield Hit %
# Sprint Speed
# xBA (if available)
# Rolling splits (last 30 / 60 / 90 days)

# --- PITCHING DATA ---
# Pitcher Handedness
# Pitch Mix % (FB, SL, CB, CH, etc.)
# Whiff %
# Zone %
# First Pitch Strike %
# Ground Ball %
# Opponent BABIP
# Average Velocity
# Velocity Year-over-Year Change
# Walk Rate (BB%)
# Strikeout Rate (K%)

# --- DEFENSE DATA (team + position) ---
# Defensive Runs Saved (DRS)
# Outs Above Average (OAA)
# Team Infield Defense Rating
# Team Outfield Defense Rating
# Shift Usage %
# Catcher Framing Runs

# --- BASERUNNING DATA ---
# Sprint Speed
# First-to-Third Rate
# Stolen Base Attempts
# Infield Hit %

# --- WEATHER & PARK DATA ---
# Stadium ID
# Park Factor (Hits)
# Park Factor (BABIP)
# Temperature
# Wind Speed
# Wind Direction
# Altitude
# Roof Open / Closed

# ------------------------------------------------------------
# 2. MATCHUP & CONTEXT FEATURES
# ------------------------------------------------------------

# Hitter vs Pitcher Platoon Advantage
# Hitter Profile Cluster (contact / power / pull / chase)
# Pitcher Profile Cluster (power / soft contact / ground-ball)
# Pitch Type Matchups (hitter vs pitch usage)
# Defense Behind Pitcher
# Park + Weather Adjustment
# Expected Hit Probability per Plate Appearance

# ------------------------------------------------------------
# 3. AGING & REGRESSION
# ------------------------------------------------------------

# Age
# Contact % Aging Curve
# Sprint Speed Aging Curve
# BABIP Aging Curve
# Plate Discipline Aging Curve
# Power Decline (secondary influence)
# Regression to Career Mean
# Regression Weight Based on Sample Size

# ------------------------------------------------------------
# 4. PLAYING TIME & RISK
# ------------------------------------------------------------

# Projected Plate Appearances
# Injury History
# Games Played per Season
# Position Depth on Team
# Platoon Risk
# Minor League / Bench Risk

# ------------------------------------------------------------
# 5. MODELING PIPELINE
# ------------------------------------------------------------

# Predict Probability of Hit per Plate Appearance
# Predict Strikeout Probability
# Predict Ball-in-Play Probability
# Aggregate Outcomes Over Projected PA
# Convert Hits / AB into Batting Average

# ------------------------------------------------------------
# 6. OUTPUT METRICS
# ------------------------------------------------------------

# Projected AVG (Median)
# 10th Percentile AVG
# 90th Percentile AVG
# Probability AVG >= Target
# Risk Flags (K-rate, speed decline, defense, playing time)

# ------------------------------------------------------------
# 7. MODEL TYPES (ITERATIVE DEVELOPMENT)
# ------------------------------------------------------------

# Phase 1: Logistic Regression
# Phase 2: Gradient Boosted Trees (XGBoost / LightGBM)
# Phase 3: Bayesian Hierarchical Model

# ------------------------------------------------------------
# 8. VALIDATION & BENCHMARKING
# ------------------------------------------------------------

# Compare vs Actual Season AVG
# Compare vs Steamer / ZiPS (if available)
# Error Metrics (MAE, RMSE)
# Bias by Player Type
# Rookie vs Veteran Accuracy

    """
# %% create database
import sqlite3

# Path to your database
DB_PATH = r"C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Sports/baseball_project_long_term.db"

# Connect to SQLite (will create the DB if it doesn't exist)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ============================================================
# DATABASE SCHEMA CREATION
# ============================================================

# TEAMS
cursor.execute("""
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT,
    season INTEGER,
    team_name TEXT,
    PRIMARY KEY (team_id, season)
);
""")

# PLAYERS
cursor.execute("""
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    full_name TEXT,
    bats TEXT,
    throws TEXT,
    birth_date DATE
);
""")

# TEAM ROSTERS
cursor.execute("""
CREATE TABLE IF NOT EXISTS team_rosters (
    player_id TEXT,
    team_id TEXT,
    season INTEGER,
    is_pitcher INTEGER,
    PRIMARY KEY (player_id, team_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id, season) REFERENCES teams(team_id, season)
);
""")

# BATTING STATS
cursor.execute("""
CREATE TABLE IF NOT EXISTS batting_stats (
    player_id TEXT,
    team_id TEXT,
    season INTEGER,
    age INTEGER,
    PA INTEGER,
    AB INTEGER,
    H INTEGER,
    BB_pct REAL,
    K_pct REAL,
    contact_pct REAL,
    zone_contact_pct REAL,
    chase_pct REAL,
    BABIP REAL,
    gb_pct REAL,
    fb_pct REAL,
    ld_pct REAL,
    pull_pct REAL,
    oppo_pct REAL,
    center_pct REAL,
    hard_hit_pct REAL,
    infield_hit_pct REAL,
    sprint_speed REAL,
    xBA REAL,
    PRIMARY KEY (player_id, team_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id, season) REFERENCES teams(team_id, season)
);
""")

# PITCHING STATS
cursor.execute("""
CREATE TABLE IF NOT EXISTS pitching_stats (
    player_id TEXT,
    team_id TEXT,
    season INTEGER,
    handedness TEXT,
    fastball_pct REAL,
    slider_pct REAL,
    curveball_pct REAL,
    changeup_pct REAL,
    cutter_pct REAL,
    whiff_pct REAL,
    zone_pct REAL,
    first_pitch_strike_pct REAL,
    ground_ball_pct REAL,
    opponent_BABIP REAL,
    avg_velocity REAL,
    velocity_yoy_change REAL,
    BB_pct REAL,
    K_pct REAL,
    PRIMARY KEY (player_id, team_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id, season) REFERENCES teams(team_id, season)
);
""")

# TEAM DEFENSE
cursor.execute("""
CREATE TABLE IF NOT EXISTS team_defense (
    team_id TEXT,
    season INTEGER,
    DRS REAL,
    OAA REAL,
    infield_defense REAL,
    outfield_defense REAL,
    shift_usage_pct REAL,
    catcher_framing_runs REAL,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id, season) REFERENCES teams(team_id, season)
);
""")

# BASERUNNING STATS
cursor.execute("""
CREATE TABLE IF NOT EXISTS baserunning_stats (
    player_id TEXT,
    team_id TEXT,
    season INTEGER,
    sprint_speed REAL,
    first_to_third_pct REAL,
    stolen_base_attempts INTEGER,
    infield_hit_pct REAL,
    PRIMARY KEY (player_id, team_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id, season) REFERENCES teams(team_id, season)
);
""")

# PARK & WEATHER
cursor.execute("""
CREATE TABLE IF NOT EXISTS park_weather (
    team_id TEXT,
    season INTEGER,
    park_factor_hits REAL,
    park_factor_babip REAL,
    avg_temperature REAL,
    avg_wind_speed REAL,
    avg_wind_direction REAL,
    altitude REAL,
    roof_type TEXT,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id, season) REFERENCES teams(team_id, season)
);
""")

# AGING FACTORS
cursor.execute("""
CREATE TABLE IF NOT EXISTS aging_factors (
    player_id TEXT,
    season INTEGER,
    age INTEGER,
    contact_age_adj REAL,
    sprint_age_adj REAL,
    babip_age_adj REAL,
    discipline_age_adj REAL,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
""")

# Commit changes and close connection
conn.commit()

print("Database and tables created successfully!")
# %% uniqe table names: 
    [teams
    players
    team_rosters
    batting_stats
    pitching_stats
    team_defense
    baserunning_stats
    park_weather
    aging_factors]
#query to get all table names
cursor.execute("Select name from sqlite_master where type='table';")
tables = cursor.fetchall()

#print uniqe table names
print("tables in database:")
for table in tables: 
    print(table[0])

#close connection
conn.close()
# %% playerid_lookup
from pybaseball import playerid_lookup

# Example: find all players named "Mike Trout"
players = playerid_lookup('Trout', 'Mike')
print(players)

# %% current mlb players
from pybaseball import statcast
import pandas as pd

# Pull all Statcast events from last season
df = statcast('2025-03-01', '2025-11-01')

# Extract unique player IDs
player_ids = df['player_id'].unique()

print(f"Number of unique batters: {len(player_ids)}")
print(player_ids[:20])  # print first 20 as example

# %% upload to database

from pybaseball import statcast_batter
import sqlite3
import pandas as pd

# Example: scrape last 7 days for a player
df = statcast_batter('2026-01-01', '2026-01-07', player_id=592450)  # Mike Trout example

# Clean / select relevant columns
df = df[['player_id','PA','AB','H','BB_pct','K_pct','xBA','max_speed']]  # simplify
df.rename(columns={'max_speed':'sprint_speed'}, inplace=True)

# Upload to SQLite
conn = sqlite3.connect(DB_PATH)
df.to_sql('batting_stats', conn, if_exists='append', index=False)
conn.close()
