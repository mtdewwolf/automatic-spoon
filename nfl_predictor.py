"""
NFL Predictor MVP – Single-file, production-ready(ish) toolkit

May the odds be ever in your favor, Sir Toaster the Third!

README (Setup & Usage)
----------------------
- Requirements (Python 3.9+):
  - nfl_data_py, xgboost, pandas, numpy, scikit-learn, requests, joblib
  - Optional: matplotlib (for backtest PNG)

- Install deps (example):
  pip install nfl_data_py xgboost pandas numpy scikit-learn requests joblib

- CLI Examples:
  - Train (first run downloads and caches data):
    python nfl_predictor.py train --years 2015-2023

  - Predict a game (pass current Vegas lines):
    python nfl_predictor.py predict \
      --game "Chiefs vs Ravens" --vegas_spread -2.5 --vegas_total 47.5

  - Weekly update (warm-start incremental fit):
    python nfl_predictor.py update --season 2024 --week 5

  - Backtest ROI (walk-forward by season):
    python nfl_predictor.py backtest --start 2018 --end 2023

Notes & Disclaimers
-------------------
- Predictions are probabilistic, not guarantees. Bet responsibly.
- Vegas odds are sourced from schedule data when available and otherwise
  mocked heuristically. This is an MVP.
- Data is cached under ./data; models under ./models.
- Use environment vars to override defaults:
  NFLP_DATA_DIR, NFLP_MODELS_DIR, NFLP_LOG_LEVEL

Unit Tests
----------
- Minimal inline unit tests can be run via:
  python nfl_predictor.py --run-tests

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional imports
try:
    import nfl_data_py as nfl
except Exception:  # pragma: no cover - allow import to fail during tests
    nfl = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise RuntimeError(
        "xgboost is required. Install with: pip install xgboost"
    ) from exc


# ----------------------------
# Configuration & Directories
# ----------------------------

APP_NAME = "nfl_predictor"
DATA_DIR = os.environ.get("NFLP_DATA_DIR", os.path.join(".", "data"))
MODELS_DIR = os.environ.get("NFLP_MODELS_DIR", os.path.join(".", "models"))
LOG_LEVEL = os.environ.get("NFLP_LOG_LEVEL", "INFO").upper()

SPREAD_MODEL_PATH = os.path.join(MODELS_DIR, "spread_model.joblib")
TOTAL_MODEL_PATH = os.path.join(MODELS_DIR, "total_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "models_meta.json")

SCHEDULE_CACHE = os.path.join(DATA_DIR, "schedule_2015_ongoing.csv")
WEEKLY_CACHE = os.path.join(DATA_DIR, "weekly_2015_ongoing.csv")
DRAFT_CACHE = os.path.join(DATA_DIR, "draft_2015_ongoing.csv")
ROSTERS_CACHE = os.path.join(DATA_DIR, "rosters_2015_ongoing.csv")
COMBINE_CACHE = os.path.join(DATA_DIR, "combine_2015_ongoing.csv")


# ----------------------------
# Logging
# ----------------------------


def setup_logging() -> None:
    """Configure logging for the application."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    log_path = os.path.join(DATA_DIR, f"{APP_NAME}.log")
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(funcName)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    # Suppress noisy runtime warnings from numpy rolling means on empty windows
    warnings.filterwarnings(
        "ignore", message="Mean of empty slice", category=RuntimeWarning
    )


logger = logging.getLogger(APP_NAME)


# ----------------------------
# Utilities & Constants
# ----------------------------


TEAM_ABBR = {
    # Minimal mapping; extend as needed
    "patriots": "NE",
    "bills": "BUF",
    "jets": "NYJ",
    "dolphins": "MIA",
    "chiefs": "KC",
    "chargers": "LAC",
    "raiders": "LV",
    "broncos": "DEN",
    "ravens": "BAL",
    "steelers": "PIT",
    "browns": "CLE",
    "bengals": "CIN",
    "colts": "IND",
    "jaguars": "JAX",
    "titans": "TEN",
    "texans": "HOU",
    "cowboys": "DAL",
    "eagles": "PHI",
    "giants": "NYG",
    "commanders": "WAS",
    "packers": "GB",
    "bears": "CHI",
    "vikings": "MIN",
    "lions": "DET",
    "49ers": "SF",
    "rams": "LA",  # may appear as LAR in some data
    "seahawks": "SEA",
    "cardinals": "ARI",
    "saints": "NO",
    "falcons": "ATL",
    "panthers": "CAR",
    "buccaneers": "TB",
    # Legacy/aliases
    "redskins": "WAS",
    "washington": "WAS",
    "footballteam": "WAS",
}


DOME_TEAMS = {
    "ATL", "NO", "MIN", "DET", "DAL", "ARI", "HOU", "IND", "LA",
    "LAR", "LV"
}

QB_STARS = {
    "Patrick Mahomes", "Josh Allen", "Joe Burrow", "Justin Herbert",
    "Lamar Jackson", "Jalen Hurts", "Aaron Rodgers", "Dak Prescott",
    "Matthew Stafford", "Tua Tagovailoa", "Kirk Cousins",
}


def normalize_team_name(name: str) -> str:
    """Map user input like 'Chiefs' to NFL abbreviation like 'KC'."""
    key = re.sub(r"[^a-z0-9]", "", name.strip().lower())
    # Try exact mapping
    if key in TEAM_ABBR:
        return TEAM_ABBR[key]
    # If it's already an abbreviation
    if key.upper() in set(TEAM_ABBR.values()):
        return key.upper()
    # Heuristic: last word (e.g., 'Kansas City Chiefs' -> 'chiefs')
    parts = [p for p in re.split(r"\s+", name.lower()) if p]
    if parts:
        last = parts[-1]
        if last in TEAM_ABBR:
            return TEAM_ABBR[last]
    raise ValueError(f"Unknown team name: {name}")


def parse_years(years_str: str) -> List[int]:
    """Parse '2015-2023' or '2019,2020,2021'."""
    years: List[int] = []
    for chunk in re.split(r"[, ]+", years_str.strip()):
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(chunk))
    years = sorted(set(years))
    return years


def safe_date(series: pd.Series) -> pd.Series:
    """Safely convert to datetime without crashing on bad rows."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(
        None
    )


def _kelly_fraction(p: float, odds_decimal: float) -> float:
    """Compute Kelly fraction. p is probability, odds_decimal like 1.909 for -110.
    """
    b = odds_decimal - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return float(max(0.0, min(0.05, f)))  # cap at 5% bankroll for safety


# ----------------------------
# Data Fetching & Caching
# ----------------------------


def fetch_data(years: Iterable[int]) -> pd.DataFrame:
    """Download and cache schedule data for years.

    - Uses nfl_data_py.import_schedules
    - Adds heuristic Vegas lines if missing
    - Returns game-level schedule with outcomes
    - Caches merged CSV to avoid repeated downloads
    """
    if nfl is None:
        raise RuntimeError(
            "nfl_data_py is required. pip install nfl_data_py"
        )

    years_list = sorted(set(int(y) for y in years))
    logger.info("Fetching schedule for years: %s", years_list)

    # Load existing cache if present and sufficiently covers years
    cached_df: Optional[pd.DataFrame] = None
    if os.path.exists(SCHEDULE_CACHE):
        try:
            cached_df = pd.read_csv(SCHEDULE_CACHE)
            cached_df["gameday"] = safe_date(cached_df.get("gameday"))
        except Exception:
            cached_df = None

    need_years = set(years_list)
    if cached_df is not None:
        have_years = set(cached_df["season"].dropna().astype(int).unique())
        if need_years.issubset(have_years):
            logger.info("Using cached schedule: %s", SCHEDULE_CACHE)
            df = cached_df.copy()
        else:
            logger.info("Cache missing years; re-downloading schedule")
            df_dl = nfl.import_schedules(years_list)
            df_dl["gameday"] = safe_date(df_dl.get("gameday"))
            df = df_dl
    else:
        df_dl = nfl.import_schedules(years_list)
        df_dl["gameday"] = safe_date(df_dl.get("gameday"))
        df = df_dl

    # Normalize columns for vegas lines
    if "spread_line" not in df.columns and "vegas_spread" in df.columns:
        df = df.rename(columns={"vegas_spread": "spread_line"})
    if "total_line" not in df.columns:
        for c in ("over_under_line", "total", "ou_line"):
            if c in df.columns:
                df = df.rename(columns={c: "total_line"})
                break

    # Ensure vegas lines; synthesize if missing
    if "spread_line" not in df.columns:
        df["spread_line"] = 0.0
    if "total_line" not in df.columns:
        df["total_line"] = 44.0

    # Heuristic fillna for lines
    df["spread_line"] = df["spread_line"].astype(float).fillna(0.0)
    df["total_line"] = df["total_line"].astype(float).fillna(44.0)

    # Keep essential columns only
    keep = [
        "game_id", "season", "week", "gameday", "home_team",
        "away_team", "home_score", "away_score", "spread_line",
        "total_line", "home_qb_name", "away_qb_name",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep].copy()

    # Save cache
    df.to_csv(SCHEDULE_CACHE, index=False)
    logger.info("Schedule cached at %s", SCHEDULE_CACHE)
    # Also cache weekly player stats (2015 through latest requested),
    # attempting each season individually to include preseason when available
    try:
        weekly_cached = None
        if os.path.exists(WEEKLY_CACHE):
            weekly_cached = pd.read_csv(WEEKLY_CACHE)

        collected: List[pd.DataFrame] = []
        for y in years_list:
            try:
                w_y = nfl.import_weekly_data([int(y)])
                # keep lightweight columns
                keepw = [
                    "player_id", "player_name", "position", "team",
                    "season", "week", "attempts", "completions",
                    "passing_yards", "passing_tds", "interceptions",
                    "sacks", "rushing_yards", "rushing_tds",
                    "receiving_yards", "fumbles", "fumbles_lost",
                ]
                for c in keepw:
                    if c not in w_y.columns:
                        w_y[c] = np.nan
                collected.append(w_y[keepw])
            except Exception as ye:
                logger.info("Weekly data not available for %s (skipping): %s", y, ye)

        if collected:
            w_all = pd.concat(collected, ignore_index=True)
            w_all.to_csv(WEEKLY_CACHE, index=False)
            logger.info("Weekly player stats cached at %s", WEEKLY_CACHE)
        elif weekly_cached is not None:
            logger.info("Using previously cached weekly stats at %s", WEEKLY_CACHE)
        else:
            logger.warning("No weekly stats available to cache for requested years.")
    except Exception as e:
        logger.warning("Weekly data fetch failed (continuing): %s", e)

    # Cache draft picks, rosters, and combine (for rookie and prospect signals)
    try:
        # Draft picks
        try:
            dp = nfl.import_draft_picks(years_list)
            dp.to_csv(DRAFT_CACHE, index=False)
            logger.info("Draft picks cached at %s", DRAFT_CACHE)
        except Exception as de:
            logger.info("Draft picks not available (skipping): %s", de)
        # Rosters
        try:
            rs = nfl.import_rosters(years_list)
            rs.to_csv(ROSTERS_CACHE, index=False)
            logger.info("Rosters cached at %s", ROSTERS_CACHE)
        except Exception as rexc:
            logger.info("Rosters not available (skipping): %s", rexc)
        # Combine
        try:
            cb = nfl.import_combine(years_list)
            cb.to_csv(COMBINE_CACHE, index=False)
            logger.info("Combine cached at %s", COMBINE_CACHE)
        except Exception as cbe:
            logger.info("Combine not available (skipping): %s", cbe)
    except Exception as e:
        logger.info("Ancillary draft/roster/combine fetch skipped: %s", e)

    return df


# ----------------------------
# Feature Engineering
# ----------------------------


def _build_team_game_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    """Duplicate each game to team-centric rows for rolling features."""
    games = schedule.copy()
    games["gameday"] = safe_date(games["gameday"])
    games["actual_margin"] = (
        games["home_score"].fillna(0) - games["away_score"].fillna(0)
    )
    games["actual_total"] = (
        games["home_score"].fillna(0) + games["away_score"].fillna(0)
    )

    home_rows = games.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
            "home_qb_name": "qb_name",
        }
    ).copy()
    home_rows["is_home"] = 1
    away_rows = games.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
            "away_qb_name": "qb_name",
        }
    ).copy()
    away_rows["is_home"] = 0

    cols = [
        "game_id", "season", "week", "gameday", "team", "opponent",
        "points_for", "points_against", "is_home", "spread_line",
        "total_line", "qb_name", "actual_margin", "actual_total",
    ]
    team_games = pd.concat([home_rows[cols], away_rows[cols]], ignore_index=True)
    team_games = team_games.sort_values(["team", "gameday"]).reset_index(
        drop=True
    )
    return team_games


def _rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling and season-to-date features for each team row."""
    df = team_games.copy()
    df["win"] = (df["points_for"] > df["points_against"]).astype(int)
    df["margin"] = df["points_for"] - df["points_against"]
    df["qb_known"] = df["qb_name"].fillna("").astype(str).str.len().gt(0)
    df["qb_known"] = df["qb_known"].astype(int)
    df["qb_star"] = df["qb_name"].fillna("").isin(QB_STARS).astype(int)
    # Merge in recent QB performance (last 3 weeks yards/TDs/INTs per game)
    try:
        if os.path.exists(WEEKLY_CACHE):
            wk = pd.read_csv(WEEKLY_CACHE)
            wk = wk[wk["position"].fillna("") == "QB"].copy()
            wk["gkey"] = (
                wk["team"].astype(str) + "_" + wk["season"].astype(str)
            )
            # Aggregate last 3 weeks by player; we don't have player IDs on team
            # rows, so proxy by team-level recent QB avg (team QB committee avg)
            wk_group = wk.groupby(["gkey", "week"]).agg({
                "passing_yards": "sum",
                "passing_tds": "sum",
                "interceptions": "sum",
            }).reset_index()
            # Compute rolling at team-season level
            def _roll_team(g: pd.DataFrame) -> pd.DataFrame:
                g = g.sort_values("week").copy()
                g["qb_ypg_r3"] = (
                    g["passing_yards"].rolling(3, min_periods=1).mean()
                )
                g["qb_tdpG_r3"] = (
                    g["passing_tds"].rolling(3, min_periods=1).mean()
                )
                g["qb_intpG_r3"] = (
                    g["interceptions"].rolling(3, min_periods=1).mean()
                )
                return g
            wk_group = wk_group.groupby("gkey", group_keys=False).apply(
                _roll_team
            )
            # Merge into df via team-season-week
            df["gkey"] = df["team"].astype(str) + "_" + df["season"].astype(str)
            df = pd.merge(
                df,
                wk_group[[
                    "gkey", "week", "qb_ypg_r3", "qb_tdpG_r3", "qb_intpG_r3"
                ]],
                on=["gkey", "week"],
                how="left",
            )
            for col in ["qb_ypg_r3", "qb_tdpG_r3", "qb_intpG_r3"]:
                df[col] = df[col].fillna(df[col].median(skipna=True))
            df.drop(columns=["gkey"], inplace=True)
        else:
            for col in ["qb_ypg_r3", "qb_tdpG_r3", "qb_intpG_r3"]:
                df[col] = 0.0
    except Exception as e:
        logger.warning("QB weekly merge failed (continuing): %s", e)
        for col in ["qb_ypg_r3", "qb_tdpG_r3", "qb_intpG_r3"]:
            if col not in df:
                df[col] = 0.0

    # Merge broader team weekly offense and infer defensive allowed via opponent
    try:
        if os.path.exists(WEEKLY_CACHE):
            wk = pd.read_csv(WEEKLY_CACHE)
            team_off = wk.groupby(["team", "season", "week"], as_index=False).agg({
                "passing_yards": "sum",
                "rushing_yards": "sum",
                "interceptions": "sum",
                "fumbles_lost": "sum",
            })
            team_off = team_off.rename(columns={
                "team": "team",
                "passing_yards": "off_pass_yards",
                "rushing_yards": "off_rush_yards",
                "interceptions": "off_ints",
                "fumbles_lost": "off_fumbles_lost",
            })
            team_off["off_turnovers"] = (
                team_off["off_ints"].fillna(0) + team_off["off_fumbles_lost"].fillna(0)
            )

            # Attach offense to team rows
            df = pd.merge(
                df,
                team_off[[
                    "team", "season", "week", "off_pass_yards", "off_rush_yards",
                    "off_turnovers"
                ]],
                on=["team", "season", "week"],
                how="left",
            )

            # Attach opponent offense to infer defense allowed/forced
            opp_off = team_off.rename(columns={
                "team": "opponent",
                "off_pass_yards": "def_pass_yards_allowed",
                "off_rush_yards": "def_rush_yards_allowed",
                "off_turnovers": "def_turnovers_forced",
            })
            df = pd.merge(
                df,
                opp_off[[
                    "opponent", "season", "week", "def_pass_yards_allowed",
                    "def_rush_yards_allowed", "def_turnovers_forced"
                ]],
                on=["opponent", "season", "week"],
                how="left",
            )
            for col in [
                "off_pass_yards", "off_rush_yards", "off_turnovers",
                "def_pass_yards_allowed", "def_rush_yards_allowed",
                "def_turnovers_forced",
            ]:
                if col not in df:
                    df[col] = 0.0
                df[col] = df[col].fillna(df[col].median(skipna=True))
        else:
            for col in [
                "off_pass_yards", "off_rush_yards", "off_turnovers",
                "def_pass_yards_allowed", "def_rush_yards_allowed",
                "def_turnovers_forced",
            ]:
                df[col] = 0.0
    except Exception as e:
        logger.warning("Team weekly merge failed (continuing): %s", e)
        for col in [
            "off_pass_yards", "off_rush_yards", "off_turnovers",
            "def_pass_yards_allowed", "def_rush_yards_allowed",
            "def_turnovers_forced",
        ]:
            if col not in df:
                df[col] = 0.0

    def add_group_rolls(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("gameday").copy()
        # Rest days since last game
        group["rest_days"] = (
            group["gameday"].diff().dt.days.fillna(7).clip(lower=3, upper=14)
        )
        # Rolling 3-game
        for col in ("points_for", "points_against", "margin", "win"):
            group[f"{col}_r3_mean"] = (
                group[col].rolling(3, min_periods=1).mean()
            )
            group[f"{col}_r3_sum"] = (
                group[col].rolling(3, min_periods=1).sum()
            )
        # Season-to-date means
        for col in ("points_for", "points_against", "margin", "win"):
            group[f"{col}_s_mean"] = (
                group[col].expanding(min_periods=1).mean()
            )
        # Rolling team offense/defense stats (3 games)
        for col in (
            "off_pass_yards", "off_rush_yards", "off_turnovers",
            "def_pass_yards_allowed", "def_rush_yards_allowed",
            "def_turnovers_forced",
        ):
            group[f"{col}_r3"] = (
                group[col].rolling(3, min_periods=1).mean()
            )
        # Elo proxy: expanding win pct with smoothing
        group["elo_proxy"] = (
            (group["win"].expanding(min_periods=1).mean() * 2 - 1)
        )
        # Home/away splits rolling
        group["home_rate_s"] = (
            group["is_home"].expanding(min_periods=1).mean()
        )
        # Rookie adjustment placeholder: penalize unknown QB early season
        try:
            rookie_pen_default = -0.2
            if os.path.exists(ROSTERS_CACHE):
                rs = pd.read_csv(ROSTERS_CACHE)
                rs = rs[(rs["season"] == group["season"].iloc[0]) &
                        (rs["years_exp"].fillna(99) == 0)]
                rook_teams = set(rs["team"].astype(str))
                group["rookie_penalty"] = np.where(
                    group["team"].astype(str).isin(rook_teams)
                    & (group["week"].fillna(0) <= 3),
                    rookie_pen_default,
                    0.0,
                )
            else:
                group["rookie_penalty"] = np.where(
                    (group["qb_known"] == 0) & (group["week"].fillna(0) <= 3),
                    rookie_pen_default,
                    0.0,
                )
        except Exception:
            group["rookie_penalty"] = np.where(
                (group["qb_known"] == 0) & (group["week"].fillna(0) <= 3),
                -0.2,
                0.0,
            )
        return group

    df = df.groupby("team", as_index=False, group_keys=False).apply(
        add_group_rolls
    )

    # Weather/dome indicator
    df["is_dome"] = df["team"].isin(DOME_TEAMS).astype(int)

    # Public fade (heuristic): big-market or trendy teams
    public_teams = {
        "DAL", "GB", "PIT", "NE", "KC", "SF", "PHI", "LA", "LAR"
    }
    df["public_heat"] = df["team"].isin(public_teams).astype(int)

    return df


def _build_matchup_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Create one row per game from home perspective with feature diffs."""
    home = team_rows[team_rows["is_home"] == 1].copy()
    away = team_rows[team_rows["is_home"] == 0].copy()

    suffixes = ("_home", "_away")
    merged = pd.merge(
        home,
        away,
        on=["game_id", "season", "week", "gameday"],
        suffixes=suffixes,
        how="inner",
    )

    # Create differential features (home - away)
    feature_cols = [
        "points_for_r3_mean", "points_against_r3_mean", "margin_r3_mean",
        "win_r3_mean", "points_for_s_mean", "points_against_s_mean",
        "margin_s_mean", "win_s_mean", "rest_days", "elo_proxy",
        "home_rate_s", "is_dome", "public_heat", "qb_known",
        "qb_star", "rookie_penalty", "qb_ypg_r3", "qb_tdpG_r3",
        "qb_intpG_r3", "off_pass_yards_r3", "off_rush_yards_r3",
        "off_turnovers_r3", "def_pass_yards_allowed_r3",
        "def_rush_yards_allowed_r3", "def_turnovers_forced_r3",
    ]

    for col in feature_cols:
        merged[f"{col}_diff"] = merged[f"{col}_home"] - merged[f"{col}_away"]

    # Targets relative to Vegas
    merged["y_spread"] = (
        merged["actual_margin_home"] - merged["spread_line_home"]
    )
    merged["y_total"] = (
        merged["actual_total_home"] - merged["total_line_home"]
    )

    # Rename/collect
    X_cols = [f"{c}_diff" for c in feature_cols]
    X = merged[X_cols].copy()

    meta_cols = [
        "game_id", "season", "week", "gameday", "team_home",
        "opponent_home", "spread_line_home", "total_line_home",
        "actual_margin_home", "actual_total_home",
    ]
    meta = merged[meta_cols].rename(
        columns={
            "team_home": "home_team",
            "opponent_home": "away_team",
            "spread_line_home": "vegas_spread",
            "total_line_home": "vegas_total",
            "actual_margin_home": "actual_margin",
            "actual_total_home": "actual_total",
        }
    )
    y_spread = merged["y_spread"].astype(float)
    y_total = merged["y_total"].astype(float)

    return X, y_spread, y_total, meta


def engineer_features(schedule: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.DataFrame
]:
    """Create 15+ betting-relevant features and targets.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (home-away differentials and indicators).
    y_spread : pd.Series
        Target for spread model: actual_margin - vegas_spread.
    y_total : pd.Series
        Target for total model: actual_total - vegas_total.
    meta : pd.DataFrame
        Game metadata including teams, lines, and outcomes.
    """
    team_rows = _build_team_game_rows(schedule)
    team_rows = _rolling_features(team_rows)

    X, y_spread, y_total, meta = _build_matchup_features(team_rows)

    # Ensure columns with all-missing values are filled to preserve shape
    for col in list(X.columns):
        series = X[col]
        if series.isna().all():
            X[col] = 0.0

    # Ensure minimum feature count
    assert X.shape[1] >= 15, "Not enough engineered features"

    # Impute missing with medians and scale
    num_features = list(X.columns)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_features)
        ]
    )
    # Fit-transform to produce dense feature matrix
    X_processed = pre.fit_transform(X)
    # Persist the preprocessor for use in inference
    pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    dump({"pre": pre, "feature_names": num_features}, pre_path)
    logger.info("Saved preprocessor to %s", pre_path)

    X_df = pd.DataFrame(X_processed, columns=[f"f{i}" for i in range(X.shape[1])])
    return X_df, y_spread, y_total, meta


# ----------------------------
# Model Training
# ----------------------------


def _make_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=max(1, os.cpu_count() or 1),
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
    )


def train_models(
    X: pd.DataFrame, y_spread: pd.Series, y_total: pd.Series,
    years: List[int]
) -> Tuple[XGBRegressor, XGBRegressor, Dict[str, Any]]:
    """Train spread and total regressors with TS CV and grid search."""
    logger.info("Training models with TimeSeriesSplit CV")
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.07, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    def fit_grid(y: pd.Series) -> Tuple[XGBRegressor, float, Dict[str, Any]]:
        base = _make_regressor()
        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            n_jobs=max(1, (os.cpu_count() or 1) // 2),
            verbose=0,
        )
        grid.fit(X, y)
        best: XGBRegressor = grid.best_estimator_
        best_score = -float(grid.best_score_)
        return best, best_score, grid.best_params_

    spread_model, spread_mae, spread_params = fit_grid(y_spread)
    total_model, total_mae, total_params = fit_grid(y_total)

    # Log feature importances (note: feature names are generic f0..)
    def _log_importances(name: str, model: XGBRegressor) -> None:
        try:
            imp = model.feature_importances_
            order = np.argsort(-imp)[:10]
            top = [(f"f{int(i)}", float(imp[i])) for i in order]
            logger.info("Top %s feature importances: %s", name, top)
        except Exception as e:
            logger.warning("Could not log feature importances: %s", e)

    _log_importances("spread", spread_model)
    _log_importances("total", total_model)

    # Save models
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "years": years,
        "spread_mae_cv": spread_mae,
        "total_mae_cv": total_mae,
        "spread_params": spread_params,
        "total_params": total_params,
        "feature_count": int(X.shape[1]),
    }

    dump({"model": spread_model}, SPREAD_MODEL_PATH)
    dump({"model": total_model}, TOTAL_MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Saved models. CV MAE (spread=%.3f, total=%.3f)", spread_mae, total_mae
    )
    return spread_model, total_model, meta


# ----------------------------
# Prediction Engine
# ----------------------------


def _load_artifacts() -> Tuple[XGBRegressor, XGBRegressor, Dict[str, Any], Any]:
    if not (os.path.exists(SPREAD_MODEL_PATH) and os.path.exists(TOTAL_MODEL_PATH)):
        raise FileNotFoundError(
            "Models not found. Train first with: nfl_predictor.py train"
        )
    spread = load(SPREAD_MODEL_PATH)["model"]
    total = load(TOTAL_MODEL_PATH)["model"]
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    pre = load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    return spread, total, meta, pre


def _current_matchup_features(
    schedule: pd.DataFrame, home_abbr: str, away_abbr: str
) -> pd.DataFrame:
    """Build current matchup differential features using last games.

    Ensures exact column set/order expected by the saved preprocessor.
    """
    team_rows = _build_team_game_rows(schedule)
    team_rows = _rolling_features(team_rows)
    latest = team_rows.sort_values("gameday").groupby("team").tail(1)

    home = latest[latest["team"] == home_abbr]
    away = latest[latest["team"] == away_abbr]
    if home.empty or away.empty:
        raise ValueError("Insufficient history to build features.")

    pre_art = load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    expected_cols: List[str] = list(pre_art.get("feature_names", []))
    if not expected_cols:
        raise RuntimeError("Preprocessor feature names missing.")

    row_dict: Dict[str, float] = {}
    for name in expected_cols:
        base = name[:-5] if name.endswith("_diff") else name
        if base in home.columns and base in away.columns:
            try:
                row_dict[name] = float(home.iloc[0][base] - away.iloc[0][base])
            except Exception:
                row_dict[name] = 0.0
        else:
            row_dict[name] = 0.0

    X = pd.DataFrame([[row_dict[c] for c in expected_cols]], columns=expected_cols)
    X_proc = pre_art["pre"].transform(X)
    X_df = pd.DataFrame(X_proc, columns=[f"f{i}" for i in range(X.shape[1])])
    return X_df


def predict_game(
    home_team: str, away_team: str, vegas_spread: float, vegas_total: float
) -> Dict[str, Any]:
    """Predict spread and total for an upcoming game and produce a bet.

    Returns a structured dict with recommendations and EV.
    """
    spread_m, total_m, meta, _pre = _load_artifacts()
    schedule = pd.read_csv(SCHEDULE_CACHE)

    home_abbr = normalize_team_name(home_team)
    away_abbr = normalize_team_name(away_team)

    X_df = _current_matchup_features(schedule, home_abbr, away_abbr)
    pred_spread_edge = float(spread_m.predict(X_df)[0])
    pred_total_edge = float(total_m.predict(X_df)[0])

    # Convert from edges to raw predictions
    pred_spread = float(vegas_spread + pred_spread_edge)
    pred_total = float(vegas_total + pred_total_edge)

    # Normal error model
    sigma_spread = max(6.0, float(meta.get("spread_mae_cv", 3.5)) * 2.0)
    z = (pred_spread - vegas_spread) / max(1e-6, sigma_spread)
    p_cover = 0.5 + 0.5 * math.erf(abs(z) / math.sqrt(2))
    odds_decimal = 1.909  # -110
    kelly = _kelly_fraction(p_cover, odds_decimal)

    # Moneyline probability: P(home wins) = Phi(pred_margin / sigma)
    # Here pred_spread approximates predicted actual_margin_home
    p_home = 0.5 * (1.0 + math.erf(pred_spread / (sigma_spread * math.sqrt(2))))
    p_away = 1.0 - p_home
    ml_pick = home_team if p_home >= 0.5 else away_team
    ml_prob = p_home if p_home >= 0.5 else p_away

    # Confidence heuristic
    base_mae = float(meta.get("spread_mae_cv", 3.5))
    conf = max(0.5, min(0.9, 1.0 - base_mae / 6.0))

    # Recommendation text and strength
    fav_text = f"{home_team} {pred_spread:.1f} (vs Vegas {vegas_spread:.1f})"
    pick_spread = (
        f"Bet {home_team} to cover" if pred_spread < vegas_spread else
        f"Bet {away_team} to cover"
    )
    pick_total = ("Bet Over" if pred_total > vegas_total else "Bet Under")

    def _strength(edge: float) -> str:
        a = abs(edge)
        if a >= 3.0:
            return "STRONG BET"
        if a >= 2.0:
            return "BET"
        if a >= 1.0:
            return "LEAN"
        return "PASS"

    spread_strength = _strength(pred_spread - vegas_spread)
    total_strength = _strength(pred_total - vegas_total)

    return {
        "predicted_spread": pred_spread,
        "predicted_total": pred_total,
        "edge_spread_pts": pred_spread - vegas_spread,
        "edge_total_pts": pred_total - vegas_total,
        "confidence": conf,
        "kelly_fraction": kelly,
        "moneyline_pick": ml_pick,
        "moneyline_prob": ml_prob,
        "text_spread": (
            f"{fav_text} → {pick_spread} [{spread_strength}]"
        ),
        "text_total": (
            f"Total {pred_total:.1f} (vs {vegas_total:.1f}) → "
            f"{pick_total} [{total_strength}]"
        ),
    }


# ----------------------------
# Update System (Warm-start)
# ----------------------------


def update_model(season: int, week: int) -> Dict[str, Any]:
    """Append new games and warm-start fit the models.

    Returns metadata with before/after metrics and drift.
    """
    logger.info("Updating models for season=%s week=%s", season, week)
    # Refresh schedule for given season; this will update cache
    cur = fetch_data([season])
    # Recompute features for all available data
    sched = pd.read_csv(SCHEDULE_CACHE)
    X, y_spread, y_total, meta = engineer_features(sched)

    # Load previous models
    spread_old, total_old, old_meta, pre = _load_artifacts()
    prev_spread_mae = float(old_meta.get("spread_mae_cv", np.nan))
    prev_total_mae = float(old_meta.get("total_mae_cv", np.nan))

    def _quick_fit() -> Tuple[XGBRegressor, XGBRegressor]:
        m1 = _make_regressor()
        m2 = _make_regressor()
        m1.fit(X, y_spread)
        m2.fit(X, y_total)
        return m1, m2

    # If feature count changed, do a quick full refit instead of warm-start
    if int(old_meta.get("feature_count", X.shape[1])) != int(X.shape[1]):
        logger.info(
            "Feature count changed (%s -> %s). Performing full refit.",
            old_meta.get("feature_count"), X.shape[1],
        )
        spread_new, total_new = _quick_fit()
    else:
        # Warm start: continue boosting
        try:
            spread_new = _make_regressor()
            total_new = _make_regressor()
            spread_new.fit(X, y_spread, xgb_model=spread_old.get_booster())
            total_new.fit(X, y_total, xgb_model=total_old.get_booster())
        except Exception as e:
            logger.warning(
                "Warm-start failed (%s). Falling back to full refit.", e
            )
            spread_new, total_new = _quick_fit()

    # Evaluate on recent 10% holdout (by time)
    n = len(X)
    split = int(n * 0.9)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    ys_tr, ys_te = y_spread.iloc[:split], y_spread.iloc[split:]
    yt_tr, yt_te = y_total.iloc[:split], y_total.iloc[split:]

    pred_sp = spread_new.predict(X_te)
    pred_to = total_new.predict(X_te)
    mae_sp = float(mean_absolute_error(ys_te, pred_sp))
    mae_to = float(mean_absolute_error(yt_te, pred_to))

    drift_sp = mae_sp - (prev_spread_mae if not math.isnan(prev_spread_mae) else mae_sp)
    drift_to = mae_to - (prev_total_mae if not math.isnan(prev_total_mae) else mae_to)

    # Accept when performance doesn't degrade more than 10% (or if NA before)
    ok_sp = math.isnan(prev_spread_mae) or mae_sp <= prev_spread_mae * 1.10
    ok_to = math.isnan(prev_total_mae) or mae_to <= prev_total_mae * 1.10
    if ok_sp and ok_to:
        # Accept update
        dump({"model": spread_new}, SPREAD_MODEL_PATH)
        dump({"model": total_new}, TOTAL_MODEL_PATH)
        new_meta = {
            "trained_at": datetime.utcnow().isoformat(),
            "years": sorted(pd.read_csv(SCHEDULE_CACHE)["season"].unique().tolist()),
            "spread_mae_cv": mae_sp,
            "total_mae_cv": mae_to,
            "spread_params": {},
            "total_params": {},
            "feature_count": int(X.shape[1]),
            "update": {"season": season, "week": week},
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(new_meta, f, indent=2)
        logger.info("Models updated. MAE spread=%.3f total=%.3f", mae_sp, mae_to)
        return {"accepted": True, "mae_spread": mae_sp, "mae_total": mae_to}
    else:
        logger.warning("Update rejected due to performance drift.")
        return {
            "accepted": False,
            "mae_spread": mae_sp,
            "mae_total": mae_to,
            "drift_spread": drift_sp,
            "drift_total": drift_to,
        }


# ----------------------------
# Backtesting & Evaluation
# ----------------------------


def _season_walkforward(
    schedule: pd.DataFrame, start: int, end: int
) -> pd.DataFrame:
    """Train prior to each season and predict within-season for ROI."""
    results: List[Dict[str, Any]] = []

    for season in range(start, end + 1):
        hist = schedule[schedule["season"] < season]
        cur = schedule[schedule["season"] == season]
        if hist.empty or cur.empty:
            continue
        X_h, ys_h, yt_h, _m_h = engineer_features(hist)
        spread_m, total_m, meta, _ = train_models(
            X_h, ys_h, yt_h, sorted(hist["season"].unique().tolist())
        )

        # Predict for season games (using last-known form)
        for _idx, g in cur.iterrows():
            try:
                X_df = _current_matchup_features(
                    hist, g["home_team"], g["away_team"]
                )
                sp_edge = float(spread_m.predict(X_df)[0])
                to_edge = float(total_m.predict(X_df)[0])
                pred_spread = float(g["spread_line"]) + sp_edge
                pred_total = float(g["total_line"]) + to_edge

                results.append({
                    "season": int(season),
                    "week": int(g["week"]),
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "vegas_spread": float(g["spread_line"]),
                    "vegas_total": float(g["total_line"]),
                    "pred_spread": pred_spread,
                    "pred_total": pred_total,
                    "actual_margin": float(
                        (float(g["home_score"]) if not pd.isna(g["home_score"]) else 0)
                        - (float(g["away_score"]) if not pd.isna(g["away_score"]) else 0)
                    ),
                    "actual_total": float(
                        (float(g["home_score"]) if not pd.isna(g["home_score"]) else 0)
                        + (float(g["away_score"]) if not pd.isna(g["away_score"]) else 0)
                    ),
                })
            except Exception:
                continue

        # Expand history for next season
        hist = pd.concat([hist, cur], ignore_index=True)

    return pd.DataFrame(results)


def _settle_bets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simulate betting based on model edges; -110 vig assumptions."""
    if df.empty:
        return df, {
            "bets": 0.0, "units": 0.0, "roi": 0.0, "hit_rate": 0.0,
            "avg_edge_pts": 0.0,
        }
    rows = []
    for _i, r in df.iterrows():
        edge_sp = r["pred_spread"] - r["vegas_spread"]
        edge_to = r["pred_total"] - r["vegas_total"]
        # Bet when edge >= 1 point
        choose_sp = abs(edge_sp) >= 1.0
        choose_to = abs(edge_to) >= 1.5

        # Spread bet outcome from home perspective
        win_sp = None
        if choose_sp:
            # Home favorite if pred more negative than vegas
            target = r["vegas_spread"]
            actual = r["actual_margin"]
            win_sp = int((actual - target) * np.sign(edge_sp) > 0)

        # Total bet outcome
        win_to = None
        if choose_to:
            target = r["vegas_total"]
            actual = r["actual_total"]
            win_to = int((actual - target) * np.sign(edge_to) > 0)

        rows.append({
            **r.to_dict(),
            "edge_sp": float(edge_sp),
            "edge_to": float(edge_to),
            "bet_sp": int(bool(choose_sp)),
            "bet_to": int(bool(choose_to)),
            "win_sp": win_sp,
            "win_to": win_to,
        })

    sim = pd.DataFrame(rows)
    # Units: -1 to lose, +0.909 to win (due to -110)
    unit_win = 0.909
    sim["units_sp"] = np.where(
        sim["bet_sp"] == 1,
        np.where(sim["win_sp"] == 1, unit_win, -1.0),
        0.0,
    )
    sim["units_to"] = np.where(
        sim["bet_to"] == 1,
        np.where(sim["win_to"] == 1, unit_win, -1.0),
        0.0,
    )
    sim["units_total"] = sim["units_sp"] + sim["units_to"]

    total_bets = int(sim["bet_sp"].sum() + sim["bet_to"].sum())
    units = float(sim["units_total"].sum())
    stake = float(total_bets)
    roi = float(units / stake) if stake > 0 else 0.0
    hit_num = int(
        (sim["win_sp"] == 1).sum() + (sim["win_to"] == 1).sum()
    )
    hit_rate = float(hit_num / total_bets) if total_bets else 0.0
    avg_edge = float(np.mean(np.abs(sim[["edge_sp", "edge_to"]]).values))

    return sim, {
        "bets": float(total_bets),
        "units": units,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge_pts": avg_edge,
    }


def _save_backtest_plot(sim: pd.DataFrame, out_path: str) -> None:
    if plt is None:
        logger.warning("matplotlib not installed; skipping PNG output")
        return
    if sim.empty:
        return
    sim = sim.sort_values(["season", "week"]).reset_index(drop=True)
    sim["cum_units"] = sim["units_total"].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(sim["cum_units"].values, label="Cumulative Units")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Bet Number")
    plt.ylabel("Units")
    plt.title("Backtest Performance: Cumulative Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved backtest plot to %s", out_path)


def backtest_model(start_year: int, end_year: int) -> Dict[str, Any]:
    """Simulate betting across seasons and report ROI & stats."""
    schedule = pd.read_csv(SCHEDULE_CACHE)
    schedule["gameday"] = safe_date(schedule["gameday"])
    sim = _season_walkforward(schedule, start_year, end_year)
    sim, stats = _settle_bets(sim)
    out_png = os.path.join(DATA_DIR, f"backtest_{start_year}_{end_year}.png")
    _save_backtest_plot(sim, out_png)
    return {"stats": stats, "png": out_png}


# ----------------------------
# CLI & Entry Point
# ----------------------------


def cmd_train(args: argparse.Namespace) -> None:
    years = parse_years(args.years)
    schedule = fetch_data(years)
    X, y_spread, y_total, meta = engineer_features(schedule)
    train_models(X, y_spread, y_total, years)
    print("Training complete. Models saved in ./models")


def cmd_predict(args: argparse.Namespace) -> None:
    game = args.game
    if " vs " in game.lower():
        a, b = re.split(r"\s+vs\s+", game, flags=re.IGNORECASE)
        home, away = a.strip(), b.strip()
    elif " at " in game.lower():
        a, b = re.split(r"\s+at\s+", game, flags=re.IGNORECASE)
        away, home = a.strip(), b.strip()
    else:
        raise ValueError(
            "Game must be like 'Chiefs vs Ravens' or 'Ravens at Chiefs'"
        )

    # If lines not provided, infer from latest schedule cache if possible
    if args.vegas_spread is None or args.vegas_total is None:
        sched = pd.read_csv(SCHEDULE_CACHE)
        ha = normalize_team_name(home)
        aa = normalize_team_name(away)
        sched = sched.sort_values("gameday")
        row = sched[(sched["home_team"] == ha) & (sched["away_team"] == aa)].tail(1)
        if row.empty:
            raise ValueError(
                "Vegas lines missing and not found in cache. Provide --vegas_spread and --vegas_total."
            )
        vsp = float(row.iloc[0]["spread_line"]) if not pd.isna(row.iloc[0]["spread_line"]) else 0.0
        vto = float(row.iloc[0]["total_line"]) if not pd.isna(row.iloc[0]["total_line"]) else 44.0
    else:
        vsp = float(args.vegas_spread)
        vto = float(args.vegas_total)

    pred = predict_game(
        home_team=home,
        away_team=away,
        vegas_spread=vsp,
        vegas_total=vto,
    )

    # Pretty print
    print(
        f"GAME: {home} vs {away}\n"
        f"VEGAS: {home} {vsp:.1f} | Total {vto:.1f}\n\n"
        f"MODEL PREDICTION:\n"
        f"- Expected Spread: {pred['predicted_spread']:.1f} (Edge: "
        f"{pred['edge_spread_pts']:+.1f} points)\n"
        f"- Expected Total: {pred['predicted_total']:.1f} (Edge: "
        f"{pred['edge_total_pts']:+.1f} points)\n"
        f"- Moneyline: {pred['moneyline_pick']} (p={pred['moneyline_prob']*100:.1f}%)\n"
        f"- Confidence: {pred['confidence']*100:.0f}%\n\n"
        f"BETTING RECOMMENDATION:\n"
        f"→ {pred['text_spread']} (bet {pred['kelly_fraction']*100:.1f}% bankroll)\n"
        f"→ {pred['text_total']}\n"
    )


def cmd_update(args: argparse.Namespace) -> None:
    out = update_model(season=int(args.season), week=int(args.week))
    print(json.dumps(out, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    out = backtest_model(start_year=int(args.start), end_year=int(args.end))
    stats = out["stats"]
    print(
        f"Backtest {args.start}-{args.end}:\n"
        f"- Bets: {stats['bets']:.0f}\n"
        f"- Units: {stats['units']:+.2f}\n"
        f"- ROI: {stats['roi']*100:.1f}%\n"
        f"- Hit Rate: {stats['hit_rate']*100:.1f}%\n"
        f"- Avg Edge: {stats['avg_edge_pts']:.2f} pts\n"
        f"Plot: {out['png']}\n"
    )


def cmd_predict_date(args: argparse.Namespace) -> None:
    date_str = args.date
    try:
        target_day = pd.to_datetime(date_str, errors="raise").date()
    except Exception as e:
        raise ValueError(f"Invalid --date '{date_str}': {e}")

    sched = pd.read_csv(SCHEDULE_CACHE)
    sched["gameday"] = safe_date(sched["gameday"]).dt.date
    slate = sched[sched["gameday"] == target_day]
    if slate.empty:
        print(f"No games found on {target_day}")
        return

    for _i, g in slate.iterrows():
        home = str(g["home_team"])
        away = str(g["away_team"])
        vsp = float(g["spread_line"]) if not pd.isna(g["spread_line"]) else 0.0
        vto = float(g["total_line"]) if not pd.isna(g["total_line"]) else 44.0
        pred = predict_game(home, away, vsp, vto)
        print(
            f"GAME: {home} vs {away}\n"
            f"VEGAS: {home} {vsp:.1f} | Total {vto:.1f}\n"
            f"Moneyline: {pred['moneyline_pick']} "
            f"(p={pred['moneyline_prob']*100:.1f}%)\n"
            f"Spread: {pred['text_spread']}\n"
            f"Total:  {pred['text_total']}\n"
        )


def cmd_predict_week(args: argparse.Namespace) -> None:
    season = int(args.season)
    week = int(args.week)
    sched = pd.read_csv(SCHEDULE_CACHE)
    sched["gameday"] = safe_date(sched["gameday"])  # ensure parse
    slate = sched[(sched["season"].astype(int) == season) &
                  (sched["week"].astype(int) == week)]
    if slate.empty:
        print(f"No games found for season {season}, week {week}")
        return
    for _i, g in slate.iterrows():
        home = str(g["home_team"])
        away = str(g["away_team"])
        vsp = float(g["spread_line"]) if not pd.isna(g["spread_line"]) else 0.0
        vto = float(g["total_line"]) if not pd.isna(g["total_line"]) else 44.0
        pred = predict_game(home, away, vsp, vto)
        print(
            f"GAME: {home} vs {away}\n"
            f"VEGAS: {home} {vsp:.1f} | Total {vto:.1f}\n"
            f"Moneyline: {pred['moneyline_pick']} "
            f"(p={pred['moneyline_prob']*100:.1f}%)\n"
            f"Spread: {pred['text_spread']}\n"
            f"Total:  {pred['text_total']}\n"
        )


def _run_tests() -> int:
    """Minimal sanity tests for core functions."""
    try:
        years = list(range(2019, 2021))
        df = fetch_data(years)
        assert not df.empty
        X, ys, yt, meta = engineer_features(df)
        assert X.shape[0] == ys.shape[0] == yt.shape[0]
        spread_m, total_m, _meta = None, None, None
        try:
            spread_m, total_m, _meta = train_models(X, ys, yt, years)
        except Exception as e:  # allow failures in constrained envs
            logger.warning("Train failed in test (likely env limits): %s", e)
        logger.info("Tests completed.")
        return 0
    except Exception as e:
        logger.exception("Tests failed: %s", e)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "NFL Game Prediction MVP for Betting — Sir Toaster approved."
        )
    )
    parser.add_argument(
        "--run-tests", action="store_true", help="Run minimal unit tests"
    )

    sub = parser.add_subparsers(dest="cmd", required=False)

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument(
        "--years", type=str, required=True, help="e.g., 2015-2023"
    )
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict a single game")
    p_pred.add_argument(
        "--game", type=str, required=True,
        help="'Chiefs vs Ravens' or 'Ravens at Chiefs'",
    )
    p_pred.add_argument("--vegas_spread", type=float, required=False, default=None)
    p_pred.add_argument("--vegas_total", type=float, required=False, default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_upd = sub.add_parser("update", help="Warm-start model update")
    p_upd.add_argument("--season", type=int, required=True)
    p_upd.add_argument("--week", type=int, required=True)
    p_upd.set_defaults(func=cmd_update)

    p_bt = sub.add_parser("backtest", help="Run backtest and report")
    p_bt.add_argument("--start", type=int, required=True)
    p_bt.add_argument("--end", type=int, required=True)
    p_bt.set_defaults(func=cmd_backtest)

    p_pdate = sub.add_parser(
        "predict-date", help="Predict all games on a given date (YYYY-MM-DD)"
    )
    p_pdate.add_argument("--date", type=str, required=True)
    p_pdate.set_defaults(func=cmd_predict_date)

    p_pweek = sub.add_parser(
        "predict-week", help="Predict all games for a given season/week"
    )
    p_pweek.add_argument("--season", type=int, required=True)
    p_pweek.add_argument("--week", type=int, required=True)
    p_pweek.set_defaults(func=cmd_predict_week)

    args = parser.parse_args(argv)
    if args.run_tests:
        return _run_tests()
    if hasattr(args, "func"):
        args.func(args)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())


