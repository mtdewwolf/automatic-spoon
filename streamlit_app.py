"""
Streamlit Dashboard for NFL Predictor

Run:
  streamlit run streamlit_app.py

Provides:
- Single-game predictions (spread/total + moneyline)
- Date slate predictions (moneyline focus)
- Week slate predictions (moneyline focus)
- Train models with selected years
- Update models for latest completed week or specific week
- Backtest with ROI metrics and chart

Note: Predictions are probabilistic; bet responsibly.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, date
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from nfl_predictor import (
    fetch_data,
    engineer_features,
    train_models,
    predict_game,
    backtest_model,
    update_model,
    normalize_team_name,
    SCHEDULE_CACHE,
    DATA_DIR,
)


# ------------------------------
# Helpers
# ------------------------------


def _read_schedule() -> pd.DataFrame:
    if not os.path.exists(SCHEDULE_CACHE):
        st.warning("Schedule cache not found. Please run training first.")
        return pd.DataFrame()
    df = pd.read_csv(SCHEDULE_CACHE)
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df


def _predict_slate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for _i, g in rows.iterrows():
        try:
            home = str(g["home_team"]) if pd.notna(g["home_team"]) else ""
            away = str(g["away_team"]) if pd.notna(g["away_team"]) else ""
            if not home or not away:
                continue
            vsp = float(g["spread_line"]) if pd.notna(g.get("spread_line")) else 0.0
            vto = float(g["total_line"]) if pd.notna(g.get("total_line")) else 44.0
            pred = predict_game(home, away, vsp, vto)
            out_rows.append(
                {
                    "season": int(g.get("season", 0) or 0),
                    "week": int(g.get("week", 0) or 0),
                    "date": g.get("gameday"),
                    "home": home,
                    "away": away,
                    "vegas_spread": vsp,
                    "vegas_total": vto,
                    "moneyline_pick": pred.get("moneyline_pick"),
                    "moneyline_prob": float(pred.get("moneyline_prob", 0.0)),
                    "pred_spread": float(pred.get("predicted_spread", 0.0)),
                    "edge_spread": float(pred.get("edge_spread_pts", 0.0)),
                    "pred_total": float(pred.get("predicted_total", 0.0)),
                    "edge_total": float(pred.get("edge_total_pts", 0.0)),
                }
            )
        except Exception as e:
            st.warning(f"Prediction failed for a game: {e}")
            continue
    return pd.DataFrame(out_rows)


# ------------------------------
# Streamlit UI
# ------------------------------


st.set_page_config(
    page_title="NFL Predictor",
    page_icon="🏈",
    layout="wide",
)

st.title("NFL Betting Predictor Dashboard")
st.caption("For Sir Toaster the Third — May the odds be ever in your favor")

tab_pred, tab_date, tab_week, tab_train, tab_update, tab_bt, tab_about = st.tabs(
    [
        "Single Game",
        "Predict by Date",
        "Predict by Week",
        "Train",
        "Update",
        "Backtest",
        "About",
    ]
)


with tab_pred:
    st.subheader("Single Game Prediction")
    c1, c2 = st.columns(2)
    with c1:
        home = st.text_input("Home Team", "Washington")
    with c2:
        away = st.text_input("Away Team", "Green Bay Packers")
    c3, c4 = st.columns(2)
    with c3:
        vegas_spread = st.number_input("Vegas Spread (home)", value=0.0, step=0.5)
    with c4:
        vegas_total = st.number_input("Vegas Total", value=44.0, step=0.5)
    if st.button("Predict", type="primary"):
        try:
            with st.spinner("Predicting..."):
                pred = predict_game(home, away, vegas_spread, vegas_total)
            st.success(
                f"Moneyline: {pred['moneyline_pick']} (p={pred['moneyline_prob']*100:.1f}%)"
            )
            st.metric(
                "Expected Spread",
                f"{pred['predicted_spread']:.1f}",
                f"{pred['edge_spread_pts']:+.1f} vs Vegas",
            )
            st.metric(
                "Expected Total",
                f"{pred['predicted_total']:.1f}",
                f"{pred['edge_total_pts']:+.1f} vs Vegas",
            )
            st.write(pred["text_spread"])
            st.write(pred["text_total"])
        except Exception:
            st.error("Prediction failed:")
            st.code(traceback.format_exc())


with tab_date:
    st.subheader("Predict All Games on Date")
    d: date = st.date_input("Date", value=date.today())
    if st.button("Predict Date Slate"):
        try:
            with st.spinner("Loading schedule and predicting..."):
                sched = _read_schedule()
                slate = sched[sched["gameday"].dt.date == pd.Timestamp(d).date()]
                res = _predict_slate_rows(slate)
            if res.empty:
                st.info("No games found on selected date.")
            else:
                res = res.sort_values(["date", "home"]).reset_index(drop=True)
                res["moneyline_prob"] = (res["moneyline_prob"] * 100).round(1)
                st.dataframe(res, width="stretch")
        except Exception:
            st.error("Date slate prediction failed:")
            st.code(traceback.format_exc())


with tab_week:
    st.subheader("Predict All Games in Week")
    season = st.number_input(
        "Season", min_value=2015, max_value=2100,
        value=datetime.now().year, key="week_season"
    )
    week = st.number_input(
        "Week", min_value=1, max_value=22, value=1, key="week_week"
    )
    if st.button("Predict Week"):
        try:
            with st.spinner("Loading schedule and predicting week..."):
                sched = _read_schedule()
                slate = sched[(sched["season"].astype(int) == int(season)) &
                              (sched["week"].astype(int) == int(week))]
                res = _predict_slate_rows(slate)
            if res.empty:
                st.info("No games found for that season/week.")
            else:
                res = res.sort_values(["date", "home"]).reset_index(drop=True)
                res["moneyline_prob"] = (res["moneyline_prob"] * 100).round(1)
                st.dataframe(res, width="stretch")
        except Exception:
            st.error("Week prediction failed:")
            st.code(traceback.format_exc())


with tab_train:
    st.subheader("Train Models")
    years_str = st.text_input("Years (e.g., 2015-2025 or 2018,2019,2020)", "2015-2025")
    if st.button("Run Training"):
        try:
            with st.spinner("Fetching data and training (this may take a while)..."):
                # Parse years
                years: List[int] = []
                for chunk in years_str.replace(" ", "").split(","):
                    if not chunk:
                        continue
                    if "-" in chunk:
                        a, b = map(int, chunk.split("-"))
                        years.extend(range(a, b + 1))
                    else:
                        years.append(int(chunk))
                years = sorted(set(years))

                sched = fetch_data(years)
                X, ys, yt, _m = engineer_features(sched)
                train_models(X, ys, yt, years)
            st.success("Training complete. Models saved to ./models")
        except Exception:
            st.error("Training failed:")
            st.code(traceback.format_exc())


with tab_update:
    st.subheader("Update Models (Warm-start or Refit)")
    c1, c2 = st.columns(2)
    with c1:
        u_season = st.number_input(
            "Season", min_value=2015, max_value=2100,
            value=datetime.now().year, key="update_season"
        )
    with c2:
        u_week = st.number_input(
            "Week", min_value=1, max_value=22, value=1, key="update_week"
        )
    if st.button("Run Update"):
        try:
            with st.spinner("Updating models..."):
                out = update_model(season=int(u_season), week=int(u_week))
            st.success("Update complete:")
            st.json(out)
        except Exception:
            st.error("Update failed:")
            st.code(traceback.format_exc())


with tab_bt:
    st.subheader("Backtest")
    b_c1, b_c2 = st.columns(2)
    with b_c1:
        b_start = st.number_input("Start Year", min_value=2015, max_value=2100, value=2018)
    with b_c2:
        b_end = st.number_input("End Year", min_value=2015, max_value=2100, value=datetime.now().year)
    if st.button("Run Backtest"):
        try:
            with st.spinner("Running backtest..."):
                out = backtest_model(start_year=int(b_start), end_year=int(b_end))
            st.success("Backtest complete.")
            st.json(out.get("stats", {}))
            img_path = out.get("png")
            if img_path and os.path.exists(img_path):
                st.image(img_path, caption="Cumulative Units", use_column_width=True)
        except Exception:
            st.error("Backtest failed:")
            st.code(traceback.format_exc())


with tab_about:
    st.subheader("About")
    st.write(
        "This dashboard wraps the single-file NFL predictor to provide"
        " interactive training, updates, and predictions."
    )
    st.write("Models and data are stored under ./models and ./data.")
    st.write("Predictions are probabilistic and not guarantees.")


