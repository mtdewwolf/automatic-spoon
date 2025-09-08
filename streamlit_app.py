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
import math
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
try:
    import nfl_data_py as nfl
except Exception:  # pragma: no cover
    nfl = None

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


def _fmt_date(dt_val: Any) -> str:
    try:
        if pd.isna(dt_val):
            return "TBD"
        d = pd.to_datetime(dt_val, errors="coerce")
        if pd.isna(d):
            return "TBD"
        return d.strftime("%Y-%m-%d")
    except Exception:
        return "TBD"


@st.cache_data(show_spinner=False)
def _team_colors() -> Dict[str, Dict[str, str]]:
    """Return mapping of team abbr -> {'c1': '#xxxxxx', 'c2': '#xxxxxx'}.
    Uses nfl_data_py.import_team_desc when available; otherwise defaults.
    """
    default = {
        "c1": "#1f2937",  # slate gray
        "c2": "#374151",
    }
    mapping: Dict[str, Dict[str, str]] = {}
    try:
        if nfl is None:
            raise RuntimeError("nfl_data_py missing")
        desc = nfl.import_team_desc()
        for _, r in desc.iterrows():
            abbr = str(r.get("team_abbr", "")).strip()
            c1 = str(r.get("team_color", "")).strip() or default["c1"]
            c2 = str(r.get("team_color2", "")).strip() or c1
            if not c1.startswith("#"):
                c1 = "#" + c1
            if not c2.startswith("#"):
                c2 = "#" + c2
            mapping[abbr] = {"c1": c1, "c2": c2}
    except Exception:
        pass
    return mapping


def _hex_to_rgba(hex_str: str, alpha: float) -> str:
    h = hex_str.lstrip('#')
    if len(h) == 3:
        h = ''.join([c*2 for c in h])
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(31,41,55,{alpha})"  # slate fallback


def _render_game_cards(df: pd.DataFrame, cols: int = 3) -> None:
    """Render games as bordered cards instead of a table."""
    if df.empty:
        st.info("No games to display.")
        return
    df = df.reset_index(drop=True)
    colors_map = _team_colors()
    num = len(df)
    i = 0
    while i < num:
        col_objs = st.columns(cols)
        for j in range(cols):
            if i + j >= num:
                break
            r = df.iloc[i + j]
            away_abbr = str(r.get('away', 'TBD'))
            home_abbr = str(r.get('home', 'TBD'))
            matchup = f"{away_abbr} at {home_abbr}"
            date_str = _fmt_date(r.get("date"))
            pick = r.get("moneyline_pick", "-")
            prob = float(r.get("moneyline_prob", 0.0)) * 100.0
            vsp = r.get("vegas_spread", float("nan"))
            vto = r.get("vegas_total", float("nan"))
            ps = r.get("pred_spread", float("nan"))
            pt = r.get("pred_total", float("nan"))
            # Colors and gradient background
            ac = colors_map.get(away_abbr, {"c1": "#7f1d1d", "c2": "#991b1b"})
            hc = colors_map.get(home_abbr, {"c1": "#065f46", "c2": "#047857"})
            a1 = _hex_to_rgba(ac["c1"], 0.80)
            a2 = _hex_to_rgba(ac["c2"], 0.80)
            h1 = _hex_to_rgba(hc["c1"], 0.80)
            h2 = _hex_to_rgba(hc["c2"], 0.80)
            bg = (
                "linear-gradient(135deg, "
                f"{a1} 0%, {a2} 48%, {h1} 52%, {h2} 100%)"
            )
            html = f"""
            <div style='border:1px solid #374151;border-radius:14px;
                        padding:12px;margin:6px;background:{bg};'>
              <div style='font-weight:700;font-size:1.05rem;'>
                {matchup}
              </div>
              <div style='color:#9ca3af;font-size:0.85rem;'>
                {date_str}
              </div>
              <hr style='opacity:0.15;'>
              <div style='margin-bottom:6px;'>
                <span style='color:#9ca3af;'>Moneyline:</span>
                <b>{pick}</b>
                <span style='color:#9ca3af;'>(p={prob:.1f}%)</span>
              </div>
              <div style='color:#9ca3af;font-size:0.9rem;'>
                Spread: <b>{ps:.1f}</b>
                <span style='color:#6b7280'>(Vegas {vsp:.1f})</span>
              </div>
              <div style='color:#9ca3af;font-size:0.9rem;'>
                Total: <b>{pt:.1f}</b>
                <span style='color:#6b7280'>(Vegas {vto:.1f})</span>
              </div>
            </div>
            """
            with col_objs[j]:
                st.markdown(html, unsafe_allow_html=True)
        i += cols


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

# Admin gating: hide Train/Update/Backtest unless enabled via env or secrets
ADMIN_MODE = False
if os.environ.get("NFLP_ADMIN"):
    ADMIN_MODE = True
else:
    # Guard against missing secrets.toml
    try:
        ADMIN_MODE = bool(st.secrets.get("ADMIN", False))
    except Exception:
        ADMIN_MODE = False

if ADMIN_MODE:
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
else:
    tab_pred, tab_date, tab_week, tab_about = st.tabs(
        [
            "Single Game",
            "Predict by Date",
            "Predict by Week",
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

    st.divider()
    st.subheader("Upcoming Games — Season Snapshot")
    try:
        sched_all = _read_schedule()
        if not sched_all.empty:
            # Default to latest season present
            latest_season = int(sched_all["season"].dropna().astype(int).max())
            up_season = st.number_input(
                "Season", min_value=2015, max_value=2100,
                value=latest_season, key="upcoming_season"
            )
            if st.button("Show Upcoming Season Predictions", key="btn_upcoming"):
                with st.spinner("Computing current predictions for upcoming games..."):
                    sched_all["gameday"] = pd.to_datetime(
                        sched_all["gameday"], errors="coerce"
                    )
                    today = pd.Timestamp(date.today())
                    # Upcoming if future date or not yet scored
                    slate = sched_all[(sched_all["season"].astype(int) == int(up_season)) & (
                        (sched_all["gameday"].isna()) | (sched_all["gameday"] >= today) |
                        (sched_all["home_score"].isna() & sched_all["away_score"].isna())
                    )]
                    if slate.empty:
                        st.info("No upcoming games found for the selected season.")
                    else:
                        res = _predict_slate_rows(slate)
                        if res.empty:
                            st.info("Unable to compute predictions for the slate.")
                        else:
                            res = res.sort_values(["date", "home"]).reset_index(False)
                            _render_game_cards(res, cols=3)
        else:
            st.info("No schedule available. Train first on the Train tab.")
    except Exception:
        st.error("Upcoming games computation failed:")
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
                st.info("No games found on selected date (train first or choose another date).")
            else:
                res = res.sort_values(["date", "home"]).reset_index(drop=True)
                _render_game_cards(res, cols=3)
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
                st.info("No games found for that season/week (train first or adjust).")
            else:
                res = res.sort_values(["date", "home"]).reset_index(drop=True)
                _render_game_cards(res, cols=3)
        except Exception:
            st.error("Week prediction failed:")
            st.code(traceback.format_exc())


if ADMIN_MODE:
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


if ADMIN_MODE:
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


if ADMIN_MODE:
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


