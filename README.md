# NFL Predictor MVP

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## CLI

```bash
# Train (first run caches data/models)
python nfl_predictor.py train --years 2015-2025

# Predict a game
python nfl_predictor.py predict --game "Chiefs vs Ravens" --vegas_spread -2.5 --vegas_total 47.5

# Update models with latest week
python nfl_predictor.py update --season 2025 --week 2

# Backtest
python nfl_predictor.py backtest --start 2018 --end 2025
```

## Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

- Admin tabs (Train/Update/Backtest) are hidden by default.
  - To reveal: set env `NFLP_ADMIN=1` before launching Streamlit.

## Data & Features
- Schedules via nfl_data_py with Vegas lines when available; cached in `./data`.
- Weekly player stats (QBs + team aggregates).
- Optional PBP integration:
  - Place CSVs in `./data` named like `pbp-2015.csv`, `pbp-2024.csv`.
  - Or let the app fetch PBP via nfl_data_py and cache team-game aggregates to `./data/pbp_agg.csv`.
- Engineered features include recent form, season-to-date, rest, dome/public proxies,
  QB signals, rolling team offense/defense, and PBP rates (YPP, pass rate, explosive,
  sack/int rates, penalties, success rate).

## Notes
- Python 3.12 recommended. Predictions are probabilistic; bet responsibly.
- Models and data are saved under `./models` and `./data`.
