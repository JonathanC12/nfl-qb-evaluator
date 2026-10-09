"""Weekly live-board refresh: no backtests, just applies the current (v2 adjustment, v3 composite)
model to the current season and saves per-week snapshots for the Streamlit app.

Run: python update_board.py [--season 2026] [--min-dropbacks 60]

For each week 1..latest, writes `boards/<season>_wk<week>.parquet` (season-to-date-through-that-week
qb_table, shrunk and graded) and one cumulative `boards/<season>_interceptions.parquet` (play-level, for
the QB-page interception list).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import qb_eval as q
import backtest as bt

BOARDS_DIR = Path(__file__).parent / "boards"
HIST_SEASONS = [2022, 2023, 2024, 2025]
V2_ADJUST = {"int": True}   # the model locked in by the backtest (qb_backtest.ipynb section 4)

INT_COLS = ["season", "week", "id", "name", "posteam", "defteam", "int_bucket", "qb_epa", "adj_epa",
            "down", "ydstogo", "yardline_100", "air_yards", "desc"]


def update(season: int, min_dropbacks: int = 60) -> None:
    BOARDS_DIR.mkdir(exist_ok=True)

    # Historical seasons give us the value models (for the INT adjustment) and the shrinkage constants.
    # This is not a backtest: no accuracy scoring, just the inputs the live model needs.
    hist = q.load_seasons(HIST_SEASONS)
    models = q.fit_value_models(hist)
    hist_db = q.adjust_dropbacks(hist, models)
    hist_db["adj_epa"] = q.recompute_adj(hist_db, V2_ADJUST)
    hist_db["garbage"] = False
    hist_runs = q.designed_qb_runs(hist, set(hist_db["id"]))
    hist_runs["garbage"] = False
    w = bt.split_half_table(hist_db, hist_runs)
    k_by_metric = {m: bt.shrinkage_k(w, m) for m in q.CONFIG["weights"]}

    live = q.load_season(season, refresh=True)
    live_db = q.adjust_dropbacks(live, models)
    live_db["adj_epa"] = q.recompute_adj(live_db, V2_ADJUST)
    live_db["garbage"] = False
    live_runs = q.designed_qb_runs(live, set(live_db["id"]))
    live_runs["garbage"] = False

    max_week = int(live_db["week"].max())
    for wk in range(1, max_week + 1):
        db_wk = live_db[live_db["week"] <= wk]
        runs_wk = live_runs[live_runs["week"] <= wk]
        tbl = q.qb_table(db_wk, runs_wk)
        tbl = tbl[tbl["dropbacks"] >= min_dropbacks].copy()
        if tbl.empty:
            continue
        for m, k in k_by_metric.items():
            tbl[m] = q.shrink(tbl[m].fillna(tbl[m].mean()), tbl["dropbacks"], k)
        tbl = q.add_grade(tbl)
        tbl.insert(1, "week", wk)
        path = BOARDS_DIR / f"{season}_wk{wk}.parquet"
        tbl.to_parquet(path, index=False)
        print(f"wrote {path} ({len(tbl)} QBs)")

    ints = live_db[live_db["interception"] == 1][INT_COLS].rename(columns={"posteam": "team"})
    ints_path = BOARDS_DIR / f"{season}_interceptions.parquet"
    ints.to_parquet(ints_path, index=False)
    print(f"wrote {ints_path} ({len(ints)} interceptions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--min-dropbacks", type=int, default=60)
    args = parser.parse_args()
    update(args.season, args.min_dropbacks)
