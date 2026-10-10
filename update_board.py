"""Weekly live-board refresh: no backtests, just applies the current (v2 adjustment, v3 composite)
model to the current season and saves per-week snapshots for the Streamlit app.

Run: python update_board.py [--season 2026] [--qualifier-per-game 14]

Saves ALL quarterbacks with at least one dropback -- no dropback filter is applied here. The app
applies every display/visibility threshold itself; this script only computes a `qualifier_dropbacks`
column (the NFL passer-rating-style qualifier: `qualifier_per_game` dropbacks per team game played) and
a `qualified` flag, and restricts shrinkage + grading to the qualified population so a handful of
mop-up dropbacks cannot distort the scale. Unqualified QBs are still saved (grade/composite = NaN).

For each week, writes two files:
  * `boards/<season>_wk<week>.parquet`         -- CUMULATIVE, season-to-date through that week
  * `boards/<season>_wk<week>_weekly.parquet`  -- that week's plays only, shrunk harder (single games
    are noisier than a season-to-date sample; the shrinkage constant is measured from consecutive-game
    reliability, not an arbitrary multiplier -- see backtest.game_pairs_table)

Only builds weeks where every scheduled game is final (nflreadpy load_schedules): a week in progress
is never included, so the latest board always reflects a fully-played week.

Also writes one cumulative `boards/<season>_interceptions.parquet` (play-level, for the QB-page
interception list), with full display names (not the "J.Daniels"-style short names pbp uses, which
collide across different players).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import qb_eval as q
import backtest as bt

BOARDS_DIR = Path(__file__).parent / "boards"
HIST_SEASONS = [2022, 2023, 2024, 2025]
V2_ADJUST = {"int": True}   # the model locked in by the backtest (qb_backtest.ipynb section 4)

INT_COLS = ["season", "week", "id", "name", "posteam", "defteam", "int_bucket", "qb_epa", "adj_epa",
            "down", "ydstogo", "yardline_100", "air_yards", "desc"]


def player_info(ids: set[str]) -> tuple[dict[str, str], set[str]]:
    """GSIS id -> full display name, and the subset of `ids` whose nflreadpy position is QB.

    Both come from nflreadpy.load_players(). Names fall back silently to the pbp short name for any
    id not found there (callers keep that as their own fallback). The position set excludes
    trick-play passers (a WR throwing a halfback pass still gets a `qb_dropback` row with a real QB
    GSIS id as passer) from ever reaching the board.
    """
    import nflreadpy as nfl
    players = nfl.load_players().to_pandas()
    players = players[players["gsis_id"].isin(ids)]
    names = dict(zip(players["gsis_id"], players["display_name"]))
    qb_ids = set(players.loc[players["position"] == "QB", "gsis_id"])
    return names, qb_ids


def completed_weeks_and_team_games(season: int) -> tuple[list[int], pd.DataFrame]:
    """Weeks where every scheduled REG game is final (as a contiguous prefix starting at week 1), and
    a long (week, team, games_played) table -- the count of that team's COMPLETED games through each
    week, which is what the passer-rating-style qualifier scales by (accounts for bye weeks).
    """
    import nflreadpy as nfl
    sched = nfl.load_schedules([season]).to_pandas()
    sched = sched[sched["game_type"] == "REG"]
    max_week = int(sched["week"].max())

    final = sched["home_score"].notna() & sched["away_score"].notna()
    all_final_by_week = sched.assign(final=final).groupby("week")["final"].all()
    completed_weeks = []
    for wk in range(1, max_week + 1):
        if not all_final_by_week.get(wk, False):
            break
        completed_weeks.append(wk)

    done = sched[final]
    team_games = pd.concat([
        done[["week", "home_team"]].rename(columns={"home_team": "team"}),
        done[["week", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    rows = []
    for wk in completed_weeks:
        played = team_games[team_games["week"] <= wk].groupby("team").size()
        rows.append(pd.DataFrame({"week": wk, "team": played.index, "games_played": played.values}))
    team_games_played = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["week", "team", "games_played"])
    return completed_weeks, team_games_played


def grade_among_qualified(tbl: pd.DataFrame, k_by_metric: dict[str, float]) -> pd.DataFrame:
    """Shrink + grade only the qualified rows (so a handful of mop-up dropbacks can't skew the scale),
    then reattach the unqualified rows with grade/composite left as NaN ("low sample", no rank).
    """
    qualified = tbl[tbl["qualified"]].copy()
    unqualified = tbl[~tbl["qualified"]].copy()
    if qualified.empty:
        return tbl.assign(composite=np.nan, grade=np.nan)
    for m, k in k_by_metric.items():
        qualified[m] = q.shrink(qualified[m].fillna(qualified[m].mean()), qualified["dropbacks"], k)
    qualified = q.add_grade(qualified)
    unqualified["composite"] = np.nan
    unqualified["grade"] = np.nan
    return pd.concat([qualified, unqualified], ignore_index=True)


def write_team_colors() -> None:
    """boards/team_colors.parquet: the app reads this instead of calling nflreadpy itself, so it
    never needs network access at startup (team colors barely ever change; refreshed each run here
    for free rather than kept as a separate maintenance step).
    """
    import nflreadpy as nfl
    teams = nfl.load_teams().to_pandas()[["team_abbr", "team_color"]]
    teams.to_parquet(BOARDS_DIR / "team_colors.parquet", index=False)


def update(season: int, qualifier_per_game: int = 14) -> None:
    BOARDS_DIR.mkdir(exist_ok=True)
    write_team_colors()

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
    k_cumulative = {m: bt.shrinkage_k(w, m) for m in q.CONFIG["weights"]}
    # Weekly (single-game) shrinkage: measured from consecutive-game reliability, floored at the
    # cumulative constant so a single game is never shrunk *less* than a season-to-date sample.
    paired = bt.game_pairs_table(hist_db, hist_runs)
    k_weekly = {m: max(bt.shrinkage_k(paired, m), k_cumulative[m]) for m in q.CONFIG["weights"]}
    print("shrinkage k, cumulative:", {m: round(k) for m, k in k_cumulative.items()})
    print("shrinkage k, weekly:    ", {m: round(k) for m, k in k_weekly.items()})

    completed_weeks, team_games_played = completed_weeks_and_team_games(season)
    if not completed_weeks:
        print(f"No fully-completed weeks yet for {season}; nothing to do.")
        return
    max_week = completed_weeks[-1]
    print(f"{season}: weeks {completed_weeks} are fully final (latest in-progress week excluded)")

    live = q.load_season(season, refresh=True)
    live_db = q.adjust_dropbacks(live, models)
    names, qb_ids = player_info(set(live_db["id"].dropna()))
    # Drop trick-play passers (e.g. a WR throwing a halfback pass still logs a qb_dropback row) --
    # only players nflreadpy lists at position QB make the board at all.
    live_db = live_db[live_db["id"].isin(qb_ids)].copy()
    live_db["adj_epa"] = q.recompute_adj(live_db, V2_ADJUST)
    live_db["garbage"] = False
    live_db["name"] = live_db["id"].map(names).fillna(live_db["name"])
    live_runs = q.designed_qb_runs(live, set(live_db["id"]))
    live_runs["garbage"] = False

    for wk in completed_weeks:
        tg = team_games_played[team_games_played["week"] == wk].set_index("team")["games_played"]

        # Cumulative: season-to-date through this week, qualifier scales with each team's games played.
        db_cum = live_db[live_db["week"] <= wk]
        runs_cum = live_runs[live_runs["week"] <= wk]
        cum = q.qb_table(db_cum, runs_cum)
        cum["team_games_played"] = cum["team"].map(tg).fillna(0).astype(int)
        cum["qualifier_dropbacks"] = qualifier_per_game * cum["team_games_played"]
        cum["qualified"] = cum["dropbacks"] >= cum["qualifier_dropbacks"]
        cum = grade_among_qualified(cum, k_cumulative)
        cum.insert(1, "week", wk)
        path = BOARDS_DIR / f"{season}_wk{wk}.parquet"
        cum.to_parquet(path, index=False)
        print(f"wrote {path} ({len(cum)} QBs, {cum['qualified'].sum()} qualified)")

        # Weekly: that week's plays only, flat one-game qualifier, stronger shrinkage.
        db_wk = live_db[live_db["week"] == wk]
        runs_wk = live_runs[live_runs["week"] == wk]
        weekly = q.qb_table(db_wk, runs_wk)
        weekly["team_games_played"] = 1
        weekly["qualifier_dropbacks"] = qualifier_per_game
        weekly["qualified"] = weekly["dropbacks"] >= weekly["qualifier_dropbacks"]
        weekly = grade_among_qualified(weekly, k_weekly)
        weekly.insert(1, "week", wk)
        wpath = BOARDS_DIR / f"{season}_wk{wk}_weekly.parquet"
        weekly.to_parquet(wpath, index=False)
        print(f"wrote {wpath} ({len(weekly)} QBs, {weekly['qualified'].sum()} qualified)")

    ints = live_db[(live_db["interception"] == 1) & (live_db["week"] <= max_week)][INT_COLS]
    ints = ints.rename(columns={"posteam": "team"})
    ints_path = BOARDS_DIR / f"{season}_interceptions.parquet"
    ints.to_parquet(ints_path, index=False)
    print(f"wrote {ints_path} ({len(ints)} interceptions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--qualifier-per-game", type=int, default=14)
    args = parser.parse_args()
    update(args.season, args.qualifier_per_game)
