"""
QB Evaluator core: fault-adjusted EPA and a 1-100 QB grade.

Data: nflverse play-by-play + FTN charting (2022+), via nflreadpy.

Adjustments to each dropback's EPA (all knobs live in CONFIG):
  * INT-worthy throw, intercepted     -> blend of actual INT value and an incompletion
  * INT-worthy throw, NOT intercepted -> charged an expected INT cost ("dropped INT")
  * INT that was NOT INT-worthy       -> treated as an incompletion (not the QB's fault)
  * Dropped pass                      -> credited with the expected value of a completion
  * Sack that was NOT the QB's fault  -> only part of the cost charged to the QB
  * Garbage time removed (win prob outside 10-90%)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path(__file__).parent / "data"

CONFIG = {
    "wp_low": 0.10,               # garbage-time filter
    "wp_high": 0.90,
    "nonfault_sack_share": 0.5,   # share of a non-QB-fault sack's cost the QB still eats
    "min_dropbacks_season": 200,  # qualifying threshold for season-level tests
    "adjust": {"int": True, "drop": True, "sack": True, "yac": False},  # toggle each adjustment (for ablation)
    "weights": {                  # composite grade weights
        # v3: swapped fault_sack_rate -> total_sack_rate, added success_rate, dropped cpoe.
        # Chosen after backtesting (post-hoc, like v2) -- see qb_backtest.ipynb section 9.
        "adj_epa_db": 0.45,
        "success_rate": 0.20,
        "int_worthy_rate": -0.15,
        "total_sack_rate": -0.15,
        "rush_epa_db": 0.10,
    },
}

FTN_BOOL_COLS = [
    "is_interception_worthy", "is_drop", "is_catchable_ball", "is_throw_away",
    "is_qb_fault_sack", "is_contested_ball", "is_created_reception",
]
# read_thrown is categorical (see adjust_dropbacks), not a bool flag like the rest of FTN_COLS.
FTN_COLS = FTN_BOOL_COLS + ["read_thrown"]

# FTN `read_thrown` codes, confirmed against the actual charting data (the nflreadr dictionary page
# describes "0"/"1" backwards from what the data shows -- verified here via sack_rate/comp_rate/aDOT
# by code: "0" is ~72% sacks with ~1% completions, "1" is the highest-volume, highest-EPA bucket).
# "1" = first read, "2" = second-or-later read, "CHK" = checkdown, "DES" = designed/screen (no real
# progression), "SD" = scramble drill (off-script). "0" / NaN = no read charted (sacks, throwaways,
# non-pass plays, or simply uncharted -- 2022 used NaN for this instead of "0").
READ_PROGRESSION_CODES = ["1", "2", "CHK", "DES", "SD"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "home_team", "posteam", "defteam",
    "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name", "id", "name",
    "qb_dropback", "qb_scramble", "pass_attempt", "sack", "interception", "complete_pass",
    "incomplete_pass", "qb_spike", "qb_kneel", "play_type", "two_point_attempt",
    "epa", "qb_epa", "air_epa", "yac_epa", "xyac_epa", "ep", "wp", "cpoe", "success",
    "air_yards", "yards_after_catch", "yards_gained", "down", "ydstogo", "yardline_100",
    "half_seconds_remaining", "game_seconds_remaining", "score_differential",
    "touchdown", "pass_touchdown", "fumble_lost", "qb_hit", "desc",
]


def load_season(season: int, refresh: bool = False) -> pd.DataFrame:
    """Play-by-play joined to FTN charting for one season (cached as parquet)."""
    import pyarrow.parquet as pq
    DATA_DIR.mkdir(exist_ok=True)
    pbp_path, ftn_path = DATA_DIR / f"pbp_{season}.parquet", DATA_DIR / f"ftn_{season}.parquet"
    stale = pbp_path.exists() and not set(PBP_COLS) <= set(pq.read_schema(pbp_path).names)
    if refresh or stale or not pbp_path.exists() or not ftn_path.exists():
        import nflreadpy as nfl
        nfl.load_pbp([season]).select(PBP_COLS).write_parquet(pbp_path)
        nfl.load_ftn_charting([season]).write_parquet(ftn_path)

    pbp = pd.read_parquet(pbp_path)
    ftn = pd.read_parquet(ftn_path).rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}
    )[["game_id", "play_id", *FTN_COLS]]
    df = pbp.merge(ftn, on=["game_id", "play_id"], how="left")
    df = df[df["season_type"] == "REG"].copy()

    # Warn if FTN charting has not caught up to the play-by-play yet
    last = df["week"].max()
    db = df[(df["week"] == last) & (df["qb_dropback"] == 1)]
    cov = db["is_interception_worthy"].notna().mean() if len(db) else 1.0
    if cov < 0.9:
        print(f"[{season}] Week {last}: only {cov:.0%} of dropbacks charted by FTN so far. "
              f"Uncharted plays use raw EPA; refresh again in a day or two for full adjustments.")
    return df


def load_seasons(seasons, refresh: bool = False) -> pd.DataFrame:
    return pd.concat([load_season(s, refresh) for s in seasons], ignore_index=True)


# --------------------------------------------------------------------------- #
# Counterfactual value models ("what would this play have been worth if...")
# --------------------------------------------------------------------------- #
SIT = ["down", "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential"]


def _fit(df: pd.DataFrame, mask: pd.Series, feats: list[str]) -> HistGradientBoostingRegressor:
    d = df.loc[mask & df["epa"].notna()]
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=40)
    m.fit(d[feats], d["epa"])
    return m


def fit_value_models(df: pd.DataFrame) -> dict:
    """EPA of an incompletion / completion / interception given the situation."""
    db = df[(df["qb_dropback"] == 1) & df["down"].notna()]
    return {
        "inc": (_fit(db, (db["incomplete_pass"] == 1) & (db["interception"] == 0), SIT), SIT),
        "comp": (_fit(db, db["complete_pass"] == 1, SIT + ["air_yards"]), SIT + ["air_yards"]),
        "int": (_fit(db, db["interception"] == 1, SIT + ["air_yards"]), SIT + ["air_yards"]),
    }


def _predict(models: dict, key: str, df: pd.DataFrame) -> np.ndarray:
    m, feats = models[key]
    return m.predict(df[feats])


# --------------------------------------------------------------------------- #
# Play-level adjustment
# --------------------------------------------------------------------------- #
def adjust_dropbacks(df: pd.DataFrame, models: dict, cfg: dict = CONFIG) -> pd.DataFrame:
    """Return QB dropbacks with raw and fault-adjusted EPA columns."""
    db = df[(df["qb_dropback"] == 1) & df["qb_epa"].notna() & df["id"].notna()
            & df["down"].notna() & (df["qb_spike"] != 1)].copy()
    # Plays FTN has not charted yet (e.g. the latest week) keep their raw EPA: no fault adjustments.
    db["charted"] = db["is_interception_worthy"].notna()
    for c in FTN_BOOL_COLS:
        db[c] = db[c].fillna(False).astype(bool)

    # read_thrown is categorical; strip stray whitespace seen in the raw charting (e.g. " CHK").
    db["read_thrown"] = db["read_thrown"].astype("string").str.strip()
    db["read_charted"] = db["read_thrown"].isin(READ_PROGRESSION_CODES)
    db["past_first_read"] = db["read_thrown"] == "2"
    db["checkdown"] = db["read_thrown"] == "CHK"

    db["garbage"] = ~db["wp"].between(cfg["wp_low"], cfg["wp_high"])
    db["inc_hat"] = _predict(models, "inc", db)
    db["comp_hat"] = _predict(models, "comp", db)
    db["int_hat"] = _predict(models, "int", db)

    # League catch rate of INT-worthy throws, per season (charting standards drift by year)
    worthy = db[db["is_interception_worthy"] & (db["pass_attempt"] == 1)]
    r = worthy.groupby("season")["interception"].mean()
    db["p_int"] = db["season"].map(r)

    raw = db["qb_epa"]
    is_int = db["interception"] == 1
    w = db["is_interception_worthy"]

    # Each adjustment is stored as its own delta so they can be toggled for ablation.
    p, inc, d_int = db["p_int"], db["inc_hat"], pd.Series(0.0, index=db.index)
    # 1) Bad throw, picked: expected value of a bad throw = p*INT + (1-p)*incompletion
    m = is_int & w
    d_int[m] = (p[m] * raw[m] + (1 - p[m]) * inc[m]) - raw[m]
    # 2) Bad throw, defender dropped it: same expected value, using the modeled INT cost
    m = ~is_int & w & (db["pass_attempt"] == 1)
    d_int[m] = (p[m] * db["int_hat"][m] + (1 - p[m]) * raw[m]) - raw[m]
    # 3) INT that was not the QB's fault: score as an incompletion
    m = is_int & ~w
    d_int[m] = inc[m] - raw[m]
    # 4) Receiver drop: credit the expected value of a completion
    m = db["is_drop"] & (db["incomplete_pass"] == 1) & db["air_yards"].notna()
    d_drop = pd.Series(0.0, index=db.index)
    d_drop[m] = db["comp_hat"][m] - raw[m]
    # 5) Sack not charged to the QB: he still eats part of it
    m = (db["sack"] == 1) & ~db["is_qb_fault_sack"]
    d_sack = pd.Series(0.0, index=db.index)
    d_sack[m] = (inc[m] + cfg["nonfault_sack_share"] * (raw[m] - inc[m])) - raw[m]

    # 6) Receiver YAC: swap actual YAC value for expected YAC value on completions
    m = (db["complete_pass"] == 1) & db["xyac_epa"].notna() & db["yac_epa"].notna()
    d_yac = pd.Series(0.0, index=db.index)
    d_yac[m] = db["xyac_epa"][m] - db["yac_epa"][m]

    ch = db["charted"]
    db["d_int"], db["d_drop"], db["d_sack"] = d_int.where(ch, 0.0), d_drop.where(ch, 0.0), d_sack.where(ch, 0.0)
    db["d_yac"] = d_yac
    db["adj_epa"] = recompute_adj(db, cfg["adjust"])
    db["int_bucket"] = np.select(
        [is_int & w, ~is_int & w, is_int & ~w], ["qb_fault_int", "dropped_int", "not_qb_fault_int"], ""
    )
    return db


def recompute_adj(db: pd.DataFrame, adjust: dict) -> pd.Series:
    """Raw EPA plus whichever adjustment deltas are switched on.

    Values in `adjust` are weights: True/1 applies the full adjustment, 0.5 applies half of it.
    """
    out = db["qb_epa"].copy()
    for k in ("int", "drop", "sack", "yac"):
        wt = float(adjust.get(k, 0))
        if wt:
            out = out + wt * db[f"d_{k}"]
    return out


def designed_qb_runs(df: pd.DataFrame, qb_ids: set) -> pd.DataFrame:
    run = df[(df["play_type"] == "run") & (df["qb_dropback"] == 0) & (df["qb_kneel"] != 1)
             & df["rusher_player_id"].isin(qb_ids) & df["epa"].notna()].copy()
    run["garbage"] = ~run["wp"].between(CONFIG["wp_low"], CONFIG["wp_high"])
    return run


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def passer_rating(cmp, att, yds, td, ints):
    a = ((cmp / att) - 0.3) * 5
    b = ((yds / att) - 3) * 0.25
    c = (td / att) * 20
    d = 2.375 - (ints / att) * 25
    return (np.clip(a, 0, 2.375) + np.clip(b, 0, 2.375) + np.clip(c, 0, 2.375) + np.clip(d, 0, 2.375)) / 6 * 100


def qb_table(db: pd.DataFrame, runs: pd.DataFrame | None = None, by=("season", "id")) -> pd.DataFrame:
    """Aggregate dropbacks (and optional designed runs) to one row per QB per group."""
    by = list(by)
    clean = db[~db["garbage"]]
    att = db["pass_attempt"] == 1

    g_all = db.groupby(by)
    g = clean.groupby(by)
    out = pd.DataFrame({
        "name": g_all["name"].agg(lambda s: s.mode().iat[0]),
        "team": g_all["posteam"].agg(lambda s: s.mode().iat[0]),
        "dropbacks": g_all.size(),
        "raw_epa_db": g_all["qb_epa"].mean(),          # the standard public number
        "raw_epa_db_clean": g["qb_epa"].mean(),       # same, garbage time removed
        "adj_epa_db": g["adj_epa"].mean(),            # fault-adjusted, garbage removed
        "cpoe": db[att].groupby(by)["cpoe"].mean(),
        "int_worthy_rate": db[att].groupby(by)["is_interception_worthy"].mean(),
        "fault_sack_rate": g_all["is_qb_fault_sack"].sum() / g_all.size(),
        "total_sack_rate": g_all["sack"].sum() / g_all.size(),
        "success_rate": g["success"].mean(),
        "ints": g_all["interception"].sum().astype(int),
        "qb_fault_ints": g_all["int_bucket"].agg(lambda s: (s == "qb_fault_int").sum()),
        "not_fault_ints": g_all["int_bucket"].agg(lambda s: (s == "not_qb_fault_int").sum()),
        "dropped_ints": g_all["int_bucket"].agg(lambda s: (s == "dropped_int").sum()),
        "drops": g_all["is_drop"].sum().astype(int),
    })

    # Read-progression rates (FTN read_thrown), among charted attempts that got a real read
    # (excludes sacks/scrambles/throwaways/uncharted, which carry no progression information).
    rc = db[db["read_charted"]]
    grc = rc.groupby(by)
    out["past_first_read_rate"] = grc["past_first_read"].mean().reindex(out.index)
    out["checkdown_rate"] = grc["checkdown"].mean().reindex(out.index)
    out["read2_epa_db"] = rc[rc["past_first_read"]].groupby(by)["qb_epa"].mean().reindex(out.index)

    a = db[att & (db["sack"] != 1)]
    ga = a.groupby(by)
    out["passer_rating"] = passer_rating(
        ga["complete_pass"].sum(), ga.size(),
        a[a["complete_pass"] == 1].groupby(by)["yards_gained"].sum().reindex(ga.size().index).fillna(0),
        ga["pass_touchdown"].sum(), ga["interception"].sum(),
    )

    if runs is not None and len(runs):
        r = runs[~runs["garbage"]].drop(columns=["id"], errors="ignore").rename(columns={"rusher_player_id": "id"})
        rush_sum = r.groupby(by)["epa"].sum()
        out["rush_epa_db"] = (rush_sum.reindex(out.index).fillna(0) / out["dropbacks"])
    else:
        out["rush_epa_db"] = 0.0
    return out.reset_index()


def add_grade(tbl: pd.DataFrame, weights: dict = CONFIG["weights"], group="season") -> pd.DataFrame:
    """Composite = weighted sum of within-group z-scores; grade = percentile on a 1-100 scale."""
    t = tbl.copy()
    comp = 0
    for k, w in weights.items():
        z = t.groupby(group)[k].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        comp = comp + w * z.fillna(0)
    t["composite"] = comp
    t["grade"] = (t.groupby(group)["composite"].rank(pct=True) * 99 + 1).round().astype(int)
    return t


def shrink(x: pd.Series, n: pd.Series, k: float) -> pd.Series:
    """Empirical-Bayes style shrinkage toward the (dropback-weighted) league mean."""
    mu = np.average(x, weights=n)
    return (n * x + k * mu) / (n + k)
