"""Backtest helpers: split-half stability, out-of-sample prediction, bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

import qb_eval as q

METRICS = ["raw_epa_db", "raw_epa_db_clean", "adj_epa_db", "cpoe", "passer_rating", "composite"]
LABELS = {
    "raw_epa_db": "Raw EPA/db (baseline)",
    "raw_epa_db_clean": "Raw EPA/db, garbage removed",
    "adj_epa_db": "Fault-adjusted EPA/db",
    "cpoe": "CPOE",
    "passer_rating": "Passer rating",
    "composite": "Composite grade",
}
TARGET = "raw_epa_db"


def split_half_table(db, runs, min_db_half=100, weights=None):
    """QB table per season x half (odd vs even weeks), wide: one row per QB-season.

    `weights` overrides CONFIG["weights"] for the "composite" column (for testing candidate weights).
    """
    db = db.assign(half=np.where(db["week"] % 2 == 1, "odd", "even"))
    runs = runs.assign(half=np.where(runs["week"] % 2 == 1, "odd", "even"))
    t = q.qb_table(db, runs, by=("season", "half", "id"))
    t = t[t["dropbacks"] >= min_db_half]
    t = q.add_grade(t, weights=weights if weights is not None else q.CONFIG["weights"], group=["season", "half"])
    odd = t[t["half"] == "odd"].set_index(["season", "id"])
    even = t[t["half"] == "even"].set_index(["season", "id"])
    return odd.join(even, lsuffix="_odd", rsuffix="_even", how="inner")


def yoy_table(season_tbl, min_db=200):
    """Pairs of (season Y metrics, season Y+1 metrics) for the same QB."""
    t = season_tbl[season_tbl["dropbacks"] >= min_db].set_index(["season", "id"])
    nxt = t.copy()
    nxt.index = pd.MultiIndex.from_arrays([nxt.index.get_level_values(0) - 1, nxt.index.get_level_values(1)],
                                          names=["season", "id"])
    return t.join(nxt, lsuffix="_y", rsuffix="_y1", how="inner")


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def split_half_results(w, metrics=None, labels=None):
    """Average of both directions (odd->even, even->odd)."""
    metrics, labels = metrics or METRICS, labels or LABELS
    rows = []
    for m in metrics:
        pred = (_corr(w[f"{m}_odd"], w[f"{TARGET}_even"]) + _corr(w[f"{m}_even"], w[f"{TARGET}_odd"])) / 2
        stab = _corr(w[f"{m}_odd"], w[f"{m}_even"])
        rows.append({"metric": labels[m], "predicts other-half raw EPA (r)": pred, "self-stability (r)": stab})
    return pd.DataFrame(rows)


def yoy_results(p, metrics=None, labels=None):
    metrics, labels = metrics or METRICS, labels or LABELS
    rows = []
    for m in metrics:
        rows.append({"metric": labels[m],
                     "predicts next-season raw EPA (r)": _corr(p[f"{m}_y"], p[f"{TARGET}_y1"]),
                     "self-stability (r)": _corr(p[f"{m}_y"], p[f"{m}_y1"])})
    return pd.DataFrame(rows)


def bootstrap_diff(x_new, x_base, y, n=5000, seed=0):
    """Bootstrap CI for corr(new, y) - corr(base, y), resampling QB-seasons."""
    rng = np.random.default_rng(seed)
    x_new, x_base, y = map(np.asarray, (x_new, x_base, y))
    idx = np.arange(len(y))
    diffs = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        diffs[i] = np.corrcoef(x_new[s], y[s])[0, 1] - np.corrcoef(x_base[s], y[s])[0, 1]
    return {"diff": _corr(x_new, y) - _corr(x_base, y),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_new_better": float((diffs > 0).mean())}


def bootstrap_split_half(w, new, base="raw_epa_db", **kw):
    """Stack both directions so every QB-season contributes twice (odd->even and even->odd)."""
    x_new = np.r_[w[f"{new}_odd"], w[f"{new}_even"]]
    x_base = np.r_[w[f"{base}_odd"], w[f"{base}_even"]]
    y = np.r_[w[f"{TARGET}_even"], w[f"{TARGET}_odd"]]
    return bootstrap_diff(x_new, x_base, y, **kw)


def shrinkage_k(w, metric, n_col="dropbacks"):
    """k such that shrunk = (n*x + k*mu)/(n+k); from split-half reliability r at average n."""
    r = _corr(w[f"{metric}_odd"], w[f"{metric}_even"])
    n = (w[f"{n_col}_odd"].mean() + w[f"{n_col}_even"].mean()) / 2
    return float(n * (1 - r) / r) if r > 0 else np.inf
