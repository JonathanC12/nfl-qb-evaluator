"""
Play-level "mixed model" for QB skill, fit as ridge regression on sparse one-hot effects.

    EPA_play = intercept + home + QB + opposing defense + offense (non-QB context) + noise

Ridge with a separate penalty per group is the same as a Gaussian random-effects model with
known variance ratios: penalty_g = noise variance / true-effect variance of group g. That ratio is
exactly the shrinkage constant k (in dropbacks) from the split-half reliability, so each group's
penalty comes from the data rather than from tuning against the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge


def _onehot(codes: np.ndarray, n_levels: int) -> sp.csr_matrix:
    rows = np.arange(len(codes))
    return sp.csr_matrix((np.ones(len(codes)), (rows, codes)), shape=(len(codes), n_levels))


def fit_effects(plays: pd.DataFrame, y_col: str, penalties: dict[str, float]) -> dict[str, pd.Series]:
    """
    plays: one row per dropback with columns id (QB), defteam, posteam, home_team.
    penalties: {"qb": k_qb, "def": k_def, "off": k_off}; omit a key to leave that group out.
    Returns each group's estimated effect (EPA per dropback vs. average).
    """
    groups = {"qb": "id", "def": "defteam", "off": "posteam"}
    blocks, meta = [], []
    for g, col in groups.items():
        if g not in penalties:
            continue
        cat = pd.Categorical(plays[col])
        # Scaling a column by 1/sqrt(k) turns one shared ridge alpha=1 into per-group penalty k.
        scale = 1.0 / np.sqrt(penalties[g])
        blocks.append(_onehot(cat.codes, len(cat.categories)) * scale)
        meta.append((g, cat.categories, scale))
    home = (plays["posteam"] == plays["home_team"]).astype(float).to_numpy()[:, None]
    X = sp.hstack(blocks + [sp.csr_matrix(home * 1e3)]).tocsr()   # home effect ~unpenalized

    model = Ridge(alpha=1.0, fit_intercept=True, solver="sparse_cg", max_iter=5000, tol=1e-6)
    model.fit(X, plays[y_col].to_numpy())

    out, start = {}, 0
    for g, cats, scale in meta:
        coef = model.coef_[start:start + len(cats)] * scale
        out[g] = pd.Series(coef, index=cats, name=f"{g}_effect")
        start += len(cats)
    out["home"] = float(model.coef_[-1] * 1e3)
    return out


def group_k(db: pd.DataFrame, col: str, y_col: str = "qb_epa", min_n: int = 100) -> float:
    """Shrinkage constant k for any grouping (e.g. defenses) from odd/even-week reliability."""
    d = db.assign(half=np.where(db["week"] % 2 == 1, "odd", "even"))
    g = d.groupby(["season", col, "half"])[y_col].agg(["mean", "size"]).unstack("half")
    g = g[(g[("size", "odd")] >= min_n) & (g[("size", "even")] >= min_n)]
    r = np.corrcoef(g[("mean", "odd")], g[("mean", "even")])[0, 1]
    n = g["size"].mean().mean()
    return float(n * (1 - r) / r) if r > 0 else np.inf


def split_half_effects(db, y_col, penalties):
    """QB effects fit on odd weeks and on even weeks of every season (wide, one row per QB-season)."""
    rows = []
    for (season, half), d in db.assign(half=np.where(db["week"] % 2 == 1, "odd", "even")).groupby(["season", "half"]):
        eff = fit_effects(d, y_col, penalties)["qb"]
        rows.append(pd.DataFrame({"season": season, "half": half, "id": eff.index, "effect": eff.values}))
    e = pd.concat(rows).pivot_table(index=["season", "id"], columns="half", values="effect")
    return e.rename(columns={"odd": "effect_odd", "even": "effect_even"})


def season_effects(db, y_col, penalties):
    rows = []
    for season, d in db.groupby("season"):
        eff = fit_effects(d, y_col, penalties)
        rows.append(pd.DataFrame({"season": season, "id": eff["qb"].index, "effect": eff["qb"].values}))
    return pd.concat(rows).set_index(["season", "id"])
