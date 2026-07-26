"""Shared fold-specific downstream primitives for the confirmatory analysis.
Donor is the inferential unit. LSI/COAST representations are precomputed full-cohort and outcome-free; downstream fitted transformations use TRAIN donors only.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

SEED = 2026
CLASSES = ["Exc", "Inh", "Ast", "Oli", "OPC", "Mic_Immune"]
METRICS = ["euclidean", "cosine", "bures"]
COVARS = ["is_male", "age", "pmi", "logfrag"]
ART = Path("artifacts/liu")
OUT = Path("confirmatory_analysis/results")


def load_obs():
    obs = pd.read_parquet(ART / "atac_obs.parquet")
    obs["logfrag"] = np.log1p(obs["nFrags"].astype(float))
    return obs


def donor_meta(obs):
    """One row per donor: label + covariates (donor-mean logfrag)."""
    g = obs.groupby("donor")
    meta = obs.drop_duplicates("donor").set_index("donor")[["pathology", "sex", "age", "pmi"]].copy()
    meta["logfrag"] = g["logfrag"].mean()
    meta["is_male"] = (meta["sex"] == "Male").astype(float)
    meta["y"] = np.where(meta["pathology"] == "lateAD", 1,
                         np.where(meta["pathology"] == "nonAD", 0, -1))
    return meta


def class_donors(obs, cell_class, meta, min_nuclei=200):
    """Labeled (lateAD/nonAD) donors with >= min_nuclei in the class. Returns ordered donor list."""
    sub = obs[obs["cell_class"] == cell_class]
    counts = sub.groupby("donor").size()
    elig = counts[counts >= min_nuclei].index
    donors = [d for d in elig if meta.loc[d, "y"] in (0, 1)]
    return sorted(donors)


def birth_persistence(dgm, max_edge=None):
    d = np.asarray(dgm, float).reshape(-1, 2)
    if d.size == 0:
        return np.empty((0, 2))
    if max_edge is not None:
        d = d.copy(); d[~np.isfinite(d[:, 1]), 1] = max_edge
    d = d[np.isfinite(d).all(1)]
    return np.c_[d[:, 0], np.maximum(d[:, 1] - d[:, 0], 0.0)]


def h1_scalars(dgm, max_edge=None):
    bp = birth_persistence(dgm, max_edge)
    life = bp[:, 1] if bp.size else np.empty(0)
    life = life[life > 1e-12]
    if life.size == 0:
        return np.zeros(4)
    p = life / life.sum()
    return np.array([life.max(), life.sum(), float(-(p * np.log(p)).sum()), float(life.size)])


def persistence_image_vec(bp_list, xr, yr, bw, res=12):
    """Vectorized persistence image for a list of birth-persistence arrays; averaged over the list.
    xr=(xmin,xmax), yr=(ymin,ymax), bw=bandwidth. Weight = persistence. Fold-specific downstream fitting:
    xr/yr/bw are passed in (fit on TRAIN only). No python loop over diagram points."""
    xs = np.linspace(xr[0], xr[1], res); ys = np.linspace(yr[0], yr[1], res)
    gx, gy = np.meshgrid(xs, ys)  # res x res
    cx = gx.ravel(); cy = gy.ravel()  # (res*res,)
    denom = 2 * bw ** 2 + 1e-12
    acc = np.zeros(res * res); n = 0
    for bp in bp_list:
        n += 1
        if bp.shape[0] == 0:
            continue
        b = bp[:, 0]; p = bp[:, 1]
        pos = p > 0
        if not pos.any():
            continue
        b = b[pos]; p = p[pos]
        d2 = (cx[None, :] - b[:, None]) ** 2 + (cy[None, :] - p[:, None]) ** 2  # (m, G)
        acc += (p[:, None] * np.exp(-d2 / denom)).sum(0)
    return acc / max(n, 1)


def fit_pi_params(bp_train_pooled):
    """im_range + bandwidth from TRAIN birth-persistence points only."""
    if bp_train_pooled.shape[0] == 0:
        return (0.0, 1.0), (0.0, 1.0), 1e-2
    b = bp_train_pooled[:, 0]; p = bp_train_pooled[:, 1]
    xr = (float(b.min()), float(b.max()) + 1e-9)
    yr = (0.0, float(p.max()) + 1e-9)
    pp = p[p > 0]
    bw = max(float(np.median(pp)) * 0.5, 1e-3) if pp.size else 1e-2
    return xr, yr, bw


def make_folds(y, n_splits=5, n_repeats=20, seed=SEED):
    """Deterministic RepeatedStratifiedKFold assignments; returns list of (repeat,fold,train_idx,test_idx)."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        out.append((i // n_splits, i % n_splits, tr, te))
    return out
