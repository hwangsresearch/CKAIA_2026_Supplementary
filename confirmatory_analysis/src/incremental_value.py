"""Phase 2 + Phase 4: incremental predictive value of topology.
Models per (latent, cell_class, metric): M0 cov | M1 cov+pseudobulk | M2 cov+topology |
M3 cov+pseudobulk+topology. Same donors and folds are used within each arm. The LSI/COAST
representations are precomputed on the full cohort without outcome labels. Fold-dependent
downstream scaling and persistence-image parameters are fitted on TRAIN donors only.
Donor is the inferential unit. Usage: python incremental_value.py <LSI|COAST>.
"""
from __future__ import annotations
import sys, pickle, warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import common as C

def scaler_fit(X, idx):
    mu = X[idx].mean(0); sd = X[idx].std(0); sd[sd == 0] = 1.0
    return mu, sd
def scaler_apply(X, mu, sd): return (X - mu) / sd
def lr(): return LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                                    max_iter=2000, solver="lbfgs")

def run_latent(latent_name):
    obs = C.load_obs(); meta = C.donor_meta(obs)
    if latent_name == "COAST":
        Z = np.load(C.ART / "z_coast_liu.npy")
    else:
        Z = np.load(C.ART / "atac_Xlsi_harmony.npy")
    with open(C.ART / f"topo_diagrams_{latent_name.lower()}.pkl", "rb") as f:
        DIAG = pickle.load(f)   # (class, metric, donor) -> list of 3 diagrams
    cls = obs["cell_class"].to_numpy(); donor = obs["donor"].to_numpy()
    per_repeat_rows, per_donor_rows = [], []
    t0 = time.time()
    for cell in C.CLASSES:
        donors = C.class_donors(obs, cell, meta)
        if len(donors) < 25: continue
        y = meta.loc[donors, "y"].to_numpy().astype(int)
        cov = meta.loc[donors, C.COVARS].to_numpy(float)   # is_male, age, pmi, logfrag
        m = cls == cell; dv = donor[m]; Zc = Z[m]
        pb = np.vstack([Zc[dv == d].mean(0) for d in donors])   # pseudobulk mean
        folds = C.make_folds(y)
        # save fold test-assignments once per (latent, class) for QC / reuse
        (C.OUT / "predictions").mkdir(parents=True, exist_ok=True)
        frows = [(rep, fo, donors[i]) for rep, fo, tr, te in folds for i in te]
        pd.DataFrame(frows, columns=["repeat", "fold", "donor"]).to_parquet(
            C.OUT / f"predictions/folds_{latent_name}_{cell}.parquet")
        for metric in C.METRICS:
            # per-donor topology inputs: bp points per seed (max_edge already applied at save time) + scalars
            bp_donor = []; scal = []
            for d in donors:
                dl = DIAG[(cell, metric, d)]
                bps = [C.birth_persistence(g) for g in dl]
                bp_donor.append(bps)
                scal.append(np.vstack([C.h1_scalars(g) for g in dl]).mean(0))
            scal = np.vstack(scal)
            n = len(donors)
            oof = {k: np.full((20, n), np.nan) for k in ["M0", "M1", "M2", "M3"]}
            for rep, fo, tr, te in folds:
                # --- fit preprocessing on train only ---
                cmu, csd = scaler_fit(cov, tr); cov_s = scaler_apply(cov, cmu, csd)
                pmu, psd = scaler_fit(pb, tr); pb_s = scaler_apply(pb, pmu, psd)
                # persistence-image params from TRAIN diagrams only
                train_bp = np.vstack([np.vstack(bp_donor[i]) if any(b.size for b in bp_donor[i]) else np.empty((0, 2)) for i in tr]) \
                    if len(tr) else np.empty((0, 2))
                xr, yr, bw = C.fit_pi_params(train_bp if train_bp.size else np.empty((0, 2)))
                PI = np.vstack([C.persistence_image_vec(bp_donor[i], xr, yr, bw, res=12) for i in range(n)])
                topo = np.c_[PI, scal]
                tmu, tsd = scaler_fit(topo, tr); topo_s = scaler_apply(topo, tmu, tsd)
                feats = {"M0": cov_s, "M1": np.c_[cov_s, pb_s],
                         "M2": np.c_[cov_s, topo_s], "M3": np.c_[cov_s, pb_s, topo_s]}
                for k, X in feats.items():
                    clf = lr().fit(X[tr], y[tr])
                    oof[k][rep, te] = clf.predict_proba(X[te])[:, 1]
            # per-repeat metrics
            for k in ["M0", "M1", "M2", "M3"]:
                for rep in range(20):
                    pr = oof[k][rep]
                    au = roc_auc_score(y, pr); ba = balanced_accuracy_score(y, (pr >= 0.5).astype(int))
                    per_repeat_rows.append(dict(latent=latent_name, cell_class=cell, metric=metric,
                                                model=k, repeat=rep, auc=au, balacc=ba, n=n, n_pos=int(y.sum())))
                # per-donor cross-fitted score (mean over repeats)
                cf = np.nanmean(oof[k], 0)
                for i, d in enumerate(donors):
                    per_donor_rows.append(dict(latent=latent_name, cell_class=cell, metric=metric,
                                               model=k, donor=d, y=int(y[i]), score=float(cf[i])))
            print(f"  [{latent_name}/{cell}/{metric}] done {time.time()-t0:.0f}s", flush=True)
    pr_df = pd.DataFrame(per_repeat_rows); pd_df = pd.DataFrame(per_donor_rows)
    (C.OUT / "tables").mkdir(parents=True, exist_ok=True)
    pr_df.to_csv(C.OUT / f"tables/incremental_value_per_repeat_{latent_name}.csv", index=False)
    pd_df.to_parquet(C.OUT / f"predictions/incremental_value_oof_{latent_name}.parquet")
    print(f"saved per-repeat + per-donor for {latent_name}", flush=True)
    return pr_df, pd_df

if __name__ == "__main__":
    run_latent(sys.argv[1] if len(sys.argv) > 1 else "COAST")
