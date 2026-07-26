"""Phase 3: matched-nucleus pseudobulk vs topology + subsample-size sensitivity.
For each donor/class/size/seed we sample ONE set of nucleus IDs and use the SAME nuclei for both
the pseudobulk mean and the topology (bures VR-H1) features, so the comparison is fully matched.
Sizes preserve the n>=200 donor set: [50,100,200]. Also all-nuclei pseudobulk. Fold-specific downstream
repeated 20x5 CV (PI fit on TRAIN donors only). Saves sampled nucleus IDs for audit.
Usage: python phase3_matched.py <LSI|COAST>
"""
import sys, pickle, hashlib, warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, str(Path(__file__).parent))
from phantom.tda.distances import pairwise_distance
from phantom.tda.vr import rips_persistence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import common as C
SIZES = [50, 100, 200]; NSEEDS = 3; METRIC = "bures"
def sseed(p): return int.from_bytes(hashlib.sha256("||".join(map(str, p)).encode()).digest()[:8], "little") % (2**32)

def topo_bp_scal(Xsub):
    D = pairwise_distance(Xsub.astype(np.float64), METRIC, knn=15)
    off = D[np.triu_indices_from(D, 1)]; me = float(np.quantile(off, 0.9)) if off.size else 1.0
    g = rips_persistence(D, maxdim=1, max_edge=me).get(1, np.empty((0, 2)))
    g = np.asarray(g, float).reshape(-1, 2)
    if g.size: g[~np.isfinite(g[:, 1]), 1] = me
    return C.birth_persistence(g), C.h1_scalars(g)

def cv_eval(feat_kind, donors, y, pb=None, bp_donor=None, scal=None):
    """repeated fold-specific-downstream 20x5 CV AUC/balacc. feat_kind in {'pb','topo'}."""
    folds = C.make_folds(y); n = len(donors); oof = np.full((20, n), np.nan)
    for rep, fo, tr, te in folds:
        if feat_kind == "pb":
            mu = pb[tr].mean(0); sd = pb[tr].std(0); sd[sd == 0] = 1; X = (pb - mu)/sd
        else:
            tb = np.vstack([np.vstack(bp_donor[i]) for i in tr if any(b.size for b in bp_donor[i])]) \
                 if any(any(b.size for b in bp_donor[i]) for i in tr) else np.empty((0, 2))
            xr, yr, bw = C.fit_pi_params(tb)
            PI = np.vstack([C.persistence_image_vec(bp_donor[i], xr, yr, bw) for i in range(n)])
            topo = np.c_[PI, scal]; mu = topo[tr].mean(0); sd = topo[tr].std(0); sd[sd == 0] = 1; X = (topo - mu)/sd
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr])
        oof[rep, te] = clf.predict_proba(X[te])[:, 1]
    aus = [roc_auc_score(y, oof[r]) for r in range(20)]
    bas = [balanced_accuracy_score(y, (oof[r] >= 0.5).astype(int)) for r in range(20)]
    return float(np.mean(aus)), float(np.mean(bas))

def main(latent):
    obs = C.load_obs(); meta = C.donor_meta(obs)
    Z = np.load(C.ART / ("z_coast_liu.npy" if latent == "COAST" else "atac_Xlsi_harmony.npy"))
    cls = obs["cell_class"].to_numpy(); donor = obs["donor"].to_numpy()
    rows = []; nuc_audit = []
    t0 = time.time()
    for cell in C.CLASSES:
        donors = C.class_donors(obs, cell, meta)
        if len(donors) < 25: continue
        y = meta.loc[donors, "y"].to_numpy().astype(int)
        m = cls == cell; dv = donor[m]; Zc = Z[m]; gidx = np.where(m)[0]
        # all-nuclei pseudobulk
        pb_all = np.vstack([Zc[dv == d].mean(0) for d in donors])
        au, ba = cv_eval("pb", donors, y, pb=pb_all)
        rows.append(dict(latent=latent, cell_class=cell, size="all", repr="pseudobulk", auc=au, balacc=ba, n=len(donors)))
        for size in SIZES:
            # matched sampling: same nuclei for pb and topology, 3 seeds
            pb_m = np.zeros((len(donors), Z.shape[1])); bp_donor = []; scal = np.zeros((len(donors), 4))
            for di, d in enumerate(donors):
                loc = np.where(dv == d)[0]
                pbs = []; bps = []; scs = []
                for s in range(NSEEDS):
                    rng = np.random.default_rng(sseed([0, METRIC, cell, d, s, size]))
                    ch = rng.choice(loc, size, replace=False)
                    nuc_audit.append(dict(latent=latent, cell_class=cell, donor=d, size=size, seed=s,
                                          nucleus_rows=";".join(map(str, gidx[ch].tolist()))))
                    Xs = Zc[ch]
                    pbs.append(Xs.mean(0))
                    bp, sc = topo_bp_scal(Xs); bps.append(bp); scs.append(sc)
                pb_m[di] = np.mean(pbs, 0); bp_donor.append(bps); scal[di] = np.mean(scs, 0)
            au_pb, ba_pb = cv_eval("pb", donors, y, pb=pb_m)
            au_tp, ba_tp = cv_eval("topo", donors, y, bp_donor=bp_donor, scal=scal)
            rows.append(dict(latent=latent, cell_class=cell, size=size, repr="matched_pseudobulk", auc=au_pb, balacc=ba_pb, n=len(donors)))
            rows.append(dict(latent=latent, cell_class=cell, size=size, repr="topology", auc=au_tp, balacc=ba_tp, n=len(donors)))
            print(f"  [{latent}/{cell}/size{size}] matched-pb AUC={au_pb:.3f} topo AUC={au_tp:.3f} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows); df.to_csv(C.OUT / f"tables/subsample_size_sensitivity_{latent}.csv", index=False)
    # matched-200 summary
    m200 = df[(df["size"] == 200)].pivot_table(index="cell_class", columns="repr", values="auc")
    m200.to_csv(C.OUT / f"tables/matched_nuclei_summary_{latent}.csv")
    pd.DataFrame(nuc_audit).to_parquet(C.OUT / f"subsamples/sampled_nucleus_ids_{latent}.parquet")
    pd.set_option("display.width", 200)
    print(f"\n=== {latent} matched-200 (AUC): pseudobulk vs topology ===")
    print(m200.round(3).to_string())
    print("\nsize sensitivity (mean AUC across classes):")
    print(df.groupby(["size", "repr"])["auc"].mean().round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "COAST")
