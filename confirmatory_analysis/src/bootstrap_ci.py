"""Phase 4: donor-level uncertainty. From per-donor cross-fitted scores (mean OOF proba over
20 repeats), compute paired, class-stratified donor bootstrap (10,000 reps) 95% CIs for each
model's AUC/balacc and for paired differences (M3-M1, M2-M0, M1-M0). Resample DONORS, not folds.
Also summarize repeat-level mean/SD (split instability) from per-repeat table.
Usage: python bootstrap_ci.py <LSI|COAST> [nboot]
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import common as C

def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
def bacc(y, s):
    return balanced_accuracy_score(y, (s >= 0.5).astype(int))

def boot_ci(y, score_by_model, pairs, nboot=10000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    stat = {f"AUC[{m}]": [] for m in score_by_model}
    stat.update({f"BACC[{m}]": [] for m in score_by_model})
    for a, b in pairs:
        stat[f"dAUC[{a}-{b}]"] = []; stat[f"dBACC[{a}-{b}]"] = []
    for _ in range(nboot):
        bp = rng.choice(pos, len(pos), replace=True); bn = rng.choice(neg, len(neg), replace=True)
        bi = np.concatenate([bp, bn]); yb = y[bi]
        au = {}; ba = {}
        for m, s in score_by_model.items():
            sb = s[bi]; au[m] = auc(yb, sb); ba[m] = bacc(yb, sb)
            stat[f"AUC[{m}]"].append(au[m]); stat[f"BACC[{m}]"].append(ba[m])
        for a, b in pairs:
            stat[f"dAUC[{a}-{b}]"].append(au[a] - au[b]); stat[f"dBACC[{a}-{b}]"].append(ba[a] - ba[b])
    rows = {}
    for k, v in stat.items():
        v = np.array(v); rows[k] = (np.nanmean(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
    return rows

def main(latent, nboot=10000):
    od = C.OUT / f"predictions/incremental_value_oof_{latent}.parquet"
    df = pd.read_parquet(od)
    pr = pd.read_csv(C.OUT / f"tables/incremental_value_per_repeat_{latent}.csv")
    pairs = [("M3", "M1"), ("M2", "M0"), ("M1", "M0"), ("M3", "M2")]
    rows = []
    for (cell, metric), g in df.groupby(["cell_class", "metric"]):
        piv = g.pivot_table(index="donor", columns="model", values="score")
        yv = g.drop_duplicates("donor").set_index("donor")["y"].reindex(piv.index).to_numpy().astype(int)
        sbm = {m: piv[m].to_numpy() for m in ["M0", "M1", "M2", "M3"]}
        ci = boot_ci(yv, sbm, pairs, nboot=nboot)
        # repeat-level mean/sd
        rl = pr[(pr.cell_class == cell) & (pr.metric == metric)].groupby("model")["auc"].agg(["mean", "std"])
        row = dict(latent=latent, cell_class=cell, metric=metric, n=len(yv), n_pos=int(yv.sum()))
        for m in ["M0", "M1", "M2", "M3"]:
            row[f"AUC_{m}"] = rl.loc[m, "mean"]; row[f"AUC_{m}_repSD"] = rl.loc[m, "std"]
        for k, (mn, lo, hi) in ci.items():
            row[f"{k}_mean"] = mn; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / f"tables/performance_with_uncertainty_{latent}.csv", index=False)
    # focused paired-difference table
    keep = ["latent", "cell_class", "metric", "n", "n_pos",
            "dAUC[M3-M1]_mean", "dAUC[M3-M1]_lo", "dAUC[M3-M1]_hi",
            "dBACC[M3-M1]_mean", "dBACC[M3-M1]_lo", "dBACC[M3-M1]_hi",
            "AUC_M1", "AUC_M3"]
    out[keep].to_csv(C.OUT / f"tables/paired_differences_{latent}.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(f"===== {latent}: dAUC(M3-M1) with 95% donor-bootstrap CI =====")
    print(out[keep].round(3).to_string(index=False))
    print(f"\nM3-M1 upper CI max across arms: {out['dAUC[M3-M1]_hi'].max():.3f}")
    return out

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "COAST",
         int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
