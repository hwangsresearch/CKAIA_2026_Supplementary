"""Generate confirmatory-analysis figures from result tables (whatever exists)."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import common as C
TAB = C.OUT / "tables"; FIG = Path("confirmatory_analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})

def forest_delta(metric_key, fname, title):
    frames = []
    for latent in ["COAST", "LSI"]:
        p = TAB / f"performance_with_uncertainty_{latent}.csv"
        if p.exists(): frames.append(pd.read_csv(p))
    if not frames: return
    df = pd.concat(frames, ignore_index=True)
    df["lab"] = df["latent"] + " " + df["cell_class"] + " " + df["metric"]
    df = df.sort_values(f"{metric_key}_mean")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28*len(df))))
    y = np.arange(len(df))
    ax.errorbar(df[f"{metric_key}_mean"], y,
                xerr=[df[f"{metric_key}_mean"]-df[f"{metric_key}_lo"], df[f"{metric_key}_hi"]-df[f"{metric_key}_mean"]],
                fmt="o", color="#8e44ad", ecolor="gray", capsize=2, ms=4)
    ax.axvline(0, ls="--", color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(df["lab"], fontsize=6)
    ax.set_xlabel(f"{metric_key}  (topology added on top of covariates+pseudobulk)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)
    print("wrote", fname)

def subsample_fig(metric_col, fname, ylab):
    p = TAB / "subsample_size_sensitivity_COAST.csv"
    if not p.exists(): return
    df = pd.read_csv(p); df = df[df["size"] != "all"]; df["size"] = df["size"].astype(int)
    g = df.groupby(["size", "repr"])[metric_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    for rep, sub in g.groupby("repr"):
        ax.plot(sub["size"], sub[metric_col], "o-", label=rep)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("nuclei per donor (matched)"); ax.set_ylabel(ylab)
    ax.set_title("Matched pseudobulk vs topology"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)
    print("wrote", fname)

def sim_scenario_fig():
    p = TAB / "simulation_results.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    d = df[df["dim"] == 2] if "dim" in df else df
    order = ["null", "mean_shift", "shape_only", "depth_confound"]
    d = d.drop_duplicates("scenario").set_index("scenario").reindex(order).dropna(how="all")
    x = np.arange(len(d)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x-w/2, d["AUC_pseudobulk"], w, label="pseudobulk", color="#27ae60")
    ax.bar(x+w/2, d["AUC_topology"], w, label="topology (Bures)", color="#8e44ad")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=20)
    ax.set_ylabel("AUC"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Simulation: pseudobulk vs topology by scenario (dim=2)")
    fig.tight_layout(); fig.savefig(FIG / "simulation_auc_by_scenario.pdf"); plt.close(fig)
    print("wrote simulation_auc_by_scenario.pdf")

def sim_power_fig():
    p = TAB / "simulation_power_curve.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5, 4))
    for metric, sub in df.groupby("metric"):
        ax.plot(sub["eff"], sub["AUC_topology"], "o-", label=f"topology {metric}")
    pb = df.groupby("eff")["AUC_pseudobulk"].mean()
    ax.plot(pb.index, pb.values, "s--", color="#27ae60", label="pseudobulk")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("shape-effect strength"); ax.set_ylabel("AUC")
    ax.set_title("Shape-only power curve (dim=2)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "simulation_power_curve.pdf"); plt.close(fig)
    print("wrote simulation_power_curve.pdf")

if __name__ == "__main__":
    forest_delta("dAUC[M3-M1]", "incremental_delta_auc_forest.pdf",
                 "Incremental value of topology: ΔAUC (M3−M1), 95% donor-bootstrap CI")
    forest_delta("dBACC[M3-M1]", "incremental_delta_balanced_accuracy_forest.pdf",
                 "Incremental value of topology: Δbalanced accuracy (M3−M1), 95% CI")
    subsample_fig("auc", "auc_vs_subsample_size.pdf", "AUC (repeated CV)")
    subsample_fig("balacc", "balanced_accuracy_vs_subsample_size.pdf", "balanced accuracy")
    sim_scenario_fig(); sim_power_fig()
    print("figures done")
