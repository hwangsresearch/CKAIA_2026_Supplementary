"""Phase 0: data manifest, reproduction audit, and preprocessing-scope flags."""
import sys, hashlib, warnings, numpy as np, pandas as pd, pickle
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

REP = Path("confirmatory_analysis/reports"); REP.mkdir(parents=True, exist_ok=True)

def sha1(p, cap=50_000_000):
    h = hashlib.sha1(); n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b); n += len(b)
            if n >= cap: h.update(b"...capped"); break
    return h.hexdigest()[:12]

files = {
    "artifacts/liu/atac_obs.parquet": "ATAC per-nucleus metadata (donor/class/region/pathology/covars/QC)",
    "artifacts/liu/atac_Xlsi_harmony.npy": "LSI-Harmony cell embedding (49d)",
    "artifacts/liu/z_coast_liu.npy": "COAST cell embedding (32d)",
    "artifacts/liu/donor_pseudobulk.npz": "RNA donor pseudobulk (counts) + donor meta",
    "artifacts/liu/topo_diagrams_coast.pkl": "COAST raw H1 diagrams per (class,metric,donor)",
    "artifacts/liu/topo_fb_coast.pkl": "COAST topology feature blocks (PI+scalars) + pseudobulk",
    "artifacts/liu/benchmark_repeatedCV.csv": "manuscript repeated-CV benchmark table",
    "artifacts/liu/ccre_keep_idx.npy": "36,731 informative cCRE indices for COAST",
    "docs/MANUSCRIPT.md": "manuscript source (markdown)",
    "topoad_manuscript.pdf": "compiled manuscript PDF",
}
rows = []
for p, purpose in files.items():
    pp = Path(p)
    if not pp.exists():
        rows.append(dict(path=p, exists=False, purpose=purpose)); continue
    r = dict(path=p, exists=True, type=pp.suffix, size_mb=round(pp.stat().st_size / 1e6, 2),
             sha1_12=sha1(pp), purpose=purpose)
    try:
        if pp.suffix == ".parquet":
            d = pd.read_parquet(pp); r["shape"] = str(d.shape); r["cols"] = ",".join(map(str, d.columns[:12]))
        elif pp.suffix == ".npy":
            a = np.load(pp, mmap_mode="r"); r["shape"] = str(a.shape)
        elif pp.suffix == ".npz":
            z = np.load(pp, allow_pickle=True); r["cols"] = ",".join(z.keys())
        elif pp.suffix == ".pkl":
            with open(pp, "rb") as f: obj = pickle.load(f)
            r["shape"] = f"{len(obj)} keys"
        elif pp.suffix == ".csv":
            d = pd.read_csv(pp); r["shape"] = str(d.shape); r["cols"] = ",".join(map(str, d.columns[:12]))
    except Exception as e:
        r["note"] = f"read_err:{e}"
    rows.append(r)
man = pd.DataFrame(rows)
man.to_csv(REP / "data_manifest.csv", index=False)
print("wrote data_manifest.csv:", man["exists"].sum(), "of", len(man), "present")

# reproduction audit: donor counts + headline pseudobulk numbers
obs = C.load_obs(); meta = C.donor_meta(obs)
from collections import Counter
dpath = Counter(meta["pathology"])
lines = ["# Reproduction audit\n",
         f"- Donors total: {meta.shape[0]} (expected 111): {'OK' if meta.shape[0]==111 else 'MISMATCH'}",
         f"- Donor Pathology: {dict(dpath)} (expected nonAD 57 / earlyAD 33 / lateAD 21)",
         f"- ATAC nuclei: {len(obs)} (expected 1,217,165): {'OK' if len(obs)==1217165 else 'MISMATCH'}",
         f"- Cell classes analysed: {C.CLASSES} (Vasc excluded)\n"]
# per-class labeled donor counts with >=200 nuclei
lines.append("## Labeled donors (lateAD/nonAD) with >=200 nuclei, per class")
for cell in C.CLASSES:
    dd = C.class_donors(obs, cell, meta); y = meta.loc[dd, "y"]
    lines.append(f"- {cell}: n={len(dd)} (lateAD {int((y==1).sum())}, nonAD {int((y==0).sum())})")
lines += ["", "## Preprocessing and representation-scope audit",
          "- Exploratory persistence-image range/bandwidth was fitted globally. The confirmatory downstream implementation fits those parameters on outer-training diagrams.",
          "- The prior outside-CV post-hoc residualized value is not retained. The binary target is unchanged and covariates enter the fold-specific downstream model.",
          "- Original single-split evaluation is superseded by repeated 20x5 CV plus donor-resampling intervals.",
          "- The released LSI-Harmony and saved COAST representations were computed once on the full cohort without outcome labels; they are not fold-wise retrained.",
          "- Target (lateAD vs nonAD) is NEVER residualized in any arm."]
(REP / "reproduction_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote reproduction_audit.md")
print("\n".join(lines))
