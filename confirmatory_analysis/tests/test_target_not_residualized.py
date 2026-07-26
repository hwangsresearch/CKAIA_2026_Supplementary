"""QC: the binary target (lateAD vs nonAD) is never residualized; labels are the raw {0,1} pathology."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

def test_labels_are_raw_binary():
    obs = C.load_obs(); meta = C.donor_meta(obs)
    labeled = meta[meta["y"].isin([0, 1])]
    assert set(labeled["y"].unique()) <= {0, 1}
    # y==1 iff pathology lateAD; y==0 iff nonAD (never a residual/continuous value)
    assert (labeled.loc[labeled["y"] == 1, "pathology"] == "lateAD").all()
    assert (labeled.loc[labeled["y"] == 0, "pathology"] == "nonAD").all()

def test_oof_labels_are_binary():
    for latent in ["COAST", "LSI"]:
        p = Path(f"confirmatory_analysis/results/predictions/incremental_value_oof_{latent}.parquet")
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        assert set(np.unique(df["y"])) <= {0, 1}, f"{latent}: target must remain binary"
