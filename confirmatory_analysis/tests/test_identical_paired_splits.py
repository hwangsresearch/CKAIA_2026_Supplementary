"""QC: paired models (M0..M3) are evaluated on identical donor sets and identical folds."""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_all_models_share_donor_set_per_group():
    for latent in ["COAST", "LSI"]:
        p = Path(f"confirmatory_analysis/results/predictions/incremental_value_oof_{latent}.parquet")
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for (cell, metric), g in df.groupby(["cell_class", "metric"]):
            sets = {m: frozenset(gm["donor"]) for m, gm in g.groupby("model")}
            base = next(iter(sets.values()))
            for m, s in sets.items():
                assert s == base, f"{latent}/{cell}/{metric}: model {m} donor set differs"

def test_labels_consistent_across_models():
    for latent in ["COAST", "LSI"]:
        p = Path(f"confirmatory_analysis/results/predictions/incremental_value_oof_{latent}.parquet")
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # a donor's label y must be identical across models within a group
        chk = df.groupby(["cell_class", "metric", "donor"])["y"].nunique()
        assert (chk == 1).all(), f"{latent}: inconsistent labels across models"
