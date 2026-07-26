"""QC: frozen outer-fold donor partitions have no train/test overlap. This test does not assess representation-stage transduction."""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

def test_frozen_folds_have_no_train_test_overlap():
    obs = C.load_obs(); meta = C.donor_meta(obs)
    for cell in C.CLASSES:
        donors = C.class_donors(obs, cell, meta)
        if len(donors) < 25:
            continue
        y = meta.loc[donors, "y"].to_numpy().astype(int)
        folds = C.make_folds(y)
        assert len(folds) == 100, "expected 20x5 = 100 folds"
        for rep, fo, tr, te in folds:
            assert set(tr).isdisjoint(set(te)), f"{cell} rep{rep} fold{fo}: train/test overlap"
            assert len(tr) + len(te) == len(donors), "train+test must cover all donors exactly once"

def test_saved_fold_files_partition_donors():
    for f in glob.glob("confirmatory_analysis/results/predictions/folds_*_*.parquet"):
        df = pd.read_parquet(f)
        # each (repeat) must have every donor exactly once across the 5 folds
        for rep, g in df.groupby("repeat"):
            vc = g["donor"].value_counts()
            assert (vc == 1).all(), f"{f} repeat {rep}: donor appears in >1 test fold"
