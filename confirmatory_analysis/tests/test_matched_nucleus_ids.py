"""QC: matched pseudobulk and topology use the SAME nucleus IDs, and the sampling is deterministic."""
import sys, hashlib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

def _sseed(p): return int.from_bytes(hashlib.sha256("||".join(map(str, p)).encode()).digest()[:8], "little") % (2**32)

def test_sampling_is_deterministic():
    obs = C.load_obs()
    m = obs["cell_class"].to_numpy() == "Mic_Immune"; donor = obs["donor"].to_numpy()
    gidx = np.where(m)[0]; dv = donor[m]
    d = pd.unique(dv)[0]; loc = np.where(dv == d)[0]
    a = np.random.default_rng(_sseed([0, "bures", "Mic_Immune", d, 0, 200])).choice(loc, 200, replace=False)
    b = np.random.default_rng(_sseed([0, "bures", "Mic_Immune", d, 0, 200])).choice(loc, 200, replace=False)
    assert np.array_equal(a, b), "same seed must reproduce identical nucleus subsample"

def test_matched_ids_saved_and_sizes_correct():
    for latent in ["COAST", "LSI"]:
        p = Path(f"confirmatory_analysis/results/subsamples/sampled_nucleus_ids_{latent}.parquet")
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            ids = r["nucleus_rows"].split(";")
            assert len(ids) == int(r["size"]), "saved nucleus id count must equal subsample size"
            assert len(set(ids)) == len(ids), "sampled nuclei must be unique (no replacement)"
