"""Compute + save raw H1 persistence diagrams per (cell_class, metric, donor) for a latent.
Deterministic nucleus subsampling (SHA-256 seeds) so matched-nucleus analysis is reproducible.
Saves artifacts/liu/topo_diagrams_<tag>.pkl and the exact sampled nucleus indices.
Usage: python compute_diagrams.py <latent_npy> <tag>
"""
import sys, pickle, hashlib, warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, "src")                       # phantom package
sys.path.insert(0, str(Path(__file__).parent))  # common.py
from phantom.tda.distances import pairwise_distance
from phantom.tda.vr import rips_persistence
import common as C

LATENT = sys.argv[1]; TAG = sys.argv[2]
SUBN = 200; NSEEDS = 3; KNN = 15; QEDGE = 0.9; SEED = 0
def sseed(parts): return int.from_bytes(hashlib.sha256("||".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32)

obs = C.load_obs(); meta = C.donor_meta(obs)
Z = np.load(LATENT)
cls = obs["cell_class"].to_numpy(); donor = obs["donor"].to_numpy()
diagrams = {}; nucids = {}
t0 = time.time()
for cell in C.CLASSES:
    m = cls == cell; dv = donor[m]; Zc = Z[m]
    idx_global = np.where(m)[0]
    udon = [d for d in pd.unique(dv) if (dv == d).sum() >= SUBN]
    for metric in C.METRICS:
        for d in udon:
            loc = np.where(dv == d)[0]
            dl = []; ids = []
            for s in range(NSEEDS):
                rng = np.random.default_rng(sseed([SEED, metric, cell, d, s]))
                ch = rng.choice(loc, SUBN, replace=False)
                ids.append(idx_global[ch])  # global nucleus row indices
                D = pairwise_distance(Zc[ch].astype(np.float64), metric, knn=KNN)
                off = D[np.triu_indices_from(D, 1)]; me = float(np.quantile(off, QEDGE)) if off.size else 1.0
                g = rips_persistence(D, maxdim=1, max_edge=me).get(1, np.empty((0, 2)))
                g = np.asarray(g, float).reshape(-1, 2)
                if g.size: g[~np.isfinite(g[:, 1]), 1] = me
                dl.append(g)
            diagrams[(cell, metric, d)] = dl
            nucids[(cell, metric, d)] = ids
        print(f"  [{TAG}/{cell}/{metric}] {time.time()-t0:.0f}s", flush=True)
with open(C.ART / f"topo_diagrams_{TAG}.pkl", "wb") as f: pickle.dump(diagrams, f)
with open(C.ART / f"topo_nucids_{TAG}.pkl", "wb") as f: pickle.dump(nucids, f)
print(f"saved topo_diagrams_{TAG}.pkl ({len(diagrams)} keys) + nucids", flush=True)
