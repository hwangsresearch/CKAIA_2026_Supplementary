"""QC: key primitives are deterministic given the frozen seed, so results reproduce from config+seed."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C

def test_folds_reproducible():
    y = np.array([1]*15 + [0]*45)
    a = C.make_folds(y); b = C.make_folds(y)
    for (r1, f1, tr1, te1), (r2, f2, tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)

def test_persistence_image_deterministic():
    bp = [np.array([[0.1, 0.4], [0.2, 0.7]]), np.array([[0.05, 0.3]])]
    xr, yr, bw = C.fit_pi_params(np.vstack(bp))
    v1 = C.persistence_image_vec(bp, xr, yr, bw)
    v2 = C.persistence_image_vec(bp, xr, yr, bw)
    assert np.allclose(v1, v2) and v1.shape == (144,)

def test_pi_params_fit_only_from_given_points():
    # leakage guard: params depend only on the passed (train) points, not any global state
    train = np.array([[0.0, 0.1], [0.0, 0.5]])
    xr, yr, bw = C.fit_pi_params(train)
    assert yr[1] >= 0.5 - 1e-6 and bw > 0
