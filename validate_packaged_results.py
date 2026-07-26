#!/usr/bin/env python3
"""Validate the headline values stored in the public supplementary artifact.

This check uses only packaged machine-readable outputs. It does not rerun feature
construction or model fitting from the large cohort arrays.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "confirmatory_analysis" / "results" / "tables"


def approx(value: float, expected: float, tol: float, label: str) -> None:
    if not math.isfinite(value) or abs(value - expected) > tol:
        raise AssertionError(f"{label}: got {value:.6f}, expected {expected:.6f} ± {tol}")


def main() -> None:
    arm = pd.read_csv(TAB / "arm_level_model_performance.csv")
    assert len(arm) == 36, f"Expected 36 arms, found {len(arm)}"

    delta_auc = arm["dAUC[M3-M1]_mean"]
    delta_bacc = arm["dBACC[M3-M1]_mean"]
    assert int((delta_auc < 0).sum()) == 33, "Expected 33/36 negative M3-M1 AUC estimates"
    assert int((delta_auc > 0).sum()) == 3, "Expected 3/36 positive M3-M1 AUC estimates"
    assert int((arm["dAUC[M3-M1]_lo"] > 0).sum()) == 0, "Unexpected robust positive AUC arm"
    assert int((arm["dAUC[M3-M1]_hi"] < 0).sum()) == 7, "Expected seven descriptive AUC upper<0 arms"
    assert int((arm["dBACC[M3-M1]_lo"] > 0).sum()) == 0, "Unexpected robust positive BAcc arm"
    assert int((arm["dBACC[M3-M1]_hi"] < 0).sum()) == 4, "Expected four descriptive BAcc upper<0 arms"
    approx(float(delta_auc.mean()), -0.094, 0.002, "Mean M3-M1 AUC")
    approx(float(delta_bacc.mean()), -0.074, 0.002, "Mean M3-M1 balanced accuracy")
    approx(float(arm["dAUC[M2-M0]_mean"].mean()), 0.000, 0.002, "Mean M2-M0 AUC")

    matched = pd.read_csv(TAB / "matched_nuclei_full_table.csv")
    m200 = matched[matched["size"].astype(str) == "200"].pivot(
        index="cell_class", columns="repr", values="auc"
    )
    assert len(m200) == 6, f"Expected six cell classes at 200 nuclei, found {len(m200)}"
    assert bool((m200["matched_pseudobulk"] > m200["topology"]).all()), (
        "Expected matched pseudobulk to exceed topology in all six 200-nucleus cell classes"
    )
    approx(float(m200["matched_pseudobulk"].mean()), 0.672, 0.002, "Matched-200 pseudobulk mean AUC")
    approx(float(m200["topology"].mean()), 0.526, 0.002, "Matched-200 topology mean AUC")

    sim = pd.read_csv(TAB / "simulation_results.csv", keep_default_na=False)
    s2b = sim[(sim.scenario == "shape_only") & (sim["dim"] == 2) & (sim.metric == "bures")].iloc[0]
    s2e = sim[(sim.scenario == "shape_only") & (sim["dim"] == 2) & (sim.metric == "euclidean")].iloc[0]
    s32b = sim[(sim.scenario == "shape_only") & (sim["dim"] == 32) & (sim.metric == "bures")].iloc[0]
    s32e = sim[(sim.scenario == "shape_only") & (sim["dim"] == 32) & (sim.metric == "euclidean")].iloc[0]
    approx(float(s2b.AUC_pseudobulk), 0.477, 0.002, "2-D shape-only pseudobulk AUC")
    approx(float(s2b.AUC_topology), 0.986, 0.002, "2-D shape-only Bures H1 AUC")
    approx(float(s2e.AUC_topology), 0.995, 0.002, "2-D shape-only Euclidean H1 AUC")
    approx(float(s32b.AUC_topology), 0.484, 0.002, "32-D shape-only Bures H1 AUC")
    approx(float(s32e.AUC_topology), 0.540, 0.002, "32-D shape-only Euclidean H1 AUC")

    print("PASS: packaged headline results are internally consistent.")
    print("- 36 arms; 33 negative M3-M1 AUC point estimates; 0 robust positive arms")
    print("- Mean M3-M1: AUC ≈ -0.094; balanced accuracy ≈ -0.074")
    print("- Matched 200 nuclei: pseudobulk ≈ 0.672 vs topology ≈ 0.526, 6/6 classes")
    print("- Shape-only controls match the reported 2-D and 32-D values")


if __name__ == "__main__":
    main()
