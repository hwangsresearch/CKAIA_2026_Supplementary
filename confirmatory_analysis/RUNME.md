# Confirmatory analysis reproduction guide

This package evaluates donor-level late-AD versus non-AD prediction with outcome-label-controlled, fold-specific downstream processing.

## Representation scope

- The released LSI-Harmony embedding and saved COAST embedding are precomputed, outcome-free full-cohort representations.
- They are not retrained within outer folds by the supplied scripts.
- Fold-dependent scalers, covariate processing, topology scaling, and persistence-image parameters are fitted on training donors only.
- The donor is the inferential unit and the binary target is never residualized.

## Environment

For packaged-output validation and figure regeneration:

```bash
python -m venv .venv
# Activate the environment for your platform.
pip install -r confirmatory_analysis/requirements.txt
```

`requirements-full-lock.txt` records the broader analysis environment, including the optional CUDA-enabled PyTorch build used for COAST-related work. PyTorch is not required to inspect the packaged predictions, tables, or figures.

## Required empirical inputs for a full rerun

See `INPUTS_NOTE.txt` and `DATA_PROVENANCE.md`. Expected files include `atac_obs.parquet`, `atac_Xlsi_harmony.npy`, `z_coast_liu.npy`, `donor_pseudobulk.npz`, and topology-diagram artifacts. These large inputs are not included.

The raw topology-generation scripts import the original `phantom.tda` distance and Vietoris-Rips module from the analysis environment. That module is not substituted with a newly written implementation in this public artifact, because doing so could silently change the reported numerical results. The packaged cross-fitted predictions, tables, frozen folds, matched nucleus IDs, and figures remain available for direct audit.

## Packaged-output checks

From the repository root:

```bash
python validate_packaged_results.py
python confirmatory_analysis/src/make_figures.py
```

## Full analysis order, after restoring the required cohort inputs and original `phantom.tda` module

```bash
python confirmatory_analysis/src/phase0_audit.py
python confirmatory_analysis/src/incremental_value.py COAST
python confirmatory_analysis/src/compute_diagrams.py artifacts/liu/atac_Xlsi_harmony.npy lsi
python confirmatory_analysis/src/incremental_value.py LSI
python confirmatory_analysis/src/bootstrap_ci.py COAST 10000
python confirmatory_analysis/src/bootstrap_ci.py LSI 10000
python confirmatory_analysis/src/phase3_matched.py COAST
python confirmatory_analysis/src/phase5_simulation.py
python confirmatory_analysis/src/make_figures.py
```

## Inferential qualification

The bootstrap resamples donors from 20-repeat-averaged cross-fitted predictions. It does not refit the full feature/classification pipeline in each bootstrap replicate. Per-arm intervals are unadjusted across 36 arms. Repeat-level SD describes split instability; repeated folds are not independent inferential replicates.
