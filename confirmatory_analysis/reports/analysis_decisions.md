# Analysis decisions and scope

## Frozen protocol

- Repeated stratified 5-fold CV × 20 repeats; global seed 2026.
- L2 logistic regression (`C=1`, balanced class weights), AUC primary, balanced accuracy secondary.
- Donor is the inferential unit; folds are not treated as independent inferential replicates.

## Representation versus downstream fitting

- LSI–Harmony and COAST are precomputed, outcome-free full-cohort representations. They were not fold-wise retrained.
- Fold-dependent downstream scalers, covariate processing, topology scaling, and persistence-image range/bandwidth were fitted using outer-training donors.
- The exploratory globally fitted persistence-image parameters were corrected.
- The binary target was never residualized. A prior outside-CV post-hoc residualized microglia value is not retained as confirmatory evidence.

## Models and scope

M0=covariates, M1=covariates+pseudobulk, M2=covariates+topology, M3=all features. Identical donor sets and folds are used within each arm. Region composition is not separately modeled. The matched analysis uses COAST–Bures standalone readouts from identical nucleus IDs; it does not estimate matched M3−M1.

## Uncertainty

Percentile, paired, class-stratified donor-bootstrap intervals use 10,000 resamples of fixed 20-repeat-averaged cross-fitted scores. The feature and classifier pipeline is not refitted within each bootstrap replicate. Repeat-level SD is separate split-instability information. Per-arm intervals are unadjusted across 36 arms.

## Interpretation

Adding 148 topology features to a small donor training set can increase variance. The M3−M1 result is operational performance of the full readout, not proof of absent signal. Simulations are detectability controls; the 32-D failure leaves pipeline power as an alternative explanation for empirical negative results.
