# Uncertainty interpretation — M3−M1 and M2−M0

The analysis uses fixed full-cohort outcome-free LSI/COAST representations and fold-specific downstream transformations. Paired percentile donor-bootstrap intervals (10,000 replicates) are computed from the 20-repeat-averaged cross-fitted scores. They do not include full-pipeline bootstrap refitting and are unadjusted across 36 arms.

## Descriptive summary

- M3−M1 AUC is negative in 33/36 arms; the mean is approximately −0.094.
- M3−M1 balanced accuracy averages approximately −0.074.
- No AUC or balanced-accuracy lower bound is above zero.
- Seven AUC and four balanced-accuracy upper bounds are below zero before multiplicity correction; these are descriptive counts, not familywise discoveries.
- M2−M0 averages approximately 0.000 AUC; no interval excludes zero.
- The most favorable AUC point estimate is +0.110 with interval [−0.040,+0.262], so meaningful improvement is not excluded in every arm.

## Interpretation boundary

The complete H1 block contains 148 features and outer-training sets contain roughly 46–59 donors. Negative M3−M1 values can therefore reflect both representation content and the estimation cost of adding high-variance features. The study supports “no robust operational gain was detected” rather than equivalence or proof of no signal.
