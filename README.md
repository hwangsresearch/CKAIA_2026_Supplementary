# Simple Pseudobulk Outperforms Topological and Quantum-Geometric Readouts of Single-Nucleus Chromatin and Transcriptomes for Donor-Level Alzheimer's Classification: A Rigorous Benchmark in a Powered Multiome Cohort

Supplementary materials and confirmatory-analysis artifact for **CKAIA 2026, Submission 10**.

**Authors:** Sunjun Hwang and Dohyun Hwang

## Contents

- `supplementary_results.pdf`: compiled supplementary document.
- `supplementary_results.tex`: LaTeX source for the supplementary document.
- `confirmatory_analysis/configs/`: frozen analysis configuration.
- `confirmatory_analysis/src/`: analysis and figure-generation scripts.
- `confirmatory_analysis/tests/`: integrity and reproducibility checks.
- `confirmatory_analysis/results/`: cross-fitted predictions, frozen folds, matched nucleus IDs, and machine-readable tables.
- `confirmatory_analysis/figures/`: figures used by the supplement.
- `confirmatory_analysis/reports/`: analysis decisions, provenance map, uncertainty interpretation, and reproduction audit.
- `confirmatory_analysis/RUNME.md`: reproduction instructions and scope.
- `confirmatory_analysis/DATA_PROVENANCE.md`: data-access and provenance notes.
- `BUILD_SUPPLEMENT.sh`: rebuilds the supplementary PDF.
- `validate_packaged_results.py`: checks the packaged headline results without requiring the large cohort arrays.

## Reproduction scope

The archive contains the reported cross-fitted predictions, fold assignments, sampled nucleus IDs, tables, figures, configuration, and analysis scripts. It does **not** redistribute the large Liu et al. cohort arrays. Full empirical reruns require the input files listed in `confirmatory_analysis/INPUTS_NOTE.txt` and access consistent with the source resource's terms.

The released LSI-Harmony embedding and the COAST representation were computed once on the full cohort without pathology labels. Fold-dependent downstream scalers, covariate handling, and persistence-image parameters were fitted on training donors only. The binary late-AD versus non-AD target was never residualized.

The donor-bootstrap intervals condition on fixed 20-repeat-averaged cross-fitted predictions; the full feature/classification pipeline was not refitted inside each bootstrap replicate. Per-arm intervals are unadjusted across 36 arms. These qualifications are documented in the supplement and `confirmatory_analysis/reports/uncertainty_interpretation.md`.

## Quick validation from packaged outputs

```bash
python -m venv .venv
# Activate the environment for your platform.
pip install pandas pyarrow numpy
python validate_packaged_results.py
```

## Rebuild the supplementary PDF

A TeX distribution with `latexmk` and `pdflatex` is required.

```bash
bash BUILD_SUPPLEMENT.sh
```

## Full empirical rerun

See `confirmatory_analysis/RUNME.md`. Large empirical inputs are intentionally excluded. Some raw topology-generation scripts also depend on the original PHANTOM distance/persistence module used in the analysis environment; packaged tables and predictions remain available for verification and figure regeneration.

## Citation

Please cite the accepted CKAIA 2026 paper using the exact title shown above. Publication identifiers can be added to this README after the proceedings record is available.
