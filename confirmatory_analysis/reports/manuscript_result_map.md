# Manuscript result map

| Manuscript item | Machine-readable source |
|---|---|
| Arm-level M0–M3 AUC/BAcc and percentile donor intervals | `results/tables/performance_with_uncertainty_{COAST,LSI}.csv` |
| M2−M0 and M3−M1 paired differences | same files; `paired_differences_{COAST,LSI}.csv` |
| Main grouped summary | `model_comparison_summary.csv`; `overall_comparison_summary.csv` |
| Repeat-level paired mean/SD/range | `incremental_value_per_repeat_*.csv`; `repeat_level_delta_summary.csv` |
| Cross-fitted donor scores | `results/predictions/incremental_value_oof_*.parquet` |
| Frozen fold assignments | `results/predictions/folds_*_*.parquet` |
| Matched nucleus IDs and performance | `results/subsamples/sampled_nucleus_ids_COAST.parquet`; `subsample_size_sensitivity_COAST.csv`; `matched_nuclei_full_table.csv` |
| Simulation mean±SD and power curve | `simulation_results.csv`; `simulation_summary_with_sd.csv`; `simulation_power_curve.csv` |

Interpretation qualifications are recorded in `analysis_decisions.md`, `uncertainty_interpretation.md`, and the manuscript supplement.
