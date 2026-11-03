# Task 13 – Attrition Model V2

## Method
- Same 80/20 stratified split and random seed as Phase 1.
- Five-fold stratified CV on the training split.
- Fold-specific preprocessing to prevent validation leakage.
- Model selection based on mean PR-AUC.
- Threshold selected from training out-of-fold predictions to target at least 60% recall.

## Selected model
- Configuration: XGBoost_A
- Parameters: `{'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05}`
- Decision threshold: 0.1562

## Held-out test results
- roc_auc: 0.6792717086834734
- pr_auc: 0.32893423993288906
- precision: 0.24242424242424243
- recall: 0.5714285714285714
- f1: 0.3404255319148936
- brier_score: 0.11594473570585251
- model: Selected XGBoost
- configuration: XGBoost_A
- threshold: 0.15621376037597656
- Test Brier score: 0.1159
- Calibration was assessed on the held-out test set. No post-hoc calibration was applied in this run.

## Important limitations
- The model is evaluated on one stratified held-out split.
- The workbook benchmark values are reference results, not a replacement for same-fold comparisons.
- A higher ranking metric does not establish fairness or causal validity.
- Predictions require human review and must not be used as automatic employment decisions.

## Evidence files
- `task13_cv_fold_metrics.csv`
- `task13_cv_summary.csv`
- `task13_tuning_log.csv`
- `task13_test_metrics.csv`
- `task13_benchmark_comparison.csv`
- `task13_calibration_curve.png`
## Calibration assessment

The held-out test Brier score was 0.11594, compared with 0.12500 for a constant-prevalence predictor. This is a 7.25% improvement over the constant baseline.

Calibration was assessed using eight quantile-based probability groups. The highest group had a mean predicted probability of 38.38% and an observed leaving rate of 38.33%. Some middle groups showed larger differences, including a 9.11 percentage-point gap in the seventh group.

Conclusion: The model performs better than the constant-prevalence probability baseline, but its probabilities are not perfectly calibrated. No post-hoc calibration was applied. Individual predictions must not be interpreted as certainty.

Calibration evidence: `task13_calibration_bins.csv`.
