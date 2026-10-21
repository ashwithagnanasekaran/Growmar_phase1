# Task 3 - Attrition Baseline Report

- Dataset rows: 2389
- Training rows: 1911
- Held-out test rows: 478
- Target: `left_within_12m`
- Cross-validation: stratified five-fold CV on training data
- Preprocessing is fitted independently within each CV fold.

## Cross-validation results

| model                   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |
|:------------------------|---------------:|--------------:|--------------:|-------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|
| Dummy (always stays)    |         0.5000 |        0.0000 |        0.1460 |       0.0010 |           0.0000 |          0.0000 |        0.0000 |       0.0000 |    0.0000 |   0.0000 |
| Logistic Regression     |         0.7068 |        0.0206 |        0.3176 |       0.0525 |           0.2593 |          0.0275 |        0.6061 |       0.0521 |    0.3629 |   0.0350 |
| Decision Tree (depth 4) |         0.6247 |        0.0429 |        0.2259 |       0.0244 |           0.2248 |          0.0370 |        0.4846 |       0.1359 |    0.3017 |   0.0478 |

## Final held-out test evaluation

- Selected model: **Logistic Regression**.
- Selection criterion: highest mean cross-validated PR-AUC among the two supervised models.
- Selected threshold: **0.4922**.
- Threshold selection: out-of-fold training predictions.
- Out-of-fold training precision: 0.2605.
- Out-of-fold training recall: 0.6201.
- Test ROC-AUC: 0.6708.
- Test PR-AUC: 0.3305.
- Test precision: 0.2395.
- Test recall: 0.5714.
- Test F1: 0.3376.

## Confusion matrix

Rows represent actual labels; columns represent predicted labels.
Label order: Stayed (0), Left (1).

```text
[[281 127]
 [ 30  40]]
```

See `baseline_confusion_matrix.png` and `baseline_roc_curve.png`.

## HR interpretation

The model estimates which employees may leave within 12 months. The threshold was selected using out-of-fold training predictions to target at least 60% recall. Higher recall may identify more employees who leave, but may also flag employees who stay.

Predictions indicate patterns, not certainty about individual intentions. HR should review results fairly, protect employee privacy, and never use predictions as the sole basis for employment decisions.

## Limitations

This is a baseline model, not a validated production system. Cross-validation scores summarize performance on training folds. The final test set is held out for evaluation. Future data, fairness, and operational performance should be assessed before using the model in real HR decisions.
