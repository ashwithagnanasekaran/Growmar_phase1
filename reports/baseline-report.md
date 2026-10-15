# Task 3 - Attrition Baseline Report

- Dataset rows: 2389
- Training rows: 1911
- Held-out test rows: 478
- Target: `left_within_12m`
- Cross-validation: stratified five-fold CV on training data

## Cross-validation results

| model                   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |
|:------------------------|---------------:|--------------:|--------------:|-------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|
| Dummy (always stays)    |         0.5000 |        0.0000 |        0.1460 |       0.0010 |           0.0000 |          0.0000 |        0.0000 |       0.0000 |    0.0000 |   0.0000 |
| Logistic Regression     |         0.7125 |        0.0173 |        0.3175 |       0.0468 |           0.2612 |          0.0224 |        0.5988 |       0.0309 |    0.3633 |   0.0256 |
| Decision Tree (depth 4) |         0.6247 |        0.0429 |        0.2259 |       0.0244 |           0.2248 |          0.0370 |        0.4846 |       0.1359 |    0.3017 |   0.0478 |

## Final held-out test evaluation

- Selected model: **Logistic Regression**.
- Selection criterion: highest mean cross-validated PR-AUC among the two supervised models.
- Selected threshold: **0.4968**.
- Threshold selection: out-of-fold training predictions.
- Out-of-fold training precision: 0.2635.
- Out-of-fold training recall: 0.6129.
- Test ROC-AUC: 0.6717.
- Test PR-AUC: 0.3273.
- Test precision: 0.2456.
- Test recall: 0.6000.
- Test F1: 0.3485.

## Confusion matrix

Rows represent actual labels; columns represent predicted labels.
Label order: Stayed (0), Left (1).

```text
[[279 129]
 [ 28  42]]
```

See `baseline_confusion_matrix.png` and `baseline_roc_curve.png`.

## HR interpretation

The model estimates which employees may leave within 12 months. The decision threshold was selected using out-of-fold training predictions to target at least 60% recall. Higher recall may help HR identify more employees who could leave, but can also flag employees who stay.

These predictions indicate patterns, not certainty about an individual's intentions. HR should review results fairly, protect employee privacy, and never use model predictions as the sole basis for employment decisions.

## Limitations

This is a baseline model, not a validated production system. Preprocessing was fitted on the training split before model cross-validation, so cross-validation scores may be optimistic. A stricter evaluation should fit preprocessing separately within each cross-validation fold. Performance and fairness should be checked on future data before operational use.
