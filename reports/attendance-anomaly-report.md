# Task 4 - Attendance Anomaly Detection

## Dataset and model

- Original attendance records: 7339
- Clean training rows: 5871
- Normal holdout rows: 1468
- Injected anomaly rows: 87
- Model: Isolation Forest
- Random seed: 42
- Selected anomaly-score threshold: 0.068527

## Injected anomalies

Three controlled cases were injected into copies of held-out rows:

- Impossible working hours: 24 average hours/day and 80 overtime hours.
- Long absence: zero present days and a full scheduled-month absence.
- Sudden pattern change: very low attendance and hours with high overtime.

The original workbook and clean training records were not modified.
Injected labels were excluded from model features and training.

## Selected threshold results

- Injected anomalies caught: 70
- Injected anomalies missed: 17
- Normal records falsely flagged: 36
- Normal records correctly left unflagged: 1432
- Catch rate: 80.46%
- False-alarm rate: 2.45%
- Precision: 66.04%

## Threshold rationale

Selected the threshold with the highest precision among thresholds achieving at least 70% catch rate and no more than 5% false-alarm rate on the synthetic evaluation set.

See `attendance_threshold_tradeoff.png` and `attendance_threshold_metrics.csv` for the tested trade-offs.

## Limitations

The threshold was selected using the same synthetic evaluation set used to report these threshold metrics. Results may therefore be optimistic for the injected anomaly patterns. They do not prove equivalent catch rates on real anomalies. Validate on independently reviewed historical cases before deployment.

Synthetic cases represent unusual measurements, not proof of employee misconduct. HR must investigate alerts in context.
