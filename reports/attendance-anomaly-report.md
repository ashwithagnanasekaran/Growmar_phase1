# Task 4 - Attendance Anomaly Detection

## Dataset and model

- Original attendance records: 7339
- Clean training rows: 5871
- Normal holdout rows: 1468
- Injected anomaly rows: 87
- Model: Isolation Forest with 300 trees
- Tested contamination settings: [0.01, 0.02, 0.03, 0.05]
- Selected contamination: 3%
- Selected native threshold: 0.0

## Contamination comparison at native threshold (0.0)

|   contamination |   threshold |   catch_rate |   false_alarm_rate |   precision |
|----------------:|------------:|-------------:|-------------------:|------------:|
|          0.0100 |      0.0000 |       0.5632 |             0.0048 |      0.8750 |
|          0.0200 |      0.0000 |       0.6782 |             0.0136 |      0.7468 |
|          0.0300 |      0.0000 |       0.8391 |             0.0204 |      0.7087 |
|          0.0500 |      0.0000 |       1.0000 |             0.0429 |      0.5800 |

## Injected anomalies

- Impossible working hours: 24 average hours/day and 80 overtime hours.
- Long absence: zero present days and a full scheduled-month absence.
- Sudden pattern change: very low attendance/hours with high overtime.

The original workbook and clean training records were not modified.
Injected labels and anomaly-type labels were excluded from model features.

## Selected setting results

- Anomalies caught: 73
- Anomalies missed: 14
- False alarms: 30
- Correctly unflagged normal records: 1438
- Catch rate: 83.91%
- False-alarm rate: 2.04%
- Precision: 70.87%

## Selection rationale

Selected the highest-precision contamination setting meeting the 70% catch-rate target and 5% false-alarm limit at the native Isolation Forest decision threshold of 0.0.

See `attendance_threshold_tradeoff.png`, `attendance_threshold_metrics.csv`, and `attendance_contamination_comparison.csv` for results.

## Limitations

Metrics are measured against synthetic injected anomalies and do not establish performance on real incidents. Validate against independently reviewed historical cases before deployment. Synthetic alerts are not evidence of employee misconduct.
