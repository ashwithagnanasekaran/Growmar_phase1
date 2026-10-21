# Task 5 — Attendance Anomaly Validation

## Method

- Attendance feature rows: 7339
- Labelled anomaly examples: 150
- Model: Isolation Forest
- Contamination: 3%
- Trees: 300
- Random state: 42
- Validation labels were excluded from model fitting.
- Rules use attendance feature values, not anomaly labels.

## Results on the labelled anomaly examples

| Method | Labelled anomalies | Caught | Missed | Catch rate |
|---|---:|---:|---:|---:|
| Isolation Forest only | 150 | 88 | 62 | 58.67% |
| Isolation Forest + rules | 150 | 111 | 39 | 74.00% |

## Workbook baseline

The workbook reference has precision 0.403 and recall
0.593, with 221 flagged records and 150 injected anomalies.

The supplied validation sheet contains only labelled anomaly examples.
It therefore measures the fraction of those examples detected, but cannot
measure false alarms or genuine precision without labelled normal records.
The reference precision must not be directly compared with this positive-only
evaluation.

## Results by anomaly type

| anomaly_type     |   labelled_count |   model_caught |   model_missed |   model_catch_rate |   combined_caught |   combined_missed |   combined_catch_rate |
|:-----------------|-----------------:|---------------:|---------------:|-------------------:|------------------:|------------------:|----------------------:|
| ABSENCE_STREAK   |               40 |             40 |              0 |              1.000 |                40 |                 0 |                 1.000 |
| IMPOSSIBLE_HOURS |               35 |              0 |             35 |              0.000 |                23 |                12 |                 0.657 |
| LATE_SPIKE       |               30 |              3 |             27 |              0.100 |                 3 |                27 |                 0.100 |
| SUDDEN_DROP      |               30 |             30 |              0 |              1.000 |                30 |                 0 |                 1.000 |
| ZERO_ATTENDANCE  |               15 |             15 |              0 |              1.000 |                15 |                 0 |                 1.000 |

See `attendance_validation_by_type.png` and
`attendance_validation_predictions.csv` for details.

## Missed anomaly analysis

Inspect rows where `model_flag` is 0 in the predictions CSV. A miss means
the unsupervised model did not flag that labelled example. The rule-based
combination may catch some patterns such as zero attendance, impossible
hours, and long absence streaks. Other patterns such as late-arrival spikes
or gradual changes may be harder to identify with these rules.

## Limitations and responsible use

These labels are injected examples, not necessarily a representative sample
of real incidents. Precision and false-alarm rate require labelled normal
validation examples. Alerts are signals for data-quality checks and human
review, not proof of misconduct. Validate with independently reviewed
normal and anomalous records before production use.
