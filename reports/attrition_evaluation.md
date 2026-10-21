# Task 4 — Attrition Metrics and Risk Bands

## Evaluation setup

- Dataset rows: 2389
- Actual leavers: 349
- Actual stays: 2040
- Model: class-weighted Logistic Regression
- Evaluation: five-fold stratified out-of-fold probabilities
- Random state: 42
- Classification threshold for binary metrics: 0.50

## Classification metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.6865 |
| Precision | 0.2555 |
| Recall | 0.5989 |
| F1 | 0.3582 |
| ROC-AUC | 0.6961 |
| PR-AUC / Average Precision | 0.2912 |
| Brier score | 0.2079 |
| True positives | 209 |
| False positives | 609 |
| False negatives | 140 |
| True negatives | 1431 |

PR-AUC measures how well the model ranks actual leavers above non-leavers,
while accounting for the imbalance between the two groups. It is more
informative than accuracy alone when leavers are the minority class.
The Brier score measures probability error; lower is better.

## Risk bands

| Risk band | Employees | Population share | Actual leavers | Observed leaver rate | Share of all leavers |
|---|---:|---:|---:|---:|---:|
| High | 239 | 10.0% | 82 | 34.3% | 23.5% |
| Medium | 478 | 20.0% | 113 | 23.6% | 32.4% |
| Low | 1672 | 70.0% | 154 | 9.2% | 44.1% |

The High band contains the highest-scoring 10% of employees; Medium contains
the next 20%; Low contains the remaining 70%. These are relative ranking
bands, not guarantees that an employee will leave.

## Lift

- Overall observed leaver rate: 14.61%
- High-risk observed leaver rate: 34.31%
- High-risk lift versus the overall rate: 2.35x

A lift above 1 means the High-risk group contains a greater proportion of
observed leavers than the population average. It does not prove that
intervention will prevent attrition.

## Calibration interpretation

See `attrition_calibration_curve.png`. Points close to the diagonal indicate
that predicted probabilities agree more closely with observed rates.
Points above the diagonal indicate underprediction; points below indicate
overprediction. Interpret the plot cautiously because it is estimated from
cross-validated predictions and finite samples.

## Manager note: appropriate use

The score is a statistical estimate of the likelihood of the recorded
`left_within_12m` outcome, based on the available historical data. It is not
a certainty, a judgement of an employee's commitment, or evidence of poor
performance. High, Medium, and Low are prioritisation bands, not employment
decisions. Use results only to offer voluntary, supportive check-ins and
investigate organisational issues such as workload, career development,
compensation, or management support. A trained HR reviewer must consider
context, data quality, privacy, and potential bias. Do not use these scores
alone to discipline, deny opportunities, or terminate employees. Validate
calibration and fairness on future data before operational use.

## Reproducibility

Run `python -m src.attrition_evaluation` from the project root. The script
rebuilds the five-fold out-of-fold scores and writes the metrics, risk-band
counts, predictions, calibration chart, lift chart, and this note.
