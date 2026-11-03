# Task 12 ? Resume Screener Fairness Audit

## 1. Counterfactual twin-resume test

Tested all 25 supplied resume twin pairs against all 10 job descriptions,
resulting in 250 score comparisons.

| Metric | Result |
|---|---:|
| Resume pairs | 25 |
| Score comparisons | 250 |
| Exact zero-gap comparisons | 250 |
| Non-zero comparisons | 0 |
| Mean absolute score gap | 0.0000 |
| Maximum absolute score gap | 0.0000 |

## 2. Results by changed attribute

| Changed attribute | Comparisons | Zero-gap comparisons |
|---|---:|---:|
| Gender-coded name only | 120 | 120 |
| Gender-coded name and college tier | 130 | 130 |

All 25 pairs had identical skills and experience. Independent validation
also found zero actual skill mismatches and zero actual experience mismatches.

## 3. Protected-feature handling

The Task 11 ranking implementation excludes candidate names, email
addresses, locations and college tier from its scoring inputs. The
counterfactual audit confirms identical scores for the supplied twin pairs.

The Task 11 ranking metrics are the quality measurements for this
protected-feature-excluded implementation:

- ROC-AUC: 0.9983
- Average precision: 0.9927
- Mean Precision@5: 0.9600
- Mean Recall@5: 0.5825

These metrics must be interpreted cautiously because human shortlist
labels are strongly associated with skill coverage.

## 4. Limitations

The zero score gaps demonstrate counterfactual consistency for the
25 supplied pairs and the 10 tested jobs. They do not establish fairness
for all applicants or all demographic groups.

Group-level score and selection-rate differences require separate analysis.
A difference between groups is not automatically evidence of discrimination,
because qualifications and job distributions may differ.

Human review remains necessary. The model must not make hiring decisions
automatically.

## Masking mitigation experiment

Two ranking configurations were compared using the same 185 resumes,
10 jobs, 555 labelled resume-job pairs, scoring weights and evaluation
method.

- Masked: excludes candidate name, email, city and college tier.
- Unmasked diagnostic: includes candidate name, city, college tier
  and raw resume text. This is an experimental comparison only and
  must not be used for candidate selection.

### Ranking quality

| Metric | Masked | Unmasked diagnostic |
|---|---:|---:|
| ROC-AUC | 0.998261 | 0.998261 |
| Average precision | 0.992685 | 0.992677 |
| Precision@5 | 0.960 | 0.960 |
| Recall@5 | 0.582459 | 0.582459 |
| NDCG@5 | 0.986989 | 0.986989 |

Masking did not reduce the measured aggregate ranking quality.

### Ranking stability

- Resume-job comparisons: 1,850
- Comparisons with a changed rank: 992
- Mean absolute rank change: 2.2605
- Maximum absolute rank change: 45
- Mean absolute score change: 0.000701
- Labelled top-five selection changes: 2

The two changed top-five decisions occurred in J05:
RES-0174 (Female) changed from selected to not selected after masking;
RES-0044 (Male) changed from not selected to selected after masking.

### Group fairness findings

Overall model top-five selection gaps:

| Group | Masked | Unmasked diagnostic |
|---|---:|---:|
| Gender | 0.00 pp | 0.75 pp |
| College tier | 2.22 pp | 2.22 pp |

The largest observed job-level gender gap was approximately
15.48 percentage points in J06. The largest college-tier gap was
18.75 percentage points in J07. These disparities remained under
both configurations.

### Fairness conclusion

Masking direct identifiers and college-tier information preserved
aggregate ranking metrics and changed two labelled top-five decisions.
It reduced the overall gender selection gap in this sample, but did
not remove all job-level gender or college-tier disparities.

Group-rate differences are diagnostic signals, not proof of
discrimination. They may reflect differences in qualifications,
sample composition or other factors. The small group sizes and
limited dataset prevent broad fairness claims.

The supplied shortlist labels are strongly related to skill coverage,
so the high ranking metrics may be optimistic and should not be
treated as independent validation.

The ranking system should remain a decision-support tool. Human
review, job-specific fairness monitoring and validation on
independent data are required before any real hiring use.

