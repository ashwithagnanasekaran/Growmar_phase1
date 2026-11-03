# Task 14 — Employee Attrition Risk Scoring and Explanations

## 1. Objective

Score all active employees in the 2026 attrition scoring dataset using the saved `attrition-v2` model. Assign risk bands and provide the three strongest SHAP-based explanations for each employee.

The output is intended as a decision-support signal for responsible human review, not as an automated employment decision.

## 2. Data and methodology

- Scoring population: 430 active employees.
- Monthly attendance and payroll features were aggregated for the scoring population.
- The saved `attrition-v2` preprocessing pipeline transformed the model inputs.
- The trained XGBoost model generated 12-month attrition risk scores.
- SHAP TreeExplainer was used to identify the strongest feature contributions for individual predictions.
- Categorical one-hot contributions were grouped into their source features for more readable explanations.
- Explanations identify whether a feature pushes the model score higher or lower.

## 3. Risk-band distribution

| Risk band | Employees | Share |
|---|---:|---:|
| High | 43 | 10% |
| Medium | 86 | 20% |
| Low | 301 | 70% |
| Total | 430 | 100% |

Risk scores ranged from 0.0094 to 0.4625.

Bands were assigned using the project's ranking policy: High = top 10%, Medium = next 20%, and Low = remaining 70%.

## 4. Global SHAP drivers

The leading grouped features by mean absolute SHAP contribution included:

1. Last engagement score
2. Raise timing
3. Tenure
4. Manager changes in the previous 12 months
5. Work-from-home share over three months
6. Compensation ratio
7. Commute distance
8. Annual fixed pay
9. Average hours worked over three months
10. Last performance rating

These are model contribution rankings, not causal explanations. Mean absolute SHAP values are on the model's raw output scale and should not be interpreted as percentage-point changes in attrition probability.

## 5. Manual review

Ten employees were selected for manual inspection across the High, Medium, and Low bands.

The review checked:
- Whether the employee appeared in the output.
- Whether the risk score and band were populated.
- Whether three explanations were present.
- Whether explanations included understandable feature values.
- Whether explanations could show both positive and negative contributions.

The sample included cases where a feature reduced the predicted score even for an employee in the High band. This is expected because the final prediction reflects the combined contribution of multiple features.

## 6. Output files

- `task14_employee_attrition_risk.csv` — employee-level predictions and explanations.
- `task14_manual_review_sample.csv` — ten-row manual review sample.
- `task14_shap_summary.csv` — grouped global SHAP feature summary.
- `task14_shap_summary.png` — SHAP summary visualization.

## 7. Limitations and responsible use

- SHAP describes model behaviour; it does not establish that a feature caused an employee to leave.
- Historical patterns may reflect biases or limitations in the training data.
- A risk score is uncertain and must not be treated as a fact about an employee.
- HR professionals should review relevant context and speak with employees before considering any intervention.
- Sensitive or protected attributes should not be used to make adverse employment decisions.
- The risk bands are relative rankings within this scoring population, not independently validated probability thresholds.

## 8. Completion status

The scoring script completed successfully for all 430 employees. The output contains 430 unique employee records and matches the expected 11-column output schema. The saved model version is `attrition-v2`.

Further automated tests and final project review should be completed before marking the full Phase 3 submission as final.
