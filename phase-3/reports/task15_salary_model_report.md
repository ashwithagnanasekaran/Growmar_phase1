# Task 15 – Salary Band Recommendation and Fairness Evaluation

## 1. Objective

Build a salary recommendation model using historical hiring data. Compare regression models, evaluate prediction errors, audit differences across gender and career-gap groups, and generate salary estimates against the available 2026 role-based salary bands.

The model is intended to support analysis and human review. It must not make automatic compensation or hiring decisions.

## 2. Dataset and preparation

- Salary training records: 2,934.
- Protected-attribute records: 5,317 across multiple record types.
- Only records with `record_type = salary_hire` were used for the salary fairness join.
- The salary-hire join matched one-to-one across all 2,934 records.
- Number of distinct roles: 47.
- Salary target: `offered_annual_fixed`.

Protected attributes were retained for fairness evaluation, not used as model predictors.

### Model features

Numeric features:
- `hire_year`
- `job_level`
- `prior_experience_years`
- `career_gap_months`
- `skills_count`
- `market_index`

Categorical features:
- `department_id`
- `role_id`
- `work_location`
- `highest_education`

Identifiers, the target, salary band reference columns, protected attributes, and source metadata were excluded from the primary model inputs. In particular, the 2026 band columns were excluded because they are highly correlated with the historical salary target and could introduce circularity.

Career gap was retained as a job-history feature and separately evaluated in the fairness analysis.

## 3. Methodology

- Train/test split: 80% training and 20% testing.
- Random state: 42.
- Training records: 2,347.
- Test records: 587.
- Numeric and categorical preprocessing was handled through a fitted preprocessing pipeline.
- Models were compared using Mean Absolute Error (MAE) and R-squared (R²).
- Negative salary predictions were counted as an additional validity check.

The role-average baseline predicts the mean salary observed in the training data for the relevant role, with a global training mean fallback for an unseen role.

## 4. Model comparison

| Model | Test MAE | Test R² | Negative predictions |
|---|---:|---:|---:|
| Log Linear Regression | ₹53,729.26 | 0.982957 | 0 |
| Linear Regression | ₹81,765.64 | 0.965431 | 1 |
| Ridge Regression | ₹92,453.11 | 0.957220 | 2 |
| Gradient Boosting | ₹99,832.07 | 0.951736 | 0 |
| Role Average Baseline | ₹115,094.24 | 0.933280 | 0 |

The log-linear model achieved the lowest test MAE and was selected as the final model. Its MAE was approximately 34.3% lower than ordinary linear regression on this test split.

The selected estimator is a `TransformedTargetRegressor` using a linear regression model with a log transformation of the salary target.

Saved model:
`models_store/salary-v1/1.0.0/model.joblib`

Model version: `salary-v1`.

## 5. Gender fairness evaluation

The selected model was evaluated separately across the gender groups. Protected attributes were used for this audit only.

| Group | Test records | MAE | Mean signed error |
|---|---:|---:|---:|
| Female | 242 | ₹58,006.52 | +₹9,510.40 |
| Male | 337 | ₹50,523.19 | -₹4,703.60 |
| Other | 8 | ₹59,398.14 | +₹1,295.91 |

- Absolute-error gap between groups: ₹8,874.95.
- Signed-error gap between groups: ₹14,214.00.

The `Other` group contains only eight test records. Its estimates are therefore highly uncertain and should not be treated as a reliable group-level conclusion.

## 6. Career-gap fairness evaluation

| Career-gap group (months) | Test records | MAE | Mean signed error |
|---|---:|---:|---:|
| 0 | 508 | ₹53,789.07 | -₹126.24 |
| 1–6 | 26 | ₹50,159.96 | +₹6,619.39 |
| 7–12 | 25 | ₹48,851.94 | -₹4,273.32 |
| 13–24 | 19 | ₹76,717.60 | +₹38,585.76 |
| 25–60 | 9 | ₹25,681.95 | -₹833.51 |

- Absolute-error gap between groups: ₹51,035.65.
- Signed-error gap between groups: ₹42,859.07.

Some career-gap groups have very small sample sizes. The results are useful for identifying areas for further review, but they do not establish discrimination or explain its cause.

## 7. Salary recommendation output

The final model was refitted on all 2,934 salary records and saved as `salary-v1`.

The generated recommendation file is:

`reports/task15_salary_recommendations.csv`

It contains 2,934 records and includes:
- Hire, department, role, and job-level identifiers.
- 2026 role-specific minimum, midpoint, and maximum salary references.
- Model-predicted salary.
- Recommended low, mid, and high reference values.
- Model estimate position relative to the role band.

### Model estimate position distribution

| Position | Records |
|---|---:|
| Below low | 1,422 |
| Low to mid | 1,324 |
| Mid to high | 188 |
| Above high | 0 |

The low, mid, and high recommendation values use the corresponding role-specific 2026 band minimum, midpoint, and maximum. They are reference values, not separate predictions made by the model.

A separate historical comparison found:
- 1,414 actual offers below the 2026 band minimum.
- 1,520 actual offers within the 2026 band.
- 0 actual offers above the 2026 band maximum.

Historical offers and 2026 reference bands may relate to different time periods. This comparison alone does not establish that an offer was incorrect or unfair.

## 8. Validation and limitations

- All six Task 15 unit tests passed.
- The complete Phase 3 test suite passed: 20 tests.
- The saved model was loaded and checked against predictions.
- All 2,934 recommendation records had a prediction.
- Predictions were finite and non-negative.
- The role-band minimum, midpoint, and maximum ordering was validated.

Important limitations:

1. The final model was refitted on all available salary records. Its predictions for those same records are in-sample estimates, not independent validation results. The reported test metrics come from the separate 587-record test split.
2. The gender and career-gap group comparisons have unequal and sometimes very small sample sizes.
3. Group-level error differences do not by themselves establish discrimination or causation.
4. Historical salary offers and 2026 reference bands may not be directly comparable.
5. The model learns patterns in historical offers. It cannot determine the objectively correct salary for an individual.

## 9. Responsible-use recommendation

Use the model as a decision-support tool only. Human reviewers should consider role responsibilities, relevant experience, performance, current market conditions, internal pay consistency, and applicable compensation policies.

Do not use protected attributes to generate salary predictions. Continue monitoring prediction errors and group-level differences as more representative data becomes available.

## 10. Output files

- `reports/task15_model_comparison.csv`
- `reports/task15_gender_fairness.csv`
- `reports/task15_career_gap_fairness.csv`
- `reports/task15_salary_recommendations.csv`
- `models_store/salary-v1/1.0.0/model.joblib`
- `reports/task15_salary_model_report.md`
