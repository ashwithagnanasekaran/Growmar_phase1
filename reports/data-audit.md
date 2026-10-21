# Task 1 — Data Audit and Leakage Hunt

## 1. Introduction

The aim of this task was to check the employee attrition dataset before using it to train machine learning models. I checked missing values, duplicate records, incorrect values, inconsistent categories, outliers, and columns that could cause data leakage.

## 2. Dataset overview

The dataset contains 2,389 rows and 42 columns. The target column is `left_within_12m`, which indicates whether an employee left within 12 months.

| Check | Result |
|---|---:|
| Total records | 2,389 |
| Total columns | 42 |
| Employees who left | 349 |
| Employee attrition rate | 14.61% |
| Exact duplicate rows | 0 |
| Duplicate records ignoring `row_id` | 6 |
| Missing target values | 0 |

The dataset is imbalanced because most employees stayed with the company. Therefore, accuracy alone will not be enough when evaluating the models.

## 3. Missing values

I found missing values in several columns. Some are related to employee details, while others are related to employees who left the company.

| Column | Missing values | Missing percentage |
|---|---:|---:|
| `months_since_last_promotion` | 1,459 | 61.07% |
| `last_engagement_score` | 528 | 22.10% |
| `last_raise_pct` | 202 | 8.46% |
| `last_performance_rating` | 182 | 7.62% |
| `commute_distance_km` | 55 | 2.30% |
| `highest_education` | 15 | 0.63% |

Exit-related columns also contain many missing values because they are not available for employees who stayed.

Before training a model, I would check why values are missing. Numeric and categorical features can be imputed inside the training pipeline. The target should not be imputed, and information recorded after an employee leaves should not be used as a model input.

## 4. Duplicate records

The audit found no completely identical rows when every column was included. However, six records have duplicate content when `row_id` is ignored.

These records should be checked before modelling. Confirmed duplicate records should be handled so that they do not affect model evaluation.

## 5. Incorrect and unusual values

I checked numeric columns against the expected value ranges.

| Column | Finding |
|---|---|
| `wfh_share_3m` | 213 values outside the 0–1 range |
| `tenure_months` | 4 negative values |
| `total_experience_years` | 10 negative values |
| `attendance_rate_3m` | 2 values above 1.0 |
| `annual_fixed_pay` | Very high salary values need investigation |

The WFH check found 190 values below 0 and 23 values above 1. The counts were confirmed directly against the Excel dataset.

I also found three salary records listed as ten times too high in the answer key. These records need to be checked against `comp_ratio` before changing their values.

Unusual values are not always errors. I would correct a value only after confirming that it is invalid and keep a record of the correction.

## 6. Inconsistent categories

The `work_location` column contains different spellings and formats for some locations.

Examples include:

- `Madras` and `Chennai`
- `Bangalore` and `Bengaluru`
- `Delhi-NCR` and `Delhi NCR`
- Differences in uppercase and lowercase letters

These values can be standardised using clear mappings after confirming that they refer to the same location.

## 7. Data leakage

I identified several columns that contain information recorded after an employee leaves.

| Column | Reason for concern |
|---|---|
| `exit_date` | Records when the employee left |
| `resignation_date` | Records resignation information |
| `exit_reason` | Describes why the employee left |
| `exit_interview_done` | Records information from the exit process |
| `fnf_settlement_amount` | Records the final settlement |
| `voluntary_left_within_12m` | Closely related to the employee's exit outcome |

These columns should be excluded from the attrition prediction features because they could give the model information that would not be available when making a prediction.

The `dept_attrition_rate_prior_12m` column also needs a time-window check to make sure it uses only information available before the observation date.

## 8. Decisions and next steps

Based on the audit, I would take the following actions before training the models:

1. Review the six duplicate records and handle confirmed duplicates.
2. Investigate missing values and choose suitable imputation methods.
3. Check invalid values against the data dictionary before correcting them.
4. Standardise confirmed category variations.
5. Investigate the three salary records using `annual_fixed_pay` and `comp_ratio`.
6. Exclude post-outcome columns and other unsuitable identifiers from model features.
7. Fit preprocessing steps on training data only to avoid data leakage.

## 9. Answer-key verification

The script checked 98 planted issues from the workbook.

| Result | Count |
|---|---:|
| Detected by the implemented checks | 95 |
| Requiring further review | 3 |
| Total planted issues | 98 |

The three salary records remain under review because an unusually high salary alone does not prove that the value is incorrect. They should not be reported as confirmed errors until they have been checked.

## 10. Conclusion

This audit helped me identify missing values, duplicate records, invalid measurements, inconsistent categories, unusual salary values, and possible data leakage. These findings show why data quality checks are important before building a machine learning model.

The main checks are running successfully, and the report and chart have been generated. The remaining work is to verify the three salary records and handle the confirmed data issues appropriately before modelling.