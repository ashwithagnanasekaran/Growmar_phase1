# Phase 1 — Task 1: Data Audit Report

## 1. Dataset overview
- Dataset: CortexHR AI/ML dataset
- Rows: 2,389
- Columns: 42
- Excel workbook contains 32 sheets.
- Main dataset sheet: `attrition_dataset`

## 2. Missing values
- `resignation_date`: 86.56% missing
- `exit_reason`: 85.39% missing
- `exit_date`: 85.39% missing
- `exit_interview_done`: 85.39% missing
- `fnf_settlement_amount`: 85.39% missing
- `months_since_last_promotion`: 61.07% missing
- `last_engagement_score`: 22.10% missing
- `last_raise_pct`: 8.46% missing
- `last_performance_rating`: 7.62% missing
- `commute_distance_km`: 2.30% missing
- `highest_education`: 0.63% missing

## 3. Attrition distribution
- `left_within_12m = 0`: 2,040 rows (85.39%)
- `left_within_12m = 1`: 349 rows (14.61%)

The target is imbalanced because substantially more employees stayed than left.

## 4. Initial decisions
- Investigate why each column has missing values before filling or dropping anything.
- Review exit-related fields for possible target leakage.
- Check numeric columns for outliers.
- Do not use answer-key sheets as model features.
- Confirm final cleaning decisions after reviewing the data dictionary and task instructions.
- **Repeated employee records:**
**Repeated employee records**

- **Finding:** Six employee IDs appear twice, affecting 12 rows. All six pairs have identical values across every column except `row_id`.
- **Validation:** Programmatic comparison confirmed six repeated rows when `row_id` is excluded.
- **Decision:** Treat these as suspected duplicate records. Retain one record per confirmed duplicate pair in a separate cleaned modelling dataset, subject to the task instructions. Preserve the original workbook unchanged and document the cleaning operation.


## 5. Outlier analysis

Four numerical columns were checked using the Interquartile Range (IQR) and Z-score methods.

| Column | IQR outliers | Z-score outliers |
|---|---:|---:|
| annual_fixed_pay | 94 | 26 |
| tenure_months | 74 | 42 |
| avg_overtime_hours_per_month_3m | 446 | 61 |
| avg_hours_worked_3m | 48 | 15 |
### Numeric value validation

The numeric validation identified the following potential data-quality issues:

| Column | Values outside expected range | Initial decision |
|---|---:|---|
| `tenure_months` | 4 | Investigate negative tenure values |
| `total_experience_years` | 10 | Investigate negative experience values |
| `commute_distance_km` | 0 | No negative values detected |
| `attendance_rate_3m` | 2 | Validate values above 1 |
| `wfh_share_3m` | 213 | Investigate values outside 0–1 |

**Decision:** Preserve the original dataset. Validate suspicious values against the data dictionary and feature-generation logic before correcting, imputing, or removing records.
### Initial findings and decisions

- Annual fixed pay: Investigate unusually high or low salaries before making changes.
- Tenure: Check extreme values against the valid employment history.
- Overtime: The IQR bounds are both zero, suggesting many values may be zero. Positive overtime should not automatically be treated as an error.
- Hours worked: Investigate extreme values to identify impossible or incorrectly recorded hours.
- **WFH share (`wfh_share_3m`):** 213 records contain values outside the expected range of 0 to 1. The minimum is -0.096 and the maximum is 1.088. The data dictionary defines this field as the share of days worked from home over three months.
- **Finding:** These values may indicate a calculation, scaling, or data-quality issue.
- **Decision:** Do not automatically clip, delete, or replace these values. Investigate the feature-generation logic and validate the values before cleaning.

No outliers have been removed. The results are preliminary flags that require validation before cleaning.

## 6. Data leakage review

The following columns require special attention before model training:

- `exit_date`: May directly reveal when an employee left.
- `resignation_date`: May reveal a resignation that has already occurred.
- `exit_reason`: Describes why an employee left and may reveal the outcome.
- `exit_interview_done`: May indicate that an employee is in the exit process.
- `fnf_settlement_amount`: Final settlement information may only become available after departure.
- `left_within_12m`: This is a possible target label for attrition prediction.
- `voluntary_left_within_12m`: This is a possible target label for voluntary attrition prediction.

### Initial decision

Exclude exit-related information from model input when it would not be available at prediction time. Keep the selected target label separate from the input features. Confirm the prediction date and label definitions before finalizing the feature list.
### Investigation: `months_since_last_promotion`

- Missing values: 1,459 (61.07%).
- All 1,459 rows with missing values have `promotions_received = 0`.
- Employees with one or more recorded promotions have no missing values in this column.
- **Preliminary interpretation:** Missingness is likely structural because employees with no promotions have no previous promotion date to measure.
- **Decision:** Preserve the original values during the audit. Do not apply median imputation or remove these rows without further investigation. Confirm the column definition in the data dictionary before deciding how to represent this information during model preparation.
### Investigation: `last_engagement_score`

- Missing values: 528 (22.10%).
- Missingness among employees who stayed: 21.57%.
- Missingness among employees who left: 25.21%.
- Department-level missingness ranges from 17.70% to 27.07%.
- **Finding:** Missingness differs by attrition status and department. These descriptive differences do not establish causation or statistical significance.
- **Decision:** Preserve the original values and investigate possible reasons for missing engagement scores. Consider median imputation with a missingness indicator during model preparation, using training data only.
### Engagement Score: Relationship with Other Features

- Average performance rating when engagement score is present: 3.33.
- Average performance rating when engagement score is missing: 3.27.
- Average training hours when engagement score is present: 26.56.
- Average training hours when engagement score is missing: 25.47.
- **Finding:** The observed averages are relatively similar. These descriptive comparisons do not establish why engagement scores are missing.
- **Decision:** Retain the original values during the audit and evaluate an appropriate imputation strategy during model preparation.
### Investigation: Salary Increase and Performance Rating

- `last_raise_pct`: 202 missing values (8.46%).
- `last_performance_rating`: 182 missing values (7.62%).
- Both fields are missing for 174 employees.
- Only `last_raise_pct` is missing for 28 employees.
- Only `last_performance_rating` is missing for 8 employees.
- Missingness is slightly higher among employees who left for both fields.
- **Finding:** The overlap suggests a possible shared data-collection issue, which requires further investigation.
- **Decision:** Preserve the original values. Consider numeric imputation and missingness indicators during model preparation, with preprocessing fitted on training data only.
### Investigation: `commute_distance_km`

- Missing values: 55 (2.30%).
- Missingness among employees who stayed: 2.45%.
- Missingness among employees who left: 1.43%.
- Missingness by work mode: Hybrid 2.26%, Onsite 2.41%, Remote 2.21%.
- **Finding:** Missingness is relatively low and broadly similar across work modes and attrition groups.
- **Decision:** Preserve the original values during the audit. If this feature is used in modelling, consider median imputation fitted on training data only, subject to validation.
## 7. Conclusion

The initial data audit covered dataset structure, missing values, attrition distribution, numerical outliers, and potential data leakage. The dataset contains 2,389 employee records and 42 columns, with an attrition rate of 14.61%.

Missing values and potential outliers require careful handling before model training. No original values have been changed or removed. The next stage is to validate the data dictionary, finalize the prediction target and eligible features, and prepare a reproducible preprocessing pipeline.
## 8. Data Quality Decisions

| Issue | Finding | Decision |
|---|---|---|
| Duplicate rows | No exact duplicate rows found in the initial audit | Do not remove rows based on exact-duplicate checks alone |
| Repeated employee IDs | 6 employee IDs appear twice, with differences only in `row_id` | Investigate whether these are repeated snapshots before deciding whether to retain them |
| Negative tenure | 4 rows contain negative `tenure_months` values | Treat as invalid; convert to missing during cleaning after validation |
| Negative experience | 10 rows contain negative `total_experience_years` values | Treat as invalid; convert to missing during cleaning after validation |
| Attendance rate | 2 rows contain values of 1.12 | Verify the feature definition; if it represents a proportion, values above 1 are invalid |
| Work-from-home share | 213 rows fall outside the expected 0–1 range | Investigate the calculation and source data; do not clip or delete automatically |
| Inconsistent city labels | Multiple spellings and capitalization variants appear in `work_location` | Standardize equivalent city names using a documented mapping |
| Missing education | 15 rows have missing `highest_education` values | Preserve during audit; consider most-frequent imputation during model preprocessing |
| Missing values | Several features contain missing values, including engagement and promotion history | Decide on imputation and missing indicators during preprocessing; fit imputers on training data only |
| Potential leakage | Exit dates, resignation details, exit reasons, and settlement information may only be available after an employee leaves | Exclude from prediction features unless availability at prediction time is established |
| Attrition imbalance | 349 employees left (14.61%) and 2,040 stayed (85.39%) | Use appropriate evaluation metrics and consider class imbalance during model evaluation |

**Cleaning policy:** Preserve the original workbook unchanged. Apply approved corrections to a separate processed dataset and document each transformation.
