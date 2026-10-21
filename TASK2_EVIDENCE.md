# Task 2: Reusable Feature Pipeline

## What I implemented

I created a reusable feature pipeline for the CortexHR attrition dataset. It removes columns that could reveal the target or identify individual employees, and it prepares the remaining data for machine learning.

I also added monthly attendance and payroll summaries. For each observation, the summaries use records from the previous three months and are joined to the attrition dataset.

The pipeline handles missing numerical values, scales numerical features, fills missing categorical values, and converts categorical values into numerical features using one-hot encoding.

## Why I fit the pipeline on training data only

1. The imputer learns how to replace missing values from the training data.
2. The scaler learns the numerical means and standard deviations from the training data.
3. The encoder learns the categories from the training data, helping prevent information leakage from validation or test data.

## Testing

I tested the pipeline using three raw attrition rows and their monthly attendance and payroll data. The test checks that the output contains three rows, includes monthly features, contains only numerical values, and has no missing values after preprocessing.

I also tested that the target and leakage columns are excluded, that the pipeline can be saved and loaded, and that transforming data before fitting raises an error.

The test results will be recorded after running the test suite.

## Files

- `src/features/engineering.py` — creates additional features.
- `src/features/monthly_aggregation.py` — aggregates monthly attendance and payroll data.
- `src/features/pipeline.py` — selects safe features and preprocesses them.
- `tests/test_pipeline.py` — tests the reusable feature pipeline.
- `tests/test_features.py` — tests feature engineering.

## Result

The feature pipeline and monthly aggregation have been connected to the existing model workflow. The final test result and feature artifact build result are recorded separately from this explanation.