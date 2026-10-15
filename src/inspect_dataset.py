
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"

excel_files = list(DATA_FOLDER.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError(f"No Excel workbook found in {DATA_FOLDER}")

file_path = excel_files[0]
df = pd.read_excel(file_path, sheet_name="attrition_dataset")

print("DATASET SIZE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nMISSING VALUES BY COLUMN")
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
})

missing = missing.sort_values("missing_percent", ascending=False)
print(missing.to_string())


print("\nATTRITION TARGET DISTRIBUTION")

for column in ["left_within_12m", "voluntary_left_within_12m"]:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).to_string())
        print("\nPercentages:")
        print(
            (df[column].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
            .to_string()
        )


print("\n--- DATA DICTIONARY ---")

dictionary_df = pd.read_excel(
    "data/raw/02_CortexHR_AI_ML_Dataset.xlsx",
    sheet_name="data_dictionary"
)

print(dictionary_df.head(15).to_string(index=False))


print("\n--- OUTLIER ANALYSIS ---")

outlier_columns = [
    "annual_fixed_pay",
    "tenure_months",
    "avg_overtime_hours_per_month_3m",
    "avg_hours_worked_3m",
]

for column in outlier_columns:
    if column not in df.columns:
        print(f"\n{column}: Column not found")
        continue

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        print(f"\n{column}: No numeric values available")
        continue

    # Method 1: Interquartile Range (IQR)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    iqr_outliers = (
        (values < lower_bound) | (values > upper_bound)
    ).sum()

    # Method 2: Z-score
    std = values.std()

    if std > 0:
        z_scores = (values - values.mean()) / std
        z_outliers = (z_scores.abs() > 3).sum()
    else:
        z_outliers = 0

    print(f"\nColumn: {column}")
    print(f"IQR outliers: {iqr_outliers}")
    print(f"Z-score outliers: {z_outliers}")
    print(f"IQR lower bound: {lower_bound:.2f}")
    print(f"IQR upper bound: {upper_bound:.2f}")

    
print("\n--- DATA LEAKAGE REVIEW ---")

possible_leakage_columns = [
    "exit_date",
    "resignation_date",
    "exit_reason",
    "exit_interview_done",
    "fnf_settlement_amount",
    "left_within_12m",
    "voluntary_left_within_12m",
]

for column in possible_leakage_columns:
    if column in df.columns:
        print(f"Review column: {column}")



print("\n--- PROMOTION MISSING-VALUE INVESTIGATION ---")

if (
    "months_since_last_promotion" in df.columns
    and "promotions_received" in df.columns
):
    missing_promotion_time = df[
        df["months_since_last_promotion"].isna()
    ]

    print(
        "\nRows with missing months_since_last_promotion:",
        len(missing_promotion_time)
    )

    print("\nPromotion history for those rows:")
    print(
        missing_promotion_time["promotions_received"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nMissing promotion-time percentage by promotion count:")
    print(
        df.groupby("promotions_received", dropna=False)[
            "months_since_last_promotion"
        ]
        .apply(lambda values: round(values.isna().mean() * 100, 2))
        .to_string()
    )
else:
    print("Required promotion columns were not found.")


print("\n--- ENGAGEMENT SCORE MISSING-VALUE INVESTIGATION ---")

if "last_engagement_score" in df.columns:

    missing_engagement = df["last_engagement_score"].isna()

    print("\nTotal missing engagement scores:")
    print(missing_engagement.sum())

    print("\nMissing engagement percentage:")
    print(round(missing_engagement.mean() * 100, 2), "%")

    # Compare missingness with attrition status
    if "left_within_12m" in df.columns:
        print("\nMissing engagement percentage by attrition status:")

        print(
            df.groupby("left_within_12m")["last_engagement_score"]
            .apply(lambda values: round(values.isna().mean() * 100, 2))
            .to_string()
        )

    # Compare missingness with department
    if "department_id" in df.columns:
        print("\nMissing engagement percentage by department:")

        print(
            df.assign(engagement_missing=missing_engagement)
            .groupby("department_id")["engagement_missing"]
            .mean()
            .mul(100)
            .round(2)
            .to_string()
        )

else:
    print("last_engagement_score column was not found.")



print("\n--- ENGAGEMENT AND OTHER FEATURES ---")

for column in [
    "last_performance_rating",
    "training_hours_12m",
]:
    if column in df.columns:
        print(f"\n{column}:")

        print(
            df.groupby(
                df["last_engagement_score"].isna()
            )[column]
            .agg(["count", "mean"])
            .round(2)
            .rename(
                index={
                    False: "Engagement score present",
                    True: "Engagement score missing",
                }
            )
            .to_string()
        )

if "department_id" in df.columns:
    print("\nDepartment distribution by engagement-score availability:")

    print(
        df.groupby(
            df["last_engagement_score"].isna()
        )["department_id"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

print("\n--- RAISE AND PERFORMANCE MISSING-VALUE INVESTIGATION ---")

columns_to_check = [
    "last_raise_pct",
    "last_performance_rating",
]

for column in columns_to_check:
    if column not in df.columns:
        print(f"\nColumn not found: {column}")
        continue

    missing = df[column].isna()

    print(f"\n{column}")
    print("Missing count:", int(missing.sum()))
    print("Missing percentage:", round(missing.mean() * 100, 2), "%")

    if "left_within_12m" in df.columns:
        print("\nMissing percentage by attrition status:")

        print(
            df.groupby("left_within_12m")[column]
            .apply(lambda values: round(values.isna().mean() * 100, 2))
            .to_string()
        )

print("\n--- MISSINGNESS OVERLAP ---")

if all(column in df.columns for column in columns_to_check):
    raise_missing = df["last_raise_pct"].isna()
    rating_missing = df["last_performance_rating"].isna()

    print("Both missing:", int((raise_missing & rating_missing).sum()))
    print("Raise missing only:", int((raise_missing & ~rating_missing).sum()))
    print("Rating missing only:", int((~raise_missing & rating_missing).sum()))
    print("Neither missing:", int((~raise_missing & ~rating_missing).sum()))


print("\n--- COMMUTE DISTANCE MISSING-VALUE INVESTIGATION ---")

column = "commute_distance_km"

if column in df.columns:
    missing = df[column].isna()

    print("\nMissing count:", int(missing.sum()))
    print("Missing percentage:", round(missing.mean() * 100, 2), "%")

    if "left_within_12m" in df.columns:
        print("\nMissing percentage by attrition status:")
        print(
            df.groupby("left_within_12m")[column]
            .apply(lambda values: round(values.isna().mean() * 100, 2))
            .to_string()
        )

    if "work_mode" in df.columns:
        print("\nMissing percentage by work mode:")
        print(
            df.groupby("work_mode")[column]
            .apply(lambda values: round(values.isna().mean() * 100, 2))
            .to_string()
        )
else:
    print("commute_distance_km column was not found.")


print("\n--- FINAL MISSING-VALUE SUMMARY ---")

missing_summary = df.isna().sum()
missing_summary = missing_summary[missing_summary > 0]

missing_report = pd.DataFrame({
    "missing_count": missing_summary,
    "missing_percentage": (
        missing_summary / len(df) * 100
    ).round(2)
})

print(missing_report.sort_values(
    "missing_percentage",
    ascending=False
).to_string())


print("\n--- DATA QUALITY CHECKS ---")

# 1. Check data types
print("\nColumn data types:")
print(df.dtypes.to_string())

# 2. Check duplicate employee IDs
if "employee_id" in df.columns:
    print("\nDuplicate employee IDs:")
    print(df["employee_id"].duplicated().sum())

# 3. Check inconsistent categorical values
categorical_columns = [
    "department_id",
    "work_location",
    "work_mode",
    "employment_type",
    "highest_education",
]

for col in categorical_columns:
    if col in df.columns:
        print(f"\nUnique values in {col}:")
        print(sorted(df[col].dropna().astype(str).unique()))

# 4. Check impossible or suspicious numeric values
checks = {
    "tenure_months": lambda s: s < 0,
    "total_experience_years": lambda s: s < 0,
    "commute_distance_km": lambda s: s < 0,
    "attendance_rate_3m": lambda s: (s < 0) | (s > 100),
    "wfh_share_3m": lambda s: (s < 0) | (s > 1),
}

print("\nSuspicious numeric values:")

for col, condition in checks.items():
    if col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        print(f"{col}: {condition(values).sum()} suspicious rows")


print("\n--- WFH SHARE INVESTIGATION ---")

if "wfh_share_3m" in df.columns:
    print("\nWFH share summary:")
    print(df["wfh_share_3m"].describe().to_string())

    print("\nLowest WFH share values:")
    print(df["wfh_share_3m"].nsmallest(10).to_list())

    print("\nHighest WFH share values:")
    print(df["wfh_share_3m"].nlargest(20).to_list())

    print("\nValues greater than 1:")
    print((df["wfh_share_3m"] > 1).sum())


print("\n--- WFH SHARE DATA DICTIONARY ---")

excel_file = pd.ExcelFile(file_path)

for sheet in excel_file.sheet_names:
    if "dictionary" in sheet.lower():
        dictionary = pd.read_excel(
            file_path,
            sheet_name=sheet
        )

        print("\nData dictionary columns:")
        print(dictionary.columns.tolist())

        matching_rows = dictionary.astype(str).apply(
            lambda row: row.str.contains(
                "wfh_share_3m",
                case=False,
                regex=False
            ).any(),
            axis=1
        )

        print("\nDefinition of wfh_share_3m:")
        print(dictionary[matching_rows].to_string(index=False))


print("\n--- SUSPICIOUS WFH RECORDS ---")

wfh_column = "wfh_share_3m"

if wfh_column in df.columns:
    suspicious_wfh = df[
        (df[wfh_column] < 0) |
        (df[wfh_column] > 1)
    ]

    columns_to_show = [
        col for col in [
            "employee_id",
            "department_id",
            "work_mode",
            "wfh_share_3m",
            "observation_date",
        ]
        if col in df.columns
    ]

    print("Suspicious record count:", len(suspicious_wfh))
    print(
        suspicious_wfh[columns_to_show]
        .sort_values(wfh_column)
        .to_string(index=False)
    )

print("\n--- DUPLICATE EMPLOYEE ID INVESTIGATION ---")

duplicate_ids = df[
    df.duplicated(subset=["employee_id"], keep=False)
].sort_values("employee_id")

print("Rows with repeated employee IDs:", len(duplicate_ids))

print(
    duplicate_ids[
        [
            "employee_id",
            "observation_date",
            "left_within_12m",
            "voluntary_left_within_12m",
        ]
    ].to_string(index=False)
)

print("\n--- CHECK EXACT DUPLICATE EMPLOYEE RECORDS ---")

duplicate_ids = df[
    df.duplicated(subset=["employee_id"], keep=False)
].sort_values("employee_id")

for employee_id, group in duplicate_ids.groupby("employee_id"):
    print(f"\nEmployee ID: {employee_id}")
    print("Number of rows:", len(group))
    print(
        "Exact duplicate rows:",
        group.duplicated(keep=False).sum()
    )
    print(
        "Columns with different values:",
        group.nunique(dropna=False)[
            group.nunique(dropna=False) > 1
        ].index.tolist()
    )


print("\n--- NUMERIC VALUE VALIDATION ---")

checks = {
    "tenure_months": (0, None),
    "total_experience_years": (0, None),
    "commute_distance_km": (0, None),
    "attendance_rate_3m": (0, 1),
    "wfh_share_3m": (0, 1),
}

for column, (lower, upper) in checks.items():
    if column not in df.columns:
        print(f"\n{column}: column not found")
        continue

    values = df[column]
    invalid = values.notna()

    if lower is not None:
        invalid &= values < lower

    if upper is not None:
        invalid |= (
            values.notna() & (values > upper)
        )

    print(f"\n{column}")
    print("Values outside expected range:", int(invalid.sum()))

    if invalid.any():
        print(
            df.loc[invalid, ["employee_id", column]]
            .head(10)
            .to_string(index=False)
        )

print("\n--- CATEGORY SPELLING INVESTIGATION ---")

categorical_columns = [
    "work_location",
    "work_mode",
    "employment_type",
    "highest_education",
]

for column in categorical_columns:
    if column not in df.columns:
        continue

    print(f"\n{column}:")
    print(
        df[column]
        .value_counts(dropna=False)
        .to_string()
    )

print("\n--- SUSPICIOUS CATEGORY VALUE CHECK ---")

for column in ["work_location", "highest_education"]:
    if column not in df.columns:
        continue

    print(f"\nColumn: {column}")

    suspicious = df[
        df[column].astype("string").str.strip().str.lower()
        == column.lower()
    ]

    print("Matching rows:", len(suspicious))

    if not suspicious.empty:
        print(
            suspicious[
                ["employee_id", "observation_date", column]
            ].to_string(index=False)
        )
