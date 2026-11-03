from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.features.monthly_aggregation import aggregate_monthly_features
from src.cortexhr_helpers import risk_bands


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
MODEL_PATH = ROOT / "models_store/attrition-v2/1.0.0/model.joblib"
REPORTS = ROOT / "reports"

OUTPUT_PATH = REPORTS / "task14_employee_attrition_risk.csv"
REVIEW_PATH = REPORTS / "task14_manual_review_sample.csv"
SHAP_SUMMARY_PATH = REPORTS / "task14_shap_summary.csv"
SHAP_CHART_PATH = REPORTS / "task14_shap_summary.png"

MODEL_VERSION = "attrition-v2"

MODEL_NOTE = (
    "Attrition-v2 support signal only; not a decision. "
    "SHAP drivers are model associations, not causes. "
    "Human review required. Bands: High = top 10%, "
    "Medium = next 20%, Low = remaining 70%."
)

# Engineered aliases are combined so one concept does not
# appear multiple times among an employee's top three reasons.
GROUP_ALIASES = {
    "tenure_months": "tenure",
    "tenure_years": "tenure",
    "tenure_years_feature": "tenure",

    "attendance_rate_3m": "attendance_rate",
    "attendance_rate_feature": "attendance_rate",
    "attendance_attendance_rate_mean_3m": "attendance_rate",
    "attendance_gap": "attendance_rate",

    "late_arrivals_3m": "late_arrivals",
    "late_arrival_count_feature": "late_arrivals",
    "attendance_late_arrivals_mean_3m": "late_arrivals",

    "avg_overtime_hours_per_month_3m": "overtime",
    "overtime_hours_feature": "overtime",
    "overtime_load": "overtime",
    "attendance_overtime_hours_mean_3m": "overtime",

    "months_since_last_raise": "raise_timing",
    "months_since_last_revision_feature": "raise_timing",
    "raise_overdue": "raise_timing",
}

GROUP_SOURCE = {
    "tenure": "tenure_months",
    "attendance_rate": "attendance_rate_3m",
    "late_arrivals": "late_arrivals_3m",
    "overtime": "avg_overtime_hours_per_month_3m",
    "raise_timing": "months_since_last_raise",
}

FRIENDLY_LABELS = {
    "tenure": "Tenure",
    "total_experience_years": "Total experience",
    "previous_employers_count": "Previous employers",
    "commute_distance_km": "Commute distance",
    "annual_fixed_pay": "Annual fixed pay",
    "comp_ratio": "Pay ratio versus role midpoint",
    "raises_received": "Raises received",
    "last_raise_pct": "Last raise percentage",
    "raise_timing": "Time since last raise",
    "promotions_received": "Promotions received",
    "months_since_last_promotion": "Time since last promotion",
    "internal_transfers": "Internal transfers",
    "last_performance_rating": "Performance rating",
    "last_engagement_score": "Engagement score",
    "manager_changes_12m": "Manager changes in 12 months",
    "training_hours_12m": "Training hours in 12 months",
    "attendance_rate": "Attendance rate",
    "late_arrivals": "Late arrivals",
    "overtime": "Overtime hours per month",
    "avg_hours_worked_3m": "Average hours worked",
    "wfh_share_3m": "Work-from-home share",
    "unpaid_leave_days_6m": "Unpaid leave days",
    "paid_leave_days_6m": "Paid leave days",
    "dept_attrition_rate_prior_12m": "Previous department attrition rate",
    "attendance_unpaid_absence_days_mean_3m": "Unpaid absence days",
    "attendance_longest_unpaid_streak_days_mean_3m": "Longest unpaid absence streak",
    "attendance_avg_hours_worked_mean_3m": "Average attendance hours",
    "payroll_lop_days_mean_3m": "Loss-of-pay days",
    "payroll_payable_days_mean_3m": "Payable days",
    "payroll_monthly_fixed_gross_mean_3m": "Monthly fixed gross pay",
    "payroll_gross_pay_mean_3m": "Monthly gross pay",
    "payroll_net_pay_mean_3m": "Monthly net pay",
    "department_id": "Department",
    "role_id": "Role",
    "work_location": "Work location",
    "work_mode": "Work mode",
    "employment_type": "Employment type",
    "highest_education": "Education level",
    "job_level": "Job level",
}


def load_inputs():
    scoring = pd.read_excel(
        WORKBOOK, sheet_name="attrition_scoring_2026"
    )
    attendance = pd.read_excel(
        WORKBOOK, sheet_name="attendance_monthly_features"
    )
    payroll = pd.read_excel(
        WORKBOOK, sheet_name="payroll_monthly_features"
    )
    expected = pd.read_excel(
        WORKBOOK, sheet_name="sample_attrition_risk"
    )

    return scoring, attendance, payroll, expected


def get_feature_group(feature_name, categorical_features):
    """Map transformed columns to understandable source concepts."""

    for source_column in sorted(
        categorical_features, key=len, reverse=True
    ):
        if feature_name.startswith(source_column + "_"):
            return source_column

    return GROUP_ALIASES.get(feature_name, feature_name)


def format_value(group, raw_row):
    source = GROUP_SOURCE.get(group, group)

    if source not in raw_row.index:
        return "value unavailable"

    value = raw_row[source]

    if pd.isna(value):
        return "missing"

    if group == "tenure":
        return f"{float(value) / 12:.1f} years"

    if group in {
        "attendance_rate",
        "wfh_share_3m",
        "dept_attrition_rate_prior_12m",
    }:
        return f"{float(value) * 100:.1f}%"

    if group == "last_engagement_score":
        return f"{float(value):.1f}/5"

    if group == "commute_distance_km":
        return f"{float(value):.0f} km"

    if group == "comp_ratio":
        return f"{float(value):.2f}x"

    if group in {"overtime", "avg_hours_worked_3m"}:
        return f"{float(value):.1f} hours"

    if group == "raise_timing":
        return f"{float(value):.0f} months ago"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.2f}"

    return str(value)


def make_reason(group, contribution, raw_row):
    label = FRIENDLY_LABELS.get(
        group, group.replace("_", " ").title()
    )
    value = format_value(group, raw_row)

    if contribution > 0:
        direction = "pushes the model score higher"
    elif contribution < 0:
        direction = "pushes the model score lower"
    else:
        direction = "has a neutral model contribution"

    return f"{label} ({value}) — {direction}"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading scoring data and saved model...")
    scoring, attendance, payroll, expected = load_inputs()
    bundle = joblib.load(MODEL_PATH)

    if scoring["employee_id"].duplicated().any():
        raise ValueError("Scoring data contains duplicate employee IDs.")

    if expected["employee_id"].duplicated().any():
        raise ValueError("Expected output contains duplicate employee IDs.")

    if set(scoring["employee_id"]) != set(expected["employee_id"]):
        raise ValueError("Scoring and expected employee IDs do not match.")

    print("Aggregating monthly features...")
    features = aggregate_monthly_features(
        scoring, attendance, payroll
    )

    print("Transforming model inputs...")
    feature_pipeline = bundle["feature_pipeline"]
    model = bundle["model"]

    X = feature_pipeline.transform(features)

    if X.shape[1] != model.n_features_in_:
        raise ValueError("Model feature count does not match transformed data.")

    print("Scoring all employees...")
    probabilities = model.predict_proba(X)[:, 1]
    bands = risk_bands(probabilities)

    print("Calculating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    if shap_values.shape != X.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}; "
            f"expected {X.shape}"
        )

    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP values contain invalid numbers.")

    feature_names = X.columns.tolist()
    categorical_features = feature_pipeline.categorical_features

    # Combine one-hot columns and engineered aliases into
    # source-level groups before selecting the top three.
    group_indices = {}

    for index, feature_name in enumerate(feature_names):
        group = get_feature_group(
            feature_name, categorical_features
        )
        group_indices.setdefault(group, []).append(index)

    group_names = list(group_indices.keys())
    grouped_shap = np.column_stack([
        shap_values[:, group_indices[group]].sum(axis=1)
        for group in group_names
    ])

    print("Creating employee explanations...")
    explanations = []

    for row_position in range(len(scoring)):
        strongest = np.argsort(
            np.abs(grouped_shap[row_position])
        )[::-1][:3]

        reasons = []

        for group_index in strongest:
            group = group_names[group_index]
            contribution = grouped_shap[row_position, group_index]

            reasons.append(
                make_reason(
                    group,
                    contribution,
                    features.iloc[row_position],
                )
            )

        explanations.append(reasons)

    # Preserve the company's original prediction IDs.
    output = expected[["prediction_id", "employee_id"]].merge(
        scoring[
            ["employee_id", "department_id", "observation_date"]
        ],
        on="employee_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Map scores and explanations by employee ID, not row position.
    prediction_lookup = pd.DataFrame({
        "employee_id": scoring["employee_id"].to_numpy(),
        "risk_score_12m": probabilities,
        "risk_band": bands,
        "top_reason_1": [item[0] for item in explanations],
        "top_reason_2": [item[1] for item in explanations],
        "top_reason_3": [item[2] for item in explanations],
    })

    output = output.merge(
        prediction_lookup,
        on="employee_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    output["as_of_date"] = pd.to_datetime(
        output["observation_date"]
    ).dt.strftime("%Y-%m-%d")

    output["model_version"] = MODEL_VERSION
    output["model_note"] = MODEL_NOTE

    output = output.rename(columns={
        "observation_date": "_observation_date"
    })

    output = output[
        [
            "prediction_id",
            "employee_id",
            "department_id",
            "as_of_date",
            "risk_score_12m",
            "risk_band",
            "top_reason_1",
            "top_reason_2",
            "top_reason_3",
            "model_version",
            "model_note",
        ]
    ]

    if len(output) != 430:
        raise ValueError("Expected exactly 430 employee predictions.")

    if output.isna().any().any():
        raise ValueError("Output contains missing values.")

    if output["employee_id"].duplicated().any():
        raise ValueError("Output contains duplicate employee IDs.")

    output.to_csv(OUTPUT_PATH, index=False)

    # Global SHAP importance summary.
    importance = pd.DataFrame({
        "feature_group": group_names,
        "mean_absolute_shap": np.abs(grouped_shap).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)

    importance.to_csv(SHAP_SUMMARY_PATH, index=False)

    # SHAP chart uses original transformed features for transparency.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    # Select a reproducible review sample across risk bands.
    review_parts = []

    for band, count in [
        ("High", 4),
        ("Medium", 3),
        ("Low", 3),
    ]:
        subset = output[output["risk_band"] == band].copy()
        subset = subset.sort_values(
            "risk_score_12m", ascending=False
        ).head(count)
        review_parts.append(subset)

    review_sample = pd.concat(review_parts, ignore_index=True)
    review_sample.to_csv(REVIEW_PATH, index=False)

    print("\n--- RISK BAND COUNTS ---")
    print(output["risk_band"].value_counts().to_string())

    print("\n--- OUTPUT VALIDATION ---")
    print("Rows:", len(output))
    print("Columns:", len(output.columns))
    print("Unique employees:", output["employee_id"].nunique())
    print("Risk score range:",
          round(output["risk_score_12m"].min(), 4),
          "to",
          round(output["risk_score_12m"].max(), 4))

    print("\n--- SAMPLE EXPLANATIONS ---")
    print(output.head(3).to_string(index=False))

    print("\n--- SAVED FILES ---")
    print(OUTPUT_PATH)
    print(REVIEW_PATH)
    print(SHAP_SUMMARY_PATH)
    print(SHAP_CHART_PATH)

    print("\nTask 14 scoring completed.")


if __name__ == "__main__":
    main()

