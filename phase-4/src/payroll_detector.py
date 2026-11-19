from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "payroll_monthly_features.csv"
MODEL_PATH = ROOT / "models_store" / "payroll" / "1.1.0" / "payroll_detector.joblib"
RESULT_PATH = ROOT / "reports" / "payroll_anomaly_results.csv"
SUMMARY_PATH = ROOT / "reports" / "payroll_anomaly_summary.md"

FEATURE_COLUMNS = [
    "gross_fixed_ratio",
    "deduction_gross_ratio",
    "net_gross_ratio",
    "overtime_fixed_ratio",
    "variable_fixed_ratio",
    "lop_days",
    "overtime_hours_paid",
]

REQUIRED_COLUMNS = [
    "payroll_item_id", "employee_id", "month", "run_id",
    "monthly_fixed_gross", "gross_pay", "total_deductions",
    "net_pay", "overtime_pay", "variable_pay",
    "lop_days", "overtime_hours_paid",
    "overtime_eligible", "pf_employee",
]


def build_features(df):
    fixed = df["monthly_fixed_gross"].replace(0, np.nan)
    gross = df["gross_pay"].replace(0, np.nan)

    features = pd.DataFrame({
        "gross_fixed_ratio": df["gross_pay"] / fixed,
        "deduction_gross_ratio": df["total_deductions"] / gross,
        "net_gross_ratio": df["net_pay"] / gross,
        "overtime_fixed_ratio": df["overtime_pay"] / fixed,
        "variable_fixed_ratio": df["variable_pay"] / fixed,
        "lop_days": df["lop_days"],
        "overtime_hours_paid": df["overtime_hours_paid"],
    }, index=df.index)

    return features.replace([np.inf, -np.inf], np.nan)


def apply_rules(df):
    duplicate_group = df.duplicated(
        subset=["employee_id", "month", "run_id"], keep=False
    )

    fixed = df["monthly_fixed_gross"].replace(0, np.nan)

    gross_fixed_ratio = df["gross_pay"] / fixed
    overtime_fixed_ratio = df["overtime_pay"] / fixed

    history = df[[
        "employee_id", "month", "monthly_fixed_gross", "pf_employee"
    ]].copy()

    history["_original_index"] = df.index
    history["_month"] = pd.to_datetime(history["month"], errors="coerce")

    history = history.sort_values(
        ["employee_id", "_month", "_original_index"]
    )

    history["previous_fixed_salary"] = (
        history.groupby("employee_id")["monthly_fixed_gross"].shift(1)
    )

    history["previous_pf"] = (
        history.groupby("employee_id")["pf_employee"].shift(1)
    )

    history = history.set_index("_original_index").reindex(df.index)

    salary_ratio = (
        df["monthly_fixed_gross"] /
        history["previous_fixed_salary"].replace(0, np.nan)
    )

    rule_names = [
        ("DUPLICATE_PAYROLL_GROUP", duplicate_group),
        ("NEGATIVE_NET_PAY", df["net_pay"] < 0),
        ("DEDUCTIONS_EXCEED_GROSS",
         df["total_deductions"] > df["gross_pay"]),
        ("ZERO_GROSS_WITH_FIXED_PAY",
         (df["gross_pay"] == 0) & (df["monthly_fixed_gross"] > 0)),
        ("WRONG_DECIMAL_SUSPECTED", gross_fixed_ratio > 3.5),
        ("OVERTIME_SPIKE",
         (df["overtime_eligible"] == "Y") &
         (overtime_fixed_ratio > 0.30)),
        ("SUDDEN_PF_DROP",
         (history["previous_pf"] > 0) &
         (df["pf_employee"] == 0)),
        ("SUDDEN_FIXED_SALARY_INCREASE", salary_ratio > 2.5),
    ]

    reasons = []

    for i in range(len(df)):
        reasons.append([
            name for name, mask in rule_names if bool(mask.iloc[i])
        ])

    return pd.Series(
        [bool(items) for items in reasons], index=df.index
    ), reasons


def run_detector():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Payroll dataset is empty.")

    if df["payroll_item_id"].isna().any():
        raise ValueError("Payroll item IDs contain missing values.")

    if df["payroll_item_id"].duplicated().any():
        raise ValueError("Payroll item IDs must be unique.")

    numeric_columns = [
        "monthly_fixed_gross", "gross_pay", "total_deductions",
        "net_pay", "overtime_pay", "variable_pay", "lop_days",
        "overtime_hours_paid", "pf_employee",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    features = build_features(df)

    if features.isna().all().any():
        bad = features.columns[features.isna().all()].tolist()
        raise ValueError(f"Features entirely missing: {bad}")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination=0.025,
            random_state=42,
        )),
    ])

    model.fit(features)

    model_flags = model.predict(features) == -1
    scores = -model.score_samples(features)

    rule_flags, reasons = apply_rules(df)

    for i, flagged in enumerate(model_flags):
        if flagged:
            reasons[i].append("ISOLATION_FOREST_ANOMALY")

    results = df[[
        "payroll_item_id", "employee_id", "month", "run_id",
        "gross_pay", "total_deductions", "net_pay",
        "monthly_fixed_gross", "pf_employee",
        "overtime_pay", "overtime_hours_paid",
    ]].copy()

    results["anomaly_score"] = scores
    results["model_flag"] = model_flags
    results["rule_flag"] = rule_flags.to_numpy()
    results["anomaly_flag"] = (
        results["model_flag"] | results["rule_flag"]
    )

    results["reason_codes"] = [
        "|".join(items) if items else "NO_RULE_TRIGGER"
        for items in reasons
    ]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "version": "1.1.0",
        "contamination": 0.025,
        "random_state": 42,
    }, MODEL_PATH)

    results.to_csv(RESULT_PATH, index=False)

    both = int((results["model_flag"] & results["rule_flag"]).sum())
    model_count = int(results["model_flag"].sum())
    rule_count = int(results["rule_flag"].sum())
    combined = int(results["anomaly_flag"].sum())

    reason_counts = {}
    for code in [
        "WRONG_DECIMAL_SUSPECTED",
        "OVERTIME_SPIKE",
        "SUDDEN_PF_DROP",
        "SUDDEN_FIXED_SALARY_INCREASE",
    ]:
        reason_counts[code] = int(
            results["reason_codes"].str.contains(code, regex=False).sum()
        )

    summary = f"""# CX-06 Payroll Anomaly Detector

- Rows analysed: {len(df)}
- Isolation Forest flags: {model_count}
- Rule flags: {rule_count}
- Flags from both approaches: {both}
- Combined unique rows flagged: {combined}
- Model version: 1.1.0
- Contamination setting: 0.025

## Additional rule signals

""" + "\n".join(
        f"- {code}: {count}" for code, count in reason_counts.items()
    ) + """

These are review candidates, not confirmed payroll errors.
Candidate thresholds were evaluated against supplied injected anomaly
labels and require validation on additional payroll data.
"""

    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("CX-06 completed.")
    print("Rows analysed:", len(df))
    print("Model flags:", model_count)
    print("Rule flags:", rule_count)
    print("Flags from both:", both)
    print("Combined unique flags:", combined)

    print("\nAdditional rule signals:")
    for code, count in reason_counts.items():
        print(f"{code}: {count}")

    print("Results:", RESULT_PATH)
    print("Saved model:", MODEL_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    run_detector()

