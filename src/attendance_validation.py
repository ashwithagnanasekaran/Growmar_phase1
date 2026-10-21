from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models_store" / "attendance_validation" / "1.0.0"
RANDOM_STATE = 42
CONTAMINATION = 0.03
TREES = 300

FEATURE_COLUMNS = [
    "scheduled_days",
    "present_days",
    "wfh_days",
    "half_days_unpaid",
    "paid_leave_days",
    "unpaid_absence_days",
    "longest_unpaid_streak_days",
    "late_arrivals",
    "avg_hours_worked",
    "overtime_hours",
    "attendance_rate",
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading attendance features and validation labels...")
    features = pd.read_excel(
        DATA_PATH, sheet_name="attendance_monthly_features"
    )
    labels = pd.read_excel(
        DATA_PATH, sheet_name="attendance_anomaly_labels"
    )

    if features["attendance_feature_id"].duplicated().any():
        raise ValueError("Attendance feature IDs must be unique.")
    if labels["attendance_feature_id"].duplicated().any():
        raise ValueError("Validation label IDs must be unique.")

    if not labels["attendance_feature_id"].isin(
        features["attendance_feature_id"]
    ).all():
        raise ValueError("Some validation IDs do not match feature records.")

    X_all = features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")

    if X_all.isna().any().any():
        raise ValueError("Missing values found in attendance features.")
    if not np.isfinite(X_all.to_numpy()).all():
        raise ValueError("Infinite values found in attendance features.")

    # Fit the model without using validation labels.
    print("Training Isolation Forest without validation labels...")
    model = IsolationForest(
        n_estimators=TREES,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_all)

    scores = -model.decision_function(X_all)
    scored = features[
        ["attendance_feature_id", "employee_id", "month"]
    ].copy()
    scored["anomaly_score"] = scores
    scored["model_flag"] = (model.predict(X_all) == -1).astype(int)

    # Join labels, scores, AND original feature columns by unique feature ID.
    # Labels are retained only for evaluation, never for model training.
    evaluation = labels.merge(
        features[["attendance_feature_id", *FEATURE_COLUMNS]],
        on="attendance_feature_id",
        how="left",
        validate="one_to_one",
    )
    evaluation = evaluation.merge(
        scored,
        on=["attendance_feature_id", "employee_id", "month"],
        how="left",
        validate="one_to_one",
    )

    if evaluation["anomaly_score"].isna().any():
        raise ValueError("Some labelled rows did not receive model scores.")

    # Simple, explainable rules. These use attendance features only.
    rule_zero_attendance = (
        (evaluation["scheduled_days"] > 0)
        & (evaluation["present_days"] == 0)
        & (
            evaluation["paid_leave_days"]
            < evaluation["scheduled_days"]
        )
    )

    rule_impossible_hours = (
        (evaluation["avg_hours_worked"] > 16)
        | (evaluation["avg_hours_worked"] < 0)
        | (evaluation["overtime_hours"] > 60)
        | (evaluation["overtime_hours"] < 0)
    )

    rule_absence_streak = (
        (evaluation["longest_unpaid_streak_days"] >= 7)
        & (evaluation["unpaid_absence_days"] >= 7)
    )

    evaluation["rule_flag"] = (
        rule_zero_attendance
        | rule_impossible_hours
        | rule_absence_streak
    ).astype(int)

    evaluation["combined_flag"] = (
        (evaluation["model_flag"] == 1)
        | (evaluation["rule_flag"] == 1)
    ).astype(int)

    # This sheet contains positive anomaly examples only.
    # Every listed record is labelled as anomalous.
    total = len(evaluation)
    model_caught = int(evaluation["model_flag"].sum())
    combined_caught = int(evaluation["combined_flag"].sum())

    model_catch_rate = model_caught / total if total else 0.0
    combined_catch_rate = combined_caught / total if total else 0.0

    summary = pd.DataFrame([
        {
            "method": "Isolation Forest only",
            "labelled_anomalies": total,
            "caught": model_caught,
            "missed": total - model_caught,
            "catch_rate": model_catch_rate,
            "precision": np.nan,
            "false_alarm_rate": np.nan,
        },
        {
            "method": "Isolation Forest + rules",
            "labelled_anomalies": total,
            "caught": combined_caught,
            "missed": total - combined_caught,
            "catch_rate": combined_catch_rate,
            "precision": np.nan,
            "false_alarm_rate": np.nan,
        },
    ])

    type_rows = []
    for anomaly_type, group in evaluation.groupby("anomaly_type"):
        type_rows.append({
            "anomaly_type": anomaly_type,
            "labelled_count": len(group),
            "model_caught": int(group["model_flag"].sum()),
            "model_missed": int((group["model_flag"] == 0).sum()),
            "model_catch_rate": float(group["model_flag"].mean()),
            "combined_caught": int(group["combined_flag"].sum()),
            "combined_missed": int((group["combined_flag"] == 0).sum()),
            "combined_catch_rate": float(group["combined_flag"].mean()),
        })

    type_df = pd.DataFrame(type_rows)

    summary.to_csv(
        REPORT_DIR / "attendance_validation_metrics.csv", index=False
    )
    type_df.to_csv(
        REPORT_DIR / "attendance_validation_by_type.csv", index=False
    )
    evaluation.to_csv(
        REPORT_DIR / "attendance_validation_predictions.csv", index=False
    )

    type_df.set_index("anomaly_type")[
        ["model_catch_rate", "combined_catch_rate"]
    ].plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Catch rate on labelled anomalies")
    plt.ylim(0, 1)
    plt.title("Attendance Anomaly Detection by Label Type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "attendance_validation_by_type.png", dpi=160
    )
    plt.close()

    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")

    # Read reference values for comparison; they are not used for training.
    baselines = pd.read_excel(
        DATA_PATH, sheet_name="anomaly_baselines"
    )
    attendance_baseline = baselines[
        baselines["task"].str.contains("Attendance anomalies", case=False)
    ]

    if attendance_baseline.empty:
        baseline_precision = np.nan
        baseline_recall = np.nan
    else:
        baseline_precision = float(
            attendance_baseline.iloc[0]["precision"]
        )
        baseline_recall = float(attendance_baseline.iloc[0]["recall"])

    report = f"""# Task 5 — Attendance Anomaly Validation

## Method

- Attendance feature rows: {len(features)}
- Labelled anomaly examples: {total}
- Model: Isolation Forest
- Contamination: {CONTAMINATION:.0%}
- Trees: {TREES}
- Random state: {RANDOM_STATE}
- Validation labels were excluded from model fitting.
- Rules use attendance feature values, not anomaly labels.

## Results on the labelled anomaly examples

| Method | Labelled anomalies | Caught | Missed | Catch rate |
|---|---:|---:|---:|---:|
| Isolation Forest only | {total} | {model_caught} | {total-model_caught} | {model_catch_rate:.2%} |
| Isolation Forest + rules | {total} | {combined_caught} | {total-combined_caught} | {combined_catch_rate:.2%} |

## Workbook baseline

The workbook reference has precision {baseline_precision:.3f} and recall
{baseline_recall:.3f}, with 221 flagged records and 150 injected anomalies.

The supplied validation sheet contains only labelled anomaly examples.
It therefore measures the fraction of those examples detected, but cannot
measure false alarms or genuine precision without labelled normal records.
The reference precision must not be directly compared with this positive-only
evaluation.

## Results by anomaly type

{type_df.to_markdown(index=False, floatfmt=".3f")}

See `attendance_validation_by_type.png` and
`attendance_validation_predictions.csv` for details.

## Missed anomaly analysis

Inspect rows where `model_flag` is 0 in the predictions CSV. A miss means
the unsupervised model did not flag that labelled example. The rule-based
combination may catch some patterns such as zero attendance, impossible
hours, and long absence streaks. Other patterns such as late-arrival spikes
or gradual changes may be harder to identify with these rules.

## Limitations and responsible use

These labels are injected examples, not necessarily a representative sample
of real incidents. Precision and false-alarm rate require labelled normal
validation examples. Alerts are signals for data-quality checks and human
review, not proof of misconduct. Validate with independently reviewed
normal and anomalous records before production use.
"""
    (REPORT_DIR / "attendance-validation-report.md").write_text(
        report, encoding="utf-8"
    )

    print("\nOverall results:")
    print(summary.to_string(index=False))
    print("\nResults by anomaly type:")
    print(type_df.to_string(index=False))
    print("\nGenerated:")
    for filename in [
        "attendance_validation_metrics.csv",
        "attendance_validation_by_type.csv",
        "attendance_validation_predictions.csv",
        "attendance_validation_by_type.png",
        "attendance-validation-report.md",
    ]:
        print(f"- reports/{filename}")
    print(f"- {MODEL_DIR / 'isolation_forest.joblib'}")
    print("\nTask 5 validation completed.")


if __name__ == "__main__":
    main()
