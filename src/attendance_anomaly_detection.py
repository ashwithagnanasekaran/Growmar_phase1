
"""Task 4: Reproducible attendance anomaly detection."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


# 1. Paths and configuration
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
MODEL_DIR = ROOT / "models_store" / "attendance_anomaly" / "1.0.0"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TREES = 300

# Target operating point for threshold selection.
MIN_CATCH_RATE = 0.70
MAX_FALSE_ALARM_RATE = 0.05

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



# 2. Inject known anomalies reproducibly

def inject_anomalies(clean_df, seed=RANDOM_STATE):
    """Create labelled synthetic anomalies without changing clean_df."""

    rng = np.random.default_rng(seed)
    anomalies = clean_df.copy().reset_index(drop=True)

    n_per_type = max(10, round(len(anomalies) * 0.02))

    if len(anomalies) < 3 * n_per_type:
        raise ValueError("Not enough rows to inject all anomaly types.")

    selected = rng.choice(
        len(anomalies),
        size=3 * n_per_type,
        replace=False,
    )

    groups = np.split(selected, 3)
    records = []

    # Type 1: physically implausible working hours.
    for idx in groups[0]:
        anomalies.loc[idx, "avg_hours_worked"] = 24.0
        anomalies.loc[idx, "overtime_hours"] = 80.0

        records.append({
            "source_index": int(idx),
            "anomaly_type": "impossible_hours",
        })

    # Type 2: unusually long, unexplained absence.
    for idx in groups[1]:
        scheduled = int(anomalies.loc[idx, "scheduled_days"])

        anomalies.loc[idx, "present_days"] = 0
        anomalies.loc[idx, "unpaid_absence_days"] = scheduled
        anomalies.loc[idx, "longest_unpaid_streak_days"] = scheduled
        anomalies.loc[idx, "attendance_rate"] = 0.0

        records.append({
            "source_index": int(idx),
            "anomaly_type": "long_absence",
        })

    # Type 3: sudden changes in attendance and work pattern.
    for idx in groups[2]:
        anomalies.loc[idx, "present_days"] = 2
        anomalies.loc[idx, "avg_hours_worked"] = 2.0
        anomalies.loc[idx, "overtime_hours"] = 30.0
        anomalies.loc[idx, "attendance_rate"] = 0.10

        records.append({
            "source_index": int(idx),
            "anomaly_type": "sudden_pattern_change",
        })

    injected = anomalies.iloc[selected].copy().reset_index(drop=True)
    injected["injected_anomaly"] = 1
    injected["anomaly_type"] = [r["anomaly_type"] for r in records]

    return injected, pd.DataFrame(records)



# 3. Calculate detection metrics

def evaluate_threshold(scores, labels, threshold):
    """Higher scores mean more anomalous."""

    predicted = (scores >= threshold).astype(int)

    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    true_negative = int(((predicted == 0) & (labels == 0)).sum())

    catch_rate = true_positive / max(true_positive + false_negative, 1)
    false_alarm_rate = false_positive / max(false_positive + true_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)

    return {
        "threshold": float(threshold),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "catch_rate": catch_rate,
        "false_alarm_rate": false_alarm_rate,
        "precision": precision,
    }



# 4. Main workflow


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading monthly attendance data...")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="attendance_monthly_features",
    )

    required = [
        "attendance_feature_id",
        "employee_id",
        "month",
        *FEATURE_COLUMNS,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["attendance_feature_id"].duplicated().any():
        raise ValueError("attendance_feature_id must be unique.")

    # Only numeric attendance measurements enter the model.
    # IDs, employee identifiers, dates, and injected labels are excluded.
    X = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "Attendance features contain missing or non-numeric values. "
            "Add an explicit missing-value handling policy before training."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain infinite values.")

    # Hold out normal records that are not used to fit the model.
    X_train, X_holdout, train_idx, holdout_idx = train_test_split(
        X,
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Total monthly records: {len(df)}")
    print(f"Clean training records: {len(X_train)}")
    print(f"Normal holdout records: {len(X_holdout)}")

    # Train only on original, unmodified attendance rows.
    print("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=TREES,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)

    # Inject anomalies only into copies of held-out records.
    holdout_df = df.iloc[holdout_idx].copy().reset_index(drop=True)
    holdout_X = X_holdout.reset_index(drop=True)

    injected_df, injection_manifest = inject_anomalies(
        holdout_df,
        seed=RANDOM_STATE,
    )

    # Keep the original held-out records as normal examples.
    normal_eval = holdout_df[FEATURE_COLUMNS].copy()
    normal_eval["injected_anomaly"] = 0
    normal_eval["anomaly_type"] = "normal"

    injected_eval = injected_df[FEATURE_COLUMNS].copy()
    injected_eval["injected_anomaly"] = 1

    evaluation = pd.concat(
        [normal_eval, injected_eval],
        ignore_index=True,
    )

    eval_X = evaluation[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )
    labels = evaluation["injected_anomaly"].to_numpy(dtype=int)

    # IsolationForest.decision_function is higher for normal records.
    # Negating it gives an intuitive score: higher means more anomalous.
    anomaly_scores = -model.decision_function(eval_X)

    # Evaluate a reproducible range of thresholds.
    # These thresholds are tuned against synthetic injected labels.
    quantiles = np.linspace(0.80, 0.999, 150)
    candidate_thresholds = np.unique(
        np.quantile(anomaly_scores, quantiles)
    )

    results = [
        evaluate_threshold(anomaly_scores, labels, threshold)
        for threshold in candidate_thresholds
    ]
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "attendance_threshold_metrics.csv",
        index=False,
    )

    # Prefer at least 70% catch rate while keeping false alarms <= 5%.
    feasible = results_df[
        (results_df["catch_rate"] >= MIN_CATCH_RATE)
        & (results_df["false_alarm_rate"] <= MAX_FALSE_ALARM_RATE)
    ]

    if not feasible.empty:
        # Among feasible choices, maximize precision, then catch rate.
        chosen = feasible.sort_values(
            ["precision", "catch_rate", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

        selection_reason = (
            "Selected the threshold with the highest precision among "
            "thresholds achieving at least 70% catch rate and no more "
            "than 5% false-alarm rate on the synthetic evaluation set."
        )
    else:
        # If no threshold meets both constraints, do not hide that fact.
        low_alarm = results_df[
            results_df["false_alarm_rate"] <= MAX_FALSE_ALARM_RATE
        ]

        if not low_alarm.empty:
            chosen = low_alarm.sort_values(
                ["catch_rate", "precision"],
                ascending=[False, False],
            ).iloc[0]
        else:
            chosen = results_df.sort_values(
                ["false_alarm_rate", "catch_rate"],
                ascending=[True, False],
            ).iloc[0]

        selection_reason = (
            "No tested threshold met both the 70% catch-rate target "
            "and 5% false-alarm limit. Selected the best available "
            "low-false-alarm operating point; review the trade-off "
            "before operational use."
        )

    threshold = float(chosen["threshold"])

    # Save the complete synthetic evaluation set and its scores.
    evaluation_output = evaluation.copy()
    evaluation_output["anomaly_score"] = anomaly_scores
    evaluation_output["predicted_anomaly"] = (
        anomaly_scores >= threshold
    ).astype(int)

    evaluation_output.to_csv(
        REPORT_DIR / "attendance_anomaly_evaluation.csv",
        index=False,
    )

    # Save exactly which source rows were injected.
    injection_manifest.to_csv(
        REPORT_DIR / "attendance_injection_manifest.csv",
        index=False,
    )

    # Plot precision and catch rate against false-alarm rate.
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["false_alarm_rate"],
        results_df["catch_rate"],
        marker=".",
        markersize=3,
        label="Catch rate",
    )
    plt.plot(
        results_df["false_alarm_rate"],
        results_df["precision"],
        marker=".",
        markersize=3,
        label="Precision",
    )
    plt.axvline(
        MAX_FALSE_ALARM_RATE,
        linestyle="--",
        label="5% false-alarm limit",
    )
    plt.axhline(
        MIN_CATCH_RATE,
        linestyle=":",
        label="70% catch-rate target",
    )
    plt.scatter(
        [chosen["false_alarm_rate"]],
        [chosen["catch_rate"]],
        s=70,
        label="Selected threshold",
    )
    plt.xlabel("False-alarm rate on normal holdout rows")
    plt.ylabel("Rate")
    plt.title("Attendance Anomaly Threshold Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "attendance_threshold_tradeoff.png",
        dpi=150,
    )
    plt.close()

    # Save model and metadata together.
    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")

    metadata = {
        "version": "1.0.0",
        "model": "IsolationForest",
        "random_state": RANDOM_STATE,
        "n_estimators": TREES,
        "contamination": "auto",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(X_train)),
        "normal_holdout_rows": int(len(X_holdout)),
        "injected_anomaly_rows": int(len(injected_df)),
        "threshold": threshold,
        "score_definition": "-model.decision_function(X); higher means more anomalous",
        "minimum_target_catch_rate": MIN_CATCH_RATE,
        "maximum_target_false_alarm_rate": MAX_FALSE_ALARM_RATE,
        "measured_catch_rate": float(chosen["catch_rate"]),
        "measured_false_alarm_rate": float(chosen["false_alarm_rate"]),
        "measured_precision": float(chosen["precision"]),
        "threshold_selection_reason": selection_reason,
        "warning": (
            "Threshold metrics use deliberately injected synthetic anomalies. "
            "They do not establish performance on real-world anomalies."
        ),
    }

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Generate the report.
    report = [
        "# Task 4 - Attendance Anomaly Detection",
        "",
        "## Dataset and model",
        "",
        f"- Original attendance records: {len(df)}",
        f"- Clean training rows: {len(X_train)}",
        f"- Normal holdout rows: {len(X_holdout)}",
        f"- Injected anomaly rows: {len(injected_df)}",
        "- Model: Isolation Forest",
        f"- Random seed: {RANDOM_STATE}",
        f"- Selected anomaly-score threshold: {threshold:.6f}",
        "",
        "## Injected anomalies",
        "",
        "Three controlled cases were injected into copies of held-out rows:",
        "",
        "- Impossible working hours: 24 average hours/day and 80 overtime hours.",
        "- Long absence: zero present days and a full scheduled-month absence.",
        "- Sudden pattern change: very low attendance and hours with high overtime.",
        "",
        "The original workbook and clean training records were not modified.",
        "Injected labels were excluded from model features and training.",
        "",
        "## Selected threshold results",
        "",
        f"- Injected anomalies caught: {int(chosen['true_positives'])}",
        f"- Injected anomalies missed: {int(chosen['false_negatives'])}",
        f"- Normal records falsely flagged: {int(chosen['false_positives'])}",
        f"- Normal records correctly left unflagged: {int(chosen['true_negatives'])}",
        f"- Catch rate: {chosen['catch_rate']:.2%}",
        f"- False-alarm rate: {chosen['false_alarm_rate']:.2%}",
        f"- Precision: {chosen['precision']:.2%}",
        "",
        "## Threshold rationale",
        "",
        selection_reason,
        "",
        "See `attendance_threshold_tradeoff.png` and "
        "`attendance_threshold_metrics.csv` for the tested trade-offs.",
        "",
        "## Limitations",
        "",
        "The threshold was selected using the same synthetic evaluation set "
        "used to report these threshold metrics. Results may therefore be "
        "optimistic for the injected anomaly patterns. They do not prove "
        "equivalent catch rates on real anomalies. Validate on independently "
        "reviewed historical cases before deployment.",
        "",
        "Synthetic cases represent unusual measurements, not proof of employee "
        "misconduct. HR must investigate alerts in context.",
        "",
    ]

    (REPORT_DIR / "attendance-anomaly-report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nSelected threshold results:")
    print(
        results_df.loc[
            results_df["threshold"] == threshold
        ].to_string(index=False)
    )
    print(f"\nThreshold: {threshold:.6f}")
    print(f"Catch rate: {chosen['catch_rate']:.2%}")
    print(f"False-alarm rate: {chosen['false_alarm_rate']:.2%}")
    print(f"Model saved: {MODEL_DIR / 'isolation_forest.joblib'}")
    print(f"Report saved: {REPORT_DIR / 'attendance-anomaly-report.md'}")
    print("Task 4 anomaly detection run completed.")


if __name__ == "__main__":
    main()
