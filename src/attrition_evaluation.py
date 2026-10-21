from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.baseline_models import RawFeaturePreprocessor, TARGET, RANDOM_STATE
from src.features.monthly_aggregation import aggregate_monthly_features
from src.cortexhr_helpers import binary_report, risk_bands


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
REPORT_DIR = ROOT / "reports"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading attrition, attendance, and payroll data...")
    df = pd.read_excel(DATA_PATH, sheet_name="attrition_dataset")
    attendance = pd.read_excel(
        DATA_PATH, sheet_name="attendance_monthly_features"
    )
    payroll = pd.read_excel(
        DATA_PATH, sheet_name="payroll_monthly_features"
    )

    df = aggregate_monthly_features(df, attendance, payroll)
    df = df.reset_index(drop=True)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET]).copy()

    print(f"Rows: {len(df)}")
    print(f"Actual leavers: {int(y.sum())}")
    print(f"Actual stays: {int((y == 0).sum())}")

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    model = SklearnPipeline(
        steps=[
            ("preprocessor", RawFeaturePreprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    print("Generating five-fold out-of-fold probabilities...")
    probabilities = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    # Evaluate the unadjusted 0.5 classification threshold.
    metrics = binary_report(y, probabilities, threshold=0.5)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        REPORT_DIR / "attrition_evaluation_metrics.csv", index=False
    )

    print("\nOut-of-fold metrics:")
    print(metrics_df.to_string(index=False))

    # Rank every employee by estimated probability and assign bands.
    bands = risk_bands(probabilities)
    predictions = pd.DataFrame({
        "employee_id": df["employee_id"].astype(str),
        "observation_date": df["observation_date"].astype(str),
        "actual_left_within_12m": y.to_numpy(),
        "predicted_probability": probabilities,
        "risk_band": bands,
    })

    predictions.to_csv(
        REPORT_DIR / "attrition_oof_predictions.csv", index=False
    )

    band_rows = []
    for band in ["High", "Medium", "Low"]:
        group = predictions[predictions["risk_band"] == band]
        count = len(group)
        actual_leavers = int(group["actual_left_within_12m"].sum())

        band_rows.append({
            "risk_band": band,
            "employee_count": count,
            "share_of_population": count / len(predictions),
            "actual_leavers": actual_leavers,
            "observed_leaver_rate": (
                actual_leavers / count if count else 0.0
            ),
            "share_of_all_leavers_captured": (
                actual_leavers / int(y.sum()) if int(y.sum()) else 0.0
            ),
        })

    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(
        REPORT_DIR / "attrition_risk_band_counts.csv", index=False
    )

    print("\nRisk-band results:")
    print(band_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Calibration curve: compare average predicted risk with observed rate.
    observed_fraction, mean_predicted = calibration_curve(
        y, probabilities, n_bins=10, strategy="quantile"
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        mean_predicted, observed_fraction,
        marker="o", label="Logistic Regression"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed leaver fraction")
    plt.title("Attrition Risk Calibration (Out-of-Fold Predictions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "attrition_calibration_curve.png", dpi=160
    )
    plt.close()

    # Lift: compare the top 10% risk group's leaver rate to the population.
    high = predictions[predictions["risk_band"] == "High"]
    overall_rate = float(y.mean())
    high_rate = (
        float(high["actual_left_within_12m"].mean())
        if len(high) else 0.0
    )
    lift = high_rate / overall_rate if overall_rate > 0 else float("nan")

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["All employees", "High-risk top 10%"],
        [overall_rate, high_rate],
    )
    plt.ylabel("Observed leaver rate")
    plt.title("Observed Attrition Rate: Overall vs High-Risk Group")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "attrition_lift_chart.png", dpi=160)
    plt.close()

    metrics["high_band_observed_leaver_rate"] = high_rate
    metrics["overall_observed_leaver_rate"] = overall_rate
    metrics["high_band_lift_vs_random"] = lift

    note = f"""# Task 4 — Attrition Metrics and Risk Bands

## Evaluation setup

- Dataset rows: {len(df)}
- Actual leavers: {int(y.sum())}
- Actual stays: {int((y == 0).sum())}
- Model: class-weighted Logistic Regression
- Evaluation: five-fold stratified out-of-fold probabilities
- Random state: {RANDOM_STATE}
- Classification threshold for binary metrics: 0.50

## Classification metrics

| Metric | Result |
|---|---:|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 | {metrics['f1']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC / Average Precision | {metrics['pr_auc']:.4f} |
| Brier score | {metrics['brier_score']:.4f} |
| True positives | {metrics['true_positives']} |
| False positives | {metrics['false_positives']} |
| False negatives | {metrics['false_negatives']} |
| True negatives | {metrics['true_negatives']} |

PR-AUC measures how well the model ranks actual leavers above non-leavers,
while accounting for the imbalance between the two groups. It is more
informative than accuracy alone when leavers are the minority class.
The Brier score measures probability error; lower is better.

## Risk bands

| Risk band | Employees | Population share | Actual leavers | Observed leaver rate | Share of all leavers |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r['risk_band']} | {r['employee_count']} | {r['share_of_population']:.1%} | {r['actual_leavers']} | {r['observed_leaver_rate']:.1%} | {r['share_of_all_leavers_captured']:.1%} |" for r in band_rows)}

The High band contains the highest-scoring 10% of employees; Medium contains
the next 20%; Low contains the remaining 70%. These are relative ranking
bands, not guarantees that an employee will leave.

## Lift

- Overall observed leaver rate: {overall_rate:.2%}
- High-risk observed leaver rate: {high_rate:.2%}
- High-risk lift versus the overall rate: {lift:.2f}x

A lift above 1 means the High-risk group contains a greater proportion of
observed leavers than the population average. It does not prove that
intervention will prevent attrition.

## Calibration interpretation

See `attrition_calibration_curve.png`. Points close to the diagonal indicate
that predicted probabilities agree more closely with observed rates.
Points above the diagonal indicate underprediction; points below indicate
overprediction. Interpret the plot cautiously because it is estimated from
cross-validated predictions and finite samples.

## Manager note: appropriate use

The score is a statistical estimate of the likelihood of the recorded
`left_within_12m` outcome, based on the available historical data. It is not
a certainty, a judgement of an employee's commitment, or evidence of poor
performance. High, Medium, and Low are prioritisation bands, not employment
decisions. Use results only to offer voluntary, supportive check-ins and
investigate organisational issues such as workload, career development,
compensation, or management support. A trained HR reviewer must consider
context, data quality, privacy, and potential bias. Do not use these scores
alone to discipline, deny opportunities, or terminate employees. Validate
calibration and fairness on future data before operational use.

## Reproducibility

Run `python -m src.attrition_evaluation` from the project root. The script
rebuilds the five-fold out-of-fold scores and writes the metrics, risk-band
counts, predictions, calibration chart, lift chart, and this note.
"""
    (REPORT_DIR / "attrition_evaluation.md").write_text(
        note, encoding="utf-8"
    )

    print("\nGenerated:")
    for filename in [
        "attrition_evaluation_metrics.csv",
        "attrition_risk_band_counts.csv",
        "attrition_oof_predictions.csv",
        "attrition_calibration_curve.png",
        "attrition_lift_chart.png",
        "attrition_evaluation.md",
    ]:
        print(f"- reports/{filename}")

    print("\nTask 4 evaluation completed.")


if __name__ == "__main__":
    main()
