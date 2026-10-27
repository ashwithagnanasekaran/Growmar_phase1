from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PAYROLL_PATH = ROOT / "data" / "raw" / "payroll_monthly_features.csv"
DETECTOR_RESULTS = ROOT / "reports" / "payroll_anomaly_results.csv"
OUTPUT_PATH = ROOT / "reports" / "payroll_cx07_results.csv"

MODEL_VERSION = "iforest-payroll-1.0.0"


def build_reason(row):
    codes = str(row["reason_codes"]).split("|")
    gross = float(row["gross_pay"])
    deductions = float(row["total_deductions"])
    net = float(row["net_pay"])
    fixed = float(row["monthly_fixed_gross"])

    candidates = []

    if "NEGATIVE_NET_PAY" in codes:
        candidates.append((
            "NEGATIVE_NET_PAY",
            "Net pay is negative and requires payroll review.",
            abs(net),
        ))

    if "DEDUCTIONS_EXCEED_GROSS" in codes:
        candidates.append((
            "DEDUCTIONS_EXCEED_GROSS",
            "Total deductions exceed gross pay.",
            max(0.0, deductions - gross),
        ))

    if "ZERO_GROSS_WITH_FIXED_PAY" in codes:
        candidates.append((
            "ZERO_GROSS_WITH_FIXED_PAY",
            "Gross pay is zero despite positive fixed gross pay.",
            fixed,
        ))

    if "DUPLICATE_PAYROLL_GROUP" in codes:
        candidates.append((
            "DUPLICATE_PAYROLL_GROUP",
            "Multiple records share employee, month, and run; verify whether they are duplicate payroll entries.",
            np.nan,
        ))

    if "ISOLATION_FOREST_ANOMALY" in codes:
        candidates.append((
            "STATISTICAL_PAYROLL_ANOMALY",
            "The payroll record has an unusual feature pattern and requires review.",
            np.nan,
        ))

    if not candidates:
        return "NO_ANOMALY", "No anomaly reason was recorded.", np.nan, ""

    # Prefer the largest explicitly quantifiable review amount.
    quantifiable = [item for item in candidates if pd.notna(item[2])]
    selected = (
        max(quantifiable, key=lambda item: item[2])
        if quantifiable else candidates[0]
    )

    all_reasons = " ".join(item[1] for item in candidates)
    return selected[0], selected[1], selected[2], all_reasons


def main():
    for path in [PAYROLL_PATH, DETECTOR_RESULTS]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    payroll = pd.read_csv(PAYROLL_PATH)
    detector = pd.read_csv(DETECTOR_RESULTS)

    required = {
        "payroll_item_id", "employee_id", "month",
        "gross_pay", "monthly_fixed_gross",
        "total_deductions", "net_pay",
    }
    missing = required - set(payroll.columns)
    if missing:
        raise ValueError(f"Payroll columns missing: {sorted(missing)}")

    if payroll["payroll_item_id"].duplicated().any():
        raise ValueError("Payroll item IDs are not unique.")

    results = detector.merge(
        payroll[[
            "payroll_item_id", "monthly_fixed_gross"
        ]],
        on="payroll_item_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )

    if results["monthly_fixed_gross"].isna().all():
        raise ValueError("Could not match detector records to payroll data.")

    results["reason_details"] = results.apply(build_reason, axis=1)

    output = pd.DataFrame({
        "result_id": [
            f"PAY-{i:05d}" for i in range(1, len(results) + 1)
        ],
        "source_dataset": "payroll_monthly_features",
        "record_id": results["payroll_item_id"],
        "employee_id": results["employee_id"],
        "period": results["month"],
        "anomaly_score": results["anomaly_score"],
        "flagged": np.where(results["anomaly_flag"], "Y", "N"),
        "reason_code": results["reason_details"].map(lambda x: x[0]),
        "reason_text": results["reason_details"].map(
            lambda x: (
                x[1]
                if not [
                    reason.strip()
                    for reason in x[3].split(x[1])
                    if reason.strip()
                ]
                else x[1] + " Other review signals: " + " ".join(
                    reason.strip()
                    for reason in x[3].split(x[1])
                    if reason.strip()
                )
            )
        ),
        "money_at_risk": results["reason_details"].map(lambda x: x[2]),
        "model_version": MODEL_VERSION,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    })

    # Rank flagged records first, with highest quantified risk at the top.
    output["_flagged_sort"] = (output["flagged"] == "Y").astype(int)
    output = output.sort_values(
        by=["_flagged_sort", "money_at_risk"],
        ascending=[False, False],
        na_position="last",
    ).drop(columns=["_flagged_sort"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    flagged = output[output["flagged"] == "Y"]
    quantified = flagged["money_at_risk"].dropna()

    print("CX-07 reason-code generation completed.")
    print("Total records:", len(output))
    print("Flagged records:", len(flagged))
    print("Records with quantified review estimates:", len(quantified))
    print("Money-at-risk estimate total (quantified records only):",
          round(float(quantified.sum()), 2))
    print("This total is not confirmed loss and may include overlapping issues.")
    print("Output:", OUTPUT_PATH)
    print("Output columns:", list(output.columns))


if __name__ == "__main__":
    main()
