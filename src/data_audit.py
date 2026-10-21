from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
REPORTS = ROOT / "reports"
REPORT_PATH = REPORTS / "data-audit.md"
FIGURE_PATH = REPORTS / "data-audit-overview.png"

TARGET = "left_within_12m"

LEAKAGE_COLUMNS = [
    "exit_date",
    "resignation_date",
    "exit_reason",
    "exit_interview_done",
    "fnf_settlement_amount",
    "voluntary_left_within_12m",
]

IMPOSSIBLE_RULES = {
    "wfh_share_3m": lambda s: (s < 0) | (s > 1),
    "tenure_months": lambda s: s < 0,
    "total_experience_years": lambda s: s < 0,
    "annual_fixed_pay": lambda s: s < 0,
    "commute_distance_km": lambda s: s < 0,
    "last_performance_rating": lambda s: (s < 1) | (s > 5),
    "last_engagement_score": lambda s: (s < 1) | (s > 5),
    "attendance_rate_3m": lambda s: (s < 0) | (s > 1),
    "avg_hours_worked_3m": lambda s: (s < 0) | (s > 24),
    "avg_overtime_hours_per_month_3m": lambda s: s < 0,
    "unpaid_leave_days_6m": lambda s: s < 0,
    "paid_leave_days_6m": lambda s: s < 0,
    "months_since_last_raise": lambda s: s < 0,
    "months_since_last_promotion": lambda s: s < 0,
    "raises_received": lambda s: s < 0,
    "promotions_received": lambda s: s < 0,
    "previous_employers_count": lambda s: s < 0,
    "manager_changes_12m": lambda s: s < 0,
    "training_hours_12m": lambda s: s < 0,
    "late_arrivals_3m": lambda s: s < 0,
    "job_level": lambda s: s < 1,
}


def md_table(frame):
    if frame.empty:
        return "_No findings._"
    return frame.to_markdown(index=False)


def normalise_category(value):
    if pd.isna(value):
        return ""

    value = re.sub(r"\s+", " ", str(value).strip().casefold())

    aliases = {
        "madras": "chennai",
        "bangalore": "bengaluru",
        "delhi-ncr": "delhi ncr",
    }

    return aliases.get(value, value)


def profile_columns(df):
    rows = []

    for column in df.columns:
        rows.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "missing_count": int(df[column].isna().sum()),
            "missing_percent": round(
                float(df[column].isna().mean() * 100), 2
            ),
            "distinct_values": int(
                df[column].nunique(dropna=True)
            ),
        })

    return pd.DataFrame(rows)


def missing_analysis(df):
    profile = profile_columns(df)

    return profile[
        profile["missing_count"] > 0
    ].sort_values("missing_count", ascending=False)


def outlier_analysis(df):
    rows = []

    for column in df.select_dtypes(include=np.number).columns:
        if column in [
            "row_id",
            TARGET,
            "voluntary_left_within_12m",
        ]:
            continue

        values = pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        std = values.std(ddof=0)

        if std > 0:
            z_count = int(
                (
                    (
                        (values - values.mean()) / std
                    ).abs() > 3
                ).sum()
            )
        else:
            z_count = 0

        rows.append({
            "column": column,
            "iqr_outliers": int(
                (
                    (values < q1 - 1.5 * iqr)
                    | (values > q3 + 1.5 * iqr)
                ).sum()
            ),
            "zscore_abs_gt_3": z_count,
            "minimum": round(float(values.min()), 4),
            "median": round(float(values.median()), 4),
            "maximum": round(float(values.max()), 4),
        })

    return pd.DataFrame(rows)


def duplicate_analysis(df):
    without_row_id = [
        column for column in df.columns
        if column != "row_id"
    ]

    return pd.DataFrame([
        {
            "check": "Exact duplicate rows across all columns",
            "count": int(df.duplicated().sum()),
        },
        {
            "check": "Duplicate content excluding row_id",
            "count": int(
                df.duplicated(
                    subset=without_row_id
                ).sum()
            ),
        },
        {
            "check": "Repeated employee_id values",
            "count": int(
                df["employee_id"].duplicated().sum()
            ) if "employee_id" in df else 0,
        },
    ])


def category_analysis(df):
    rows = []

    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:

        variants = {}

        for value in df[column].dropna().astype(str).unique():
            key = normalise_category(value)
            variants.setdefault(key, set()).add(value)

        for key, values in variants.items():
            if len(values) > 1:
                rows.append({
                    "column": column,
                    "normalised_category": key,
                    "variants": ", ".join(sorted(values)),
                    "decision": (
                        "Review and standardise confirmed equivalent "
                        "values using an explicit mapping."
                    ),
                })

    if "work_location" in df.columns:
        locations = set(
            df["work_location"].dropna().astype(str)
        )

        aliases = [
            ("Madras", "Chennai"),
            ("Bangalore", "Bengaluru"),
            ("Delhi-NCR", "Delhi NCR"),
        ]

        for alias, canonical in aliases:
            if alias in locations:
                rows.append({
                    "column": "work_location",
                    "normalised_category": normalise_category(alias),
                    "variants": f"{canonical}, {alias}",
                    "decision": (
                        "Confirm that both names represent the same "
                        "location before standardising."
                    ),
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "normalised_category",
                "variants",
                "decision",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates()


def impossible_value_analysis(df):
    rows = []

    for column, rule in IMPOSSIBLE_RULES.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column], errors="coerce"
        )

        invalid = values.notna() & rule(values)

        rows.append({
            "column": column,
            "count": int(invalid.sum()),
            "examples": ", ".join(
                values[invalid].astype(str).head(5)
            ),
            "decision": (
                "Review and correct confirmed invalid values, "
                "keeping an audit trail."
                if invalid.any()
                else "No values flagged by this rule."
            ),
        })

    return pd.DataFrame(rows)


def class_balance(df):
    if TARGET not in df.columns:
        return pd.DataFrame(
            columns=["target_value", "count", "percent"]
        )

    counts = df[TARGET].value_counts(
        dropna=False
    ).sort_index()

    return pd.DataFrame([
        {
            "target_value": str(label),
            "count": int(count),
            "percent": round(
                float(count / len(df) * 100), 2
            ),
        }
        for label, count in counts.items()
    ])


def leakage_analysis(df):
    rows = []

    for column in LEAKAGE_COLUMNS:
        if column in df.columns:
            rows.append({
                "column": column,
                "risk": "High",
                "reason": (
                    "Post-outcome or target-derived information."
                ),
                "decision": "Exclude from model features.",
            })

    for column in [
        "employee_id",
        "row_id",
        "observation_date",
        "source",
    ]:
        if column in df.columns:
            rows.append({
                "column": column,
                "risk": "Identifier / operational",
                "reason": (
                    "Identifier, date, or source metadata may "
                    "not generalise to new observations."
                ),
                "decision": (
                    "Exclude unless a justified use is documented."
                ),
            })

    if "dept_attrition_rate_prior_12m" in df.columns:
        rows.append({
            "column": "dept_attrition_rate_prior_12m",
            "risk": "Needs time-window check",
            "reason": (
                "The feature is intended to describe the prior "
                "12 months."
            ),
            "decision": (
                "Confirm that only information available before "
                "the observation date was used."
            ),
        })

    return pd.DataFrame(rows)


def verify_answer_key(df, answer_key):
    results = []

    for _, item in answer_key.iterrows():
        employee_id = str(item["employee_id"]).strip()
        issue_type = str(item["issue_type"]).strip()
        detail = str(item["detail"]).strip()

        records = df[
            df["employee_id"].astype(str).str.strip()
            == employee_id
        ]

        detected = False
        evidence = "Needs manual verification."

        if records.empty:
            evidence = "Employee ID was not found."

        elif issue_type == "MISSING_VALUE":
            match = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)\s+blank",
                detail,
            )

            column = match.group(1) if match else None

            detected = bool(
                column
                and column in df.columns
                and records[column].isna().any()
            )

            evidence = (
                f"{column} is missing for this employee."
                if detected
                else f"{column or 'Column'} missing value not confirmed."
            )

        elif issue_type == "INCONSISTENT_CATEGORY":
            match = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)\s+'.*?'"
                r"\s+written as\s+'(.*?)'",
                detail,
            )

            column, malformed = (
                (match.group(1), match.group(2))
                if match
                else (None, None)
            )

            actual = (
                records[column].dropna().astype(str).str.strip()
                if column in df.columns
                else pd.Series(dtype=str)
            )

            detected = bool(
                malformed
                and actual.str.casefold().eq(
                    malformed.casefold()
                ).any()
            )

            evidence = (
                f"{column} contains the planted variant {malformed!r}."
                if detected
                else "The expected category variant was not confirmed."
            )

        elif issue_type == "NEGATIVE_TENURE":
            detected = (
                "tenure_months" in df.columns
                and pd.to_numeric(
                    records["tenure_months"], errors="coerce"
                ).lt(0).any()
            )

            evidence = (
                "Negative tenure confirmed."
                if detected
                else "Negative tenure not confirmed."
            )

        elif issue_type == "IMPOSSIBLE_VALUE":
            detected = (
                "attendance_rate_3m" in df.columns
                and pd.to_numeric(
                    records["attendance_rate_3m"],
                    errors="coerce",
                ).gt(1).any()
            )

            evidence = (
                "Attendance rate exceeds 1.0."
                if detected
                else "Attendance rate above 1.0 not confirmed."
            )

        elif issue_type == "SALARY_TYPO_OUTLIER":
            if {
                "annual_fixed_pay",
                "comp_ratio",
            }.issubset(df.columns):
                pay = pd.to_numeric(
                    records["annual_fixed_pay"],
                    errors="coerce",
                )

                ratio = pd.to_numeric(
                    records["comp_ratio"],
                    errors="coerce",
                )

                evidence = (
                    "Salary requires manual verification against "
                    "comp_ratio and the data dictionary. An unusually "
                    "high salary alone does not confirm a typo."
                )

                detected = False

            else:
                evidence = (
                    "Salary or comp_ratio column is unavailable; "
                    "manual verification is required."
                )
                detected = False

        elif issue_type == "DUPLICATE_ROW":
            without_row_id = [
                column for column in df.columns
                if column != "row_id"
            ]

            duplicate_rows = df[
                df["employee_id"].astype(str).str.strip()
                == employee_id
            ]

            detected = (
                len(duplicate_rows) > 1
                and duplicate_rows.duplicated(
                    subset=without_row_id,
                    keep=False,
                ).any()
            )

            evidence = (
                "Duplicate content found when row_id is ignored."
                if detected
                else "Duplicate content not confirmed for this employee."
            )

        results.append({
            "employee_id": employee_id,
            "issue_type": issue_type,
            "detail": detail,
            "status": (
                "Detected by rule"
                if detected
                else "Needs review"
            ),
            "evidence": evidence,
        })

    return pd.DataFrame(results)


def create_chart(df, missing, outliers, balance):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if not missing.empty:
        top = missing.sort_values(
            "missing_percent"
        ).tail(10)

        axes[0].barh(
            top["column"],
            top["missing_percent"],
        )
        axes[0].set_title("Columns with most missing values")
        axes[0].set_xlabel("Missing (%)")
    else:
        axes[0].text(
            0.5, 0.5, "No missing values",
            ha="center",
        )

    if not outliers.empty:
        axes[1].bar(
            outliers["column"],
            outliers["iqr_outliers"],
        )
        axes[1].tick_params(
            axis="x", rotation=70
        )
        axes[1].set_title("IQR outlier screening")
    else:
        axes[1].text(
            0.5, 0.5, "No numeric results",
            ha="center",
        )

    if not balance.empty:
        axes[2].bar(
            balance["target_value"],
            balance["count"],
        )
        axes[2].set_title("Target class balance")
        axes[2].set_ylabel("Rows")
    else:
        axes[2].text(
            0.5, 0.5, "Target unavailable",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(
        FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not WORKBOOK.exists():
        raise FileNotFoundError(
            f"Workbook not found: {WORKBOOK}"
        )

    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(
        WORKBOOK,
        sheet_name="attrition_dataset",
    )

    dictionary = pd.read_excel(
        WORKBOOK,
        sheet_name="data_dictionary",
    )

    # Calculate the independent audit before comparing the answer key.
    profile = profile_columns(df)
    missing = missing_analysis(df)
    outliers = outlier_analysis(df)
    duplicates = duplicate_analysis(df)
    categories = category_analysis(df)
    impossible = impossible_value_analysis(df)
    balance = class_balance(df)
    leakage = leakage_analysis(df)

    answer_key = pd.read_excel(
        WORKBOOK,
        sheet_name="answer_key_planted_dirt",
    )

    notes = pd.read_excel(
        WORKBOOK,
        sheet_name="answer_key_notes",
    )

    comparison = verify_answer_key(df, answer_key)

    detected = int(
        comparison["status"].eq("Detected by rule").sum()
    )
    needs_review = int(
        comparison["status"].eq("Needs review").sum()
    )

    if {
        "sheet", "column", "description", "data_type"
    }.issubset(dictionary.columns):

        master_dictionary = dictionary[
            dictionary["sheet"].astype(str).eq(
                "attrition_dataset"
            )
        ][["column", "description", "data_type"]]
    else:
        master_dictionary = pd.DataFrame()

    create_chart(df, missing, outliers, balance)

    leavers = int(
        pd.to_numeric(df[TARGET], errors="coerce").eq(1).sum()
    )

    leaver_rate = leavers / len(df) * 100

    duplicate_content = int(
        df.duplicated(
            subset=[
                column for column in df.columns
                if column != "row_id"
            ]
        ).sum()
    )

    summary_by_issue = (
        comparison.groupby(["issue_type", "status"])
        .size()
        .reset_index(name="count")
    )

    report = f"""# Task 1 — Data Audit and Leakage Hunt

## 1. Dataset overview

The workbook contains **{len(df):,} rows and {len(df.columns)} columns**.
The target column is `left_within_12m`.

- Employees labelled as leavers: {leavers:,} ({leaver_rate:.2f}%)
- Exact duplicate rows across all columns: {int(df.duplicated().sum())}
- Duplicate content excluding `row_id`: {duplicate_content}
- Missing target values: {int(df[TARGET].isna().sum())}

## 2. Column profile

{md_table(profile)}

## 3. Data dictionary

The following descriptions come from the workbook's data dictionary.

{md_table(master_dictionary)}

## 4. Missing values

{md_table(missing)}

For modelling, use median imputation for numeric features and most-frequent
imputation for categorical features. Fit imputers on training data only.
For features with more than 2% missing values, add a missingness indicator
where appropriate. Do not impute the target or use post-outcome columns
as model inputs.

## 5. Outlier screening

IQR and absolute z-scores greater than 3 are used to screen numeric values.
These methods identify unusual values but do not prove that a value is wrong.

{md_table(outliers)}

Investigate extreme values against the data dictionary and related columns.
Keep valid extreme observations and correct only confirmed errors.

## 6. Duplicate records

{md_table(duplicates)}

There are six records with duplicate content when `row_id` is ignored.
Review these records and exclude confirmed duplicates from modelling.
A different generated row identifier does not make the remaining data unique.

## 7. Inconsistent category spellings

{md_table(categories)}

The checks cover case and whitespace variations and known location aliases,
including `Madras`/`Chennai`, `Bangalore`/`Bengaluru`, and
`Delhi-NCR`/`Delhi NCR`. Confirm equivalent meanings before standardising
category values.

## 8. Impossible values

{md_table(impossible)}

Negative tenure and attendance rates above 1.0 are invalid under the
feature definitions. Review other flagged values against the data dictionary
before changing them. Keep a record of any corrections.

## 9. Target class balance

{md_table(balance)}

The positive-class rate is {leaver_rate:.2f}%. Because the classes are
imbalanced, accuracy alone is not enough. Later model evaluation should
include precision, recall, F1, ROC-AUC and PR-AUC.

![Audit overview](data-audit-overview.png)

## 10. Potential leakage

{md_table(leakage)}

Exclude exit dates, exit reasons, exit-interview information, settlement
amounts, and `voluntary_left_within_12m` from predictive features.
Confirm that `dept_attrition_rate_prior_12m` uses only information available
before the observation date.

## 11. Decisions

| Issue | Decision | Reason |
|---|---|---|
| Missing values | Investigate, then impute inside the training pipeline | Avoid fitting preprocessing on held-out data |
| Duplicate content | Review and remove confirmed duplicate records from modelling | Avoid duplicated observations affecting evaluation |
| Category variants | Use explicit mappings after confirming equivalent values | Keep category labels consistent |
| Impossible values | Correct confirmed errors or set them to missing with an audit trail | Avoid treating invalid measurements as real |
| Statistical outliers | Investigate instead of automatically deleting | Some extreme values are valid |
| Leakage candidates | Exclude from model features | Use only information available at prediction time |

## 12. Answer-key comparison

After calculating the independent audit checks, the script checked the
98 planted issues against the employee IDs and relevant fields.

- Planted issues listed in the answer key: {len(answer_key)}
- Detected by the implemented checks: {detected}
- Still needing review: {needs_review}

### Results by issue type

{md_table(summary_by_issue)}

### Record-level results

{md_table(comparison)}

### Answer-key notes

{md_table(notes)}

## 13. What I missed / needs manual review

Items marked **Needs review** were not counted as detected. Salary outliers
need special care: an unusually high salary is a warning, not proof of an
error. Check `annual_fixed_pay` against `comp_ratio` and the data dictionary
before changing the value.

## 14. Reproducibility

Run from the project root:

`python -m src.data_audit`

This regenerates the report and chart from the workbook. Task 1 performs
data checks; it does not train a model.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"Dataset shape: {df.shape[0]} rows x "
        f"{df.shape[1]} columns"
    )
    print(
        f"Exact duplicate rows: {int(df.duplicated().sum())}"
    )
    print(
        "Duplicate content excluding row_id: "
        f"{duplicate_content}"
    )
    print(
        f"Target leavers: {leavers} ({leaver_rate:.2f}%)"
    )
    print(
        f"Answer-key checks: {detected}/{len(answer_key)} "
        f"detected; {needs_review} need review"
    )
    print(f"Report saved: {REPORT_PATH}")
    print(f"Chart saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()