from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.task11_resume_ranking import load_data, build_rankings

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
PROTECTED_FILE = ROOT / "data/unlocked/unlock_CX-15_protected_attributes.csv"
REPORTS = ROOT / "reports"
MODEL_FILE = ROOT / "models_store/salary-v1/1.0.0/model.joblib"

FEATURES = [
    "tenure_months",
    "comp_ratio",
    "last_engagement_score",
    "attendance_rate_3m",
    "late_arrivals_3m",
    "avg_overtime_hours_per_month_3m",
    "months_since_last_raise",
    "training_hours_12m",
]

EPSILON = 0.000001

def dataframe_to_markdown(frame):
    """Render a small DataFrame as Markdown without optional dependencies."""
    columns = [str(c) for c in frame.columns]
    rows = frame.fillna("").astype(str).values.tolist()

    def clean(value):
        return str(value).replace("|", "\\|").replace("\\n", " ")

    header = "| " + " | ".join(clean(c) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in rows
    ]
    return "\\n".join([header, divider] + body)




def psi_for_feature(expected, actual):
    """Calculate PSI using baseline quantile bins and a separate missing bucket."""
    expected = pd.to_numeric(expected, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")

    baseline_values = expected.dropna()

    if baseline_values.empty:
        raise ValueError("Baseline feature contains no numeric values.")

    quantiles = np.linspace(0, 1, 11)
    edges = np.unique(baseline_values.quantile(quantiles).to_numpy())

    if len(edges) == 1:
        value = edges[0]
        edges = np.array([-np.inf, value, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    def assign_bins(series):
        result = pd.Series(index=series.index, dtype="object")
        valid = series.notna()
        result.loc[valid] = pd.cut(
            series.loc[valid],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        ).astype(str)
        result.loc[~valid] = "MISSING"
        return result

    expected_bins = assign_bins(expected)
    actual_bins = assign_bins(actual)

    categories = sorted(set(expected_bins) | set(actual_bins))
    expected_counts = expected_bins.value_counts().reindex(categories, fill_value=0)
    actual_counts = actual_bins.value_counts().reindex(categories, fill_value=0)

    expected_pct = (expected_counts / len(expected_bins)).clip(lower=EPSILON)
    actual_pct = (actual_counts / len(actual_bins)).clip(lower=EPSILON)

    psi = float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())
    return psi, int(expected.isna().sum()), int(actual.isna().sum())


def run_psi_monitor():
    baseline = pd.read_excel(WORKBOOK, sheet_name="attrition_dataset")
    current = pd.read_excel(WORKBOOK, sheet_name="attrition_scoring_2026")

    rows = []
    for feature in FEATURES:
        if feature not in baseline.columns or feature not in current.columns:
            raise ValueError(f"PSI feature missing from source data: {feature}")

        score, baseline_nulls, current_nulls = psi_for_feature(
            baseline[feature], current[feature]
        )

        verdict = "ALERT" if score > 0.25 else "WATCH" if score >= 0.10 else "OK"

        rows.append({
            "feature": feature,
            "psi": score,
            "verdict": verdict,
            "baseline_rows": len(baseline),
            "current_rows": len(current),
            "baseline_missing": baseline_nulls,
            "current_missing": current_nulls,
        })

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "monitoring_psi.csv", index=False)
    return result




def run_data_quality_checks():
    baseline = pd.read_excel(WORKBOOK, sheet_name="attrition_dataset")
    current = pd.read_excel(WORKBOOK, sheet_name="attrition_scoring_2026")

    checks = [
        {
            "check": "Baseline attendance above 1.0",
            "count": int((baseline["attendance_rate_3m"] > 1.0).sum()),
            "total": len(baseline),
            "verdict": "ALERT" if (baseline["attendance_rate_3m"] > 1.0).any() else "OK",
        },
        {
            "check": "2026 zero attendance",
            "count": int((current["attendance_rate_3m"] == 0).sum()),
            "total": len(current),
            "verdict": "REVIEW" if (current["attendance_rate_3m"] == 0).any() else "OK",
        },
        {
            "check": "2026 attendance below 0.5",
            "count": int((current["attendance_rate_3m"] < 0.5).sum()),
            "total": len(current),
            "verdict": "REVIEW" if (current["attendance_rate_3m"] < 0.5).any() else "OK",
        },
        {
            "check": "Baseline raise interval exactly 5 months",
            "count": int((baseline["months_since_last_raise"] == 5).sum()),
            "total": len(baseline),
            "verdict": "INFO",
        },
        {
            "check": "2026 raise interval exactly 5 months",
            "count": int((current["months_since_last_raise"] == 5).sum()),
            "total": len(current),
            "verdict": "REVIEW",
        },
    ]

    result = pd.DataFrame(checks)
    result["percentage"] = (result["count"] / result["total"] * 100).round(2)
    result.to_csv(REPORTS / "monitoring_data_quality.csv", index=False)
    return result


def run_resume_fairness():
    resumes, jobs, _ = load_data()
    pairs = pd.read_excel(WORKBOOK, sheet_name="resume_bias_pairs")

    rankings = build_rankings(resumes, jobs)

    rows = []
    for _, pair in pairs.iterrows():
        original_id = pair["resume_id_original"]
        twin_id = pair["resume_id_twin"]

        original = rankings[rankings["resume_id"] == original_id].set_index("job_id")
        twin = rankings[rankings["resume_id"] == twin_id].set_index("job_id")

        common_jobs = original.index.intersection(twin.index)
        if len(common_jobs) == 0:
            raise ValueError(f"No common job ranking for pair {pair['pair_id']}")

        score_differences = (
            original.loc[common_jobs, "ranking_score"]
            - twin.loc[common_jobs, "ranking_score"]
        ).abs()

        rank_differences = (
            original.loc[common_jobs, "overall_rank"]
            - twin.loc[common_jobs, "overall_rank"]
        ).abs()

        max_score_gap = float(score_differences.max())
        max_rank_gap = int(rank_differences.max())

        rows.append({
            "pair_id": pair["pair_id"],
            "attribute_changed": pair["attribute_changed"],
            "jobs_compared": len(common_jobs),
            "max_absolute_score_gap": max_score_gap,
            "max_absolute_rank_gap": max_rank_gap,
            "verdict": "OK" if max_score_gap < 1e-10 else "ALERT",
        })

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "monitoring_resume_fairness.csv", index=False)
    return result


def run_salary_fairness():
    salary = pd.read_excel(WORKBOOK, sheet_name="salary_training")
    protected = pd.read_csv(PROTECTED_FILE)

    protected = protected[protected["record_type"] == "salary_hire"].copy()

    if protected["person_ref"].duplicated().any():
        raise ValueError("Duplicate salary protected-attribute IDs.")

    merged = salary.merge(
        protected[["person_ref", "gender", "age_band"]],
        left_on="hire_id",
        right_on="person_ref",
        how="left",
        validate="one_to_one",
    )

    if merged["gender"].isna().any():
        raise ValueError("Some salary rows have no matched gender attribute.")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = bundle["features"]

    predictions = model.predict(merged[features])
    merged["predicted_salary"] = predictions
    merged["prediction_error"] = (
        merged["predicted_salary"] - merged["offered_annual_fixed"]
    )
    merged["absolute_error"] = merged["prediction_error"].abs()

    rows = []
    for group_column in ["gender", "age_band"]:
        for group, frame in merged.groupby(group_column, dropna=False):
            rows.append({
                "group_type": group_column,
                "group": str(group),
                "count": len(frame),
                "mean_predicted_salary": frame["predicted_salary"].mean(),
                "mean_actual_salary": frame["offered_annual_fixed"].mean(),
                "mean_prediction_error": frame["prediction_error"].mean(),
                "mae_training_data": frame["absolute_error"].mean(),
            })

    result = pd.DataFrame(rows)

    for group_type in result["group_type"].unique():
        subset = result[result["group_type"] == group_type]
        gap = (
            subset["mean_predicted_salary"].max()
            - subset["mean_predicted_salary"].min()
        )
        result.loc[result["group_type"] == group_type, "predicted_salary_group_gap"] = gap

    result["verdict"] = "REVIEW"
    result.to_csv(REPORTS / "monitoring_salary_fairness.csv", index=False)
    return result


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("\n=== DAY 18: RESPONSIBLE AI MONITORING ===")

    print("\n[1/3] Population Stability Index")
    psi = run_psi_monitor()
    print(psi.to_string(index=False))
    quality = run_data_quality_checks()
    print("\nData quality checks:")
    print(quality.to_string(index=False))

    print("\n[2/3] Twin-resume fairness")
    resume = run_resume_fairness()
    print(resume.to_string(index=False))
    print(
        f"\nResume pairs passing: "
        f"{(resume['verdict'] == 'OK').sum()}/{len(resume)}"
    )

    print("\n[3/3] Salary fairness diagnostics")
    salary = run_salary_fairness()
    print(salary.to_string(index=False))

    alert_count = int((psi["verdict"] == "ALERT").sum())
    watch_count = int((psi["verdict"] == "WATCH").sum())
    resume_alerts = int((resume["verdict"] == "ALERT").sum())

    report = [
        "# Day 18 — Responsible AI Monitoring",
        "",
        "## Population Stability Index",
        "",
        dataframe_to_markdown(psi),
        "",
        "Thresholds: OK < 0.10; WATCH 0.10–0.25; ALERT > 0.25.",
        "",
        "PSI compares the baseline attrition dataset with the 2026 scoring population.",
        "Missing values are treated as a separate population bucket.",
        "",
        "## Data quality checks",
        "",
        dataframe_to_markdown(quality),
        "",
        "Attendance values above 1.0 are flagged for validation. Zero and low attendance values are reviewed rather than automatically corrected. A concentration of five-month raise intervals is reported for investigation.",
        "",
        "## Twin-resume fairness",
        "",
        dataframe_to_markdown(resume),
        "",
        "The paired resumes have identical skills and experience. The ranking",
        "implementation excludes candidate name and college tier from its scoring text.",
        "Any non-zero score difference is flagged for investigation.",
        "",
        "## Salary fairness diagnostics",
        "",
        dataframe_to_markdown(salary),
        "",
        "These are in-sample diagnostics from the saved salary model, not independent",
        "test-set performance estimates. Group differences require human review and",
        "context; they do not alone establish discrimination.",
        "",
        "## Monitoring summary",
        "",
        f"- PSI ALERT features: {alert_count}",
        f"- PSI WATCH features: {watch_count}",
        f"- Resume pairs flagged: {resume_alerts}/{len(resume)}",
        f"- Salary results: descriptive group diagnostics; human review required.",
        "",
    ]

    (REPORTS / "day18_monitoring_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n=== MONITORING COMPLETE ===")
    print("Reports saved:")
    print(" - reports/monitoring_psi.csv")
    print(" - reports/monitoring_data_quality.csv")
    print(" - reports/monitoring_resume_fairness.csv")
    print(" - reports/monitoring_salary_fairness.csv")
    print(" - reports/day18_monitoring_report.md")


if __name__ == "__main__":
    main()

