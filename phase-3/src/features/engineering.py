

import pandas as pd


def engineer_features(df):
    """Create reusable features while preserving existing source columns."""
    result = df.copy()

    # Features required by the existing test
    if "attendance_rate_3m" in result.columns:
        result["attendance_rate_feature"] = result["attendance_rate_3m"]

    if "late_arrivals_3m" in result.columns:
        result["late_arrival_count_feature"] = result["late_arrivals_3m"]

    if "avg_overtime_hours_per_month_3m" in result.columns:
        result["overtime_hours_feature"] = (
            result["avg_overtime_hours_per_month_3m"]
        )

    if "tenure_months" in result.columns:
        result["tenure_years_feature"] = result["tenure_months"] / 12.0
        result["tenure_years"] = result["tenure_months"] / 12.0

    if "months_since_last_raise" in result.columns:
        result["months_since_last_revision_feature"] = (
            result["months_since_last_raise"]
        )
        result["raise_overdue"] = (
            result["months_since_last_raise"] > 12
        ).astype(int)

    # GrowMar-required attendance feature
    if "attendance_rate_3m" in result.columns:
        result["attendance_gap"] = 1 - result["attendance_rate_3m"]

    # GrowMar-required overtime feature
    if "avg_overtime_hours_per_month_3m" in result.columns:
        result["overtime_load"] = (
            result["avg_overtime_hours_per_month_3m"]
        )

    return result
