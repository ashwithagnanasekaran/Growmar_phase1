
"""Feature engineering for the CortexHR dataset."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reusable features without modifying the input DataFrame."""

    result = df.copy()

    # 1. Convert employee tenure from months to years.
    if "tenure_months" in result.columns:
        result["tenure_years"] = (
            pd.to_numeric(result["tenure_months"], errors="coerce") / 12
        )

    # 2. Flag employees whose last raise was more than 12 months ago.
    if "months_since_last_raise" in result.columns:
        months = pd.to_numeric(
            result["months_since_last_raise"], errors="coerce"
        )
        result["raise_overdue"] = (months > 12).astype("float64")
        result.loc[months.isna(), "raise_overdue"] = float("nan")

    # 3. Calculate the attendance gap: 1 - attendance rate.
    if "attendance_rate_3m" in result.columns:
        attendance = pd.to_numeric(
            result["attendance_rate_3m"], errors="coerce"
        )
        result["attendance_gap"] = 1 - attendance

    # 4. Convert six-month unpaid leave into an average per month.
    if "unpaid_leave_days_6m" in result.columns:
        unpaid = pd.to_numeric(
            result["unpaid_leave_days_6m"], errors="coerce"
        )
        result["unpaid_per_month"] = unpaid / 6

    elif "unpaid_absence_days" in result.columns:
        # Attendance data is already monthly.
        result["unpaid_per_month"] = pd.to_numeric(
            result["unpaid_absence_days"], errors="coerce"
        )

    # 5. Preserve overtime as a numeric workload measure.
    if "avg_overtime_hours_per_month_3m" in result.columns:
        result["overtime_load"] = pd.to_numeric(
            result["avg_overtime_hours_per_month_3m"],
            errors="coerce",
        )
    elif "overtime_hours" in result.columns:
        result["overtime_load"] = pd.to_numeric(
            result["overtime_hours"], errors="coerce"
        )

    return result