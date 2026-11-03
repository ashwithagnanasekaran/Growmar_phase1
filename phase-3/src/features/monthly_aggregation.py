
"""Aggregate prior three-month attendance and payroll features."""

import pandas as pd


def _aggregate_one_source(
    attrition_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    """Summarise monthly records before each observation date."""

    monthly = monthly_df.copy()
    monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce")

    records = []

    for row_position, row in enumerate(
        attrition_df[["employee_id", "observation_date"]].itertuples(
            index=False, name=None
        )
    ):
        employee_id, observation_date = row
        observation_date = pd.to_datetime(
            observation_date, errors="coerce"
        )

        start_date = (
            observation_date - pd.DateOffset(months=3)
            if pd.notna(observation_date)
            else pd.NaT
        )

        eligible = monthly[
            (monthly["employee_id"] == employee_id)
            & (monthly["month"] < observation_date)
            & (monthly["month"] >= start_date)
        ]

        record = {"_row_position": row_position}

        for column in columns:
            record[f"{prefix}_{column}_mean_3m"] = (
                eligible[column].mean()
                if not eligible.empty
                else float("nan")
            )

        records.append(record)

    return pd.DataFrame(records)


def aggregate_monthly_features(
    attrition_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    payroll_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join prior three-month summaries without changing row count."""

    result = attrition_df.copy()
    result["_row_position"] = range(len(result))

    attendance_columns = [
        "unpaid_absence_days",
        "longest_unpaid_streak_days",
        "late_arrivals",
        "avg_hours_worked",
        "overtime_hours",
        "attendance_rate",
    ]

    payroll_columns = [
        "lop_days",
        "payable_days",
        "monthly_fixed_gross",
        "gross_pay",
        "net_pay",
    ]

    attendance_summary = _aggregate_one_source(
        result, attendance_df, "attendance", attendance_columns
    )
    payroll_summary = _aggregate_one_source(
        result, payroll_df, "payroll", payroll_columns
    )

    result = result.merge(
        attendance_summary,
        on="_row_position",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        payroll_summary,
        on="_row_position",
        how="left",
        validate="one_to_one",
    )

    result = result.sort_values("_row_position")
    result = result.drop(columns="_row_position")
    result.index = attrition_df.index

    if len(result) != len(attrition_df):
        raise ValueError("Monthly aggregation changed the row count.")

    return result
