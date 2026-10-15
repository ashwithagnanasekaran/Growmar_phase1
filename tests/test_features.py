
import pandas as pd

from src.features.engineering import engineer_features


def test_engineer_features():
    sample = pd.DataFrame({
        "attendance_rate_3m": [0.95, 0.80],
        "late_arrivals_3m": [2, 5],
        "avg_overtime_hours_per_month_3m": [3.0, 6.0],
        "tenure_months": [24, 36],
        "months_since_last_raise": [8, 12],
    })

    result = engineer_features(sample)

    # Check that all expected features were created.
    expected_columns = [
        "attendance_rate_feature",
        "late_arrival_count_feature",
        "overtime_hours_feature",
        "tenure_years_feature",
        "months_since_last_revision_feature",
    ]

    for column in expected_columns:
        assert column in result.columns, f"Missing feature: {column}"

    # Check that calculations are correct.
    assert result["tenure_years_feature"].tolist() == [2.0, 3.0]
    assert result["months_since_last_revision_feature"].tolist() == [8, 12]

    # Check that the original DataFrame was not modified.
    assert "tenure_years_feature" not in sample.columns

    print("Feature engineering test passed!")


if __name__ == "__main__":
    test_engineer_features()
