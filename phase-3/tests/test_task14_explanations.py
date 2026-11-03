import pandas as pd

from src.task14_employee_explanations import (
    get_feature_group,
    format_value,
    make_reason,
)


def test_get_feature_group_maps_categorical_one_hot_feature():
    categorical = ["employment_type", "department_id"]

    result = get_feature_group(
        "employment_type_Fixed-Term Contract",
        categorical,
    )

    assert result == "employment_type"


def test_get_feature_group_maps_engineered_alias():
    result = get_feature_group("tenure_months", [])

    assert result == "tenure"


def test_get_feature_group_keeps_original_feature():
    result = get_feature_group("last_engagement_score", [])

    assert result == "last_engagement_score"


def test_format_value_formats_tenure_in_years():
    row = pd.Series({"tenure_months": 96})

    assert format_value("tenure", row) == "8.0 years"


def test_format_value_formats_engagement_score():
    row = pd.Series({"last_engagement_score": 4.25})

    assert format_value("last_engagement_score", row) == "4.2/5"


def test_format_value_formats_compensation_ratio():
    row = pd.Series({"comp_ratio": 0.825})

    assert format_value("comp_ratio", row) == "0.82x"


def test_format_value_returns_missing_for_nan():
    row = pd.Series({"last_engagement_score": float("nan")})

    assert format_value("last_engagement_score", row) == "missing"


def test_format_value_returns_unavailable_for_absent_column():
    row = pd.Series({"some_other_feature": 10})

    assert format_value("last_engagement_score", row) == "value unavailable"


def test_make_reason_describes_positive_contribution():
    row = pd.Series({"last_engagement_score": 2.5})

    result = make_reason("last_engagement_score", 0.25, row)

    assert "Engagement score (2.5/5)" in result
    assert "pushes the model score higher" in result


def test_make_reason_describes_negative_contribution():
    row = pd.Series({"last_engagement_score": 4.5})

    result = make_reason("last_engagement_score", -0.15, row)

    assert "Engagement score (4.5/5)" in result
    assert "pushes the model score lower" in result


def test_make_reason_describes_zero_contribution():
    row = pd.Series({"last_engagement_score": 3.0})

    result = make_reason("last_engagement_score", 0.0, row)

    assert "has a neutral model contribution" in result
