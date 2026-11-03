import numpy as np
import pandas as pd

from src.task15_salary_model import (
    FEATURES,
    TARGET,
    calculate_metrics,
    fairness_table,
    load_data,
    role_average_predictions,
)


def test_calculate_metrics_reports_mae_r2_and_negative_predictions():
    actual = np.array([100000, 200000, 300000])
    predicted = np.array([110000, 190000, 300000])

    result = calculate_metrics(actual, predicted)

    assert result["MAE"] == 6666.666666666667
    assert result["R2"] >= 0.99
    assert result["negative_predictions"] == 0


def test_calculate_metrics_detects_negative_predictions():
    actual = np.array([100000, 200000, 300000])
    predicted = np.array([-10000, 210000, 290000])

    result = calculate_metrics(actual, predicted)

    assert result["negative_predictions"] == 1


def test_role_average_baseline_uses_training_role_mean():
    train = pd.DataFrame({
        "role_id": ["R1", "R1", "R2"],
        TARGET: [100000, 200000, 300000],
    })

    test = pd.DataFrame({
        "role_id": ["R1", "R2", "R3"],
    })

    predictions = role_average_predictions(train, test)

    assert np.allclose(
        predictions,
        [150000, 300000, 200000],
    )


def test_fairness_table_reports_group_errors():
    test = pd.DataFrame({
        TARGET: [100000, 200000, 300000, 400000],
        "gender": ["Female", "Female", "Male", "Male"],
    })

    predictions = np.array([110000, 220000, 290000, 380000])

    result = fairness_table(test, predictions, "gender")

    female = result[result["gender"] == "Female"].iloc[0]
    male = result[result["gender"] == "Male"].iloc[0]

    assert female["count"] == 2
    assert male["count"] == 2
    assert female["MAE"] == 15000
    assert male["MAE"] == 15000


def test_protected_attributes_are_not_model_features():
    assert "gender" not in FEATURES
    assert "age_years" not in FEATURES
    assert "age_band" not in FEATURES
    assert "person_ref" not in FEATURES


def test_salary_data_matches_protected_attributes_one_to_one():
    data = load_data()

    assert len(data) == 2934
    assert data["hire_id"].is_unique
    assert data["gender"].notna().all()
    assert data[TARGET].notna().all()
    assert set(data["gender"].unique()) == {
        "Male", "Female", "Other"
    }


