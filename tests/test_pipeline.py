from pathlib import Path

import pandas as pd
import pytest

from src.features.monthly_aggregation import (
    aggregate_monthly_features,
)
from src.features.pipeline import (
    FeaturePipeline,
    EXCLUDED_COLUMNS,
    prepare_features,
)

DATA_PATH = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")


def load_three_rows_with_monthly_features():
    """Load three raw rows and add their monthly summaries."""

    attrition = pd.read_excel(
        DATA_PATH,
        sheet_name="attrition_dataset",
    ).head(3)

    attendance = pd.read_excel(
        DATA_PATH,
        sheet_name="attendance_monthly_features",
    )

    payroll = pd.read_excel(
        DATA_PATH,
        sheet_name="payroll_monthly_features",
    )

    return aggregate_monthly_features(
        attrition,
        attendance,
        payroll,
    )


def test_three_raw_rows_produce_model_ready_features():
    df = load_three_rows_with_monthly_features()

    pipeline = FeaturePipeline()
    pipeline.fit(df)
    result = pipeline.transform(df)

    assert len(result) == 3
    assert result.shape[1] > 0
    assert result.select_dtypes(exclude="number").empty
    assert not result.isna().any().any()

    assert any(
        column.startswith("attendance_")
        for column in result.columns
    )
    assert any(
        column.startswith("payroll_")
        for column in result.columns
    )


def test_target_and_leakage_columns_are_excluded():
    df = load_three_rows_with_monthly_features()

    X, _, _ = prepare_features(df)

    forbidden = {
        "left_within_12m",
        "voluntary_left_within_12m",
        "exit_date",
        "resignation_date",
        "exit_reason",
        "exit_interview_done",
        "fnf_settlement_amount",
    }

    assert forbidden.isdisjoint(X.columns)
    assert forbidden.issubset(set(EXCLUDED_COLUMNS))

    pipeline = FeaturePipeline().fit(df)
    output = pipeline.transform(df)

    assert forbidden.isdisjoint(output.columns)


def test_pipeline_save_and_load(tmp_path):
    df = load_three_rows_with_monthly_features()

    pipeline = FeaturePipeline()
    pipeline.fit(df)
    expected = pipeline.transform(df)

    path = tmp_path / "features.joblib"
    pipeline.save(path)

    loaded = FeaturePipeline.load(path)
    actual = loaded.transform(df)

    pd.testing.assert_frame_equal(expected, actual)


def test_transform_before_fit_raises_error():
    pipeline = FeaturePipeline()
    df = load_three_rows_with_monthly_features()

    with pytest.raises(RuntimeError):
        pipeline.transform(df)