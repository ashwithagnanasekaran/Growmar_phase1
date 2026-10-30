import numpy as np
import pandas as pd

from src.payroll_detector import (
    DATA_PATH,
    RESULT_PATH,
    build_features,
    apply_rules,
)


def load_payroll():
    return pd.read_csv(DATA_PATH)


def test_payroll_dataset_has_expected_columns():
    df = load_payroll()
    assert len(df) == 7355
    assert "payroll_item_id" in df.columns
    assert "net_pay" in df.columns


def test_features_have_expected_columns_and_no_infinity():
    df = load_payroll()
    features = build_features(df)

    expected = {
        "gross_fixed_ratio",
        "deduction_gross_ratio",
        "net_gross_ratio",
        "overtime_fixed_ratio",
        "variable_fixed_ratio",
        "lop_days",
        "overtime_hours_paid",
    }

    assert set(features.columns) == expected
    assert not np.isinf(features.to_numpy()).any()


def test_missing_ratios_are_handled_as_missing_values():
    df = pd.DataFrame({
        "monthly_fixed_gross": [0.0, 1000.0],
        "gross_pay": [0.0, 1200.0],
        "total_deductions": [0.0, 100.0],
        "net_pay": [0.0, 1100.0],
        "overtime_pay": [0.0, 0.0],
        "variable_pay": [0.0, 100.0],
        "lop_days": [0.0, 1.0],
        "overtime_hours_paid": [0.0, 2.0],
    })

    features = build_features(df)

    assert pd.isna(features.loc[0, "gross_fixed_ratio"])
    assert pd.isna(features.loc[0, "deduction_gross_ratio"])
    assert features.loc[1, "gross_fixed_ratio"] == 1.2


def test_payroll_rules_flag_expected_rows():
    df = load_payroll()
    rule_flags, reasons = apply_rules(df)

    assert int(rule_flags.sum()) == 100

    expected_signals = {
        "WRONG_DECIMAL_SUSPECTED": 15,
        "OVERTIME_SPIKE": 13,
        "SUDDEN_PF_DROP": 14,
        "SUDDEN_FIXED_SALARY_INCREASE": 15,
    }

    for code, expected_count in expected_signals.items():
        actual_count = sum(code in str(reason) for reason in reasons)
        assert actual_count == expected_count, (
            f"{code}: expected {expected_count}, got {actual_count}"
        )
    assert len(reasons) == len(df)


def test_detector_outputs_are_consistent():
    results = pd.read_csv(RESULT_PATH)

    assert len(results) == 7355
    assert results["payroll_item_id"].is_unique
    assert results["anomaly_flag"].equals(
        results["model_flag"] | results["rule_flag"]
    )
    assert results["anomaly_score"].notna().all()
    assert results["reason_codes"].notna().all()

