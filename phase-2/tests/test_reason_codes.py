import pandas as pd

from src.reason_codes import OUTPUT_PATH


def load_results():
    assert OUTPUT_PATH.exists(), "Run CX-07 before testing."
    return pd.read_csv(OUTPUT_PATH)


def test_output_has_expected_schema():
    results = load_results()

    expected = [
        "result_id",
        "source_dataset",
        "record_id",
        "employee_id",
        "period",
        "anomaly_score",
        "flagged",
        "reason_code",
        "reason_text",
        "money_at_risk",
        "model_version",
        "scored_at",
    ]

    assert results.columns.tolist() == expected


def test_output_contains_all_payroll_records():
    results = load_results()

    assert len(results) == 7355
    assert results["record_id"].is_unique


def test_flagged_count_matches_cx06():
    results = load_results()

    assert (results["flagged"] == "Y").sum() == 219
    assert set(results["flagged"].unique()).issubset({"Y", "N"})


def test_flagged_records_have_explanations():
    results = load_results()
    flagged = results[results["flagged"] == "Y"]

    assert flagged["reason_code"].notna().all()
    assert flagged["reason_text"].str.strip().ne("").all()
    assert flagged["model_version"].eq("iforest-payroll-1.0.0").all()


def test_money_at_risk_is_not_negative():
    results = load_results()
    amounts = results["money_at_risk"].dropna()

    assert (amounts >= 0).all()


def test_quantified_estimates_are_flagged():
    results = load_results()

    quantified = results[results["money_at_risk"].notna()]
    assert (quantified["flagged"] == "Y").all()

def test_flagged_records_are_ranked_by_money_at_risk():
    results = load_results()

    # Flagged rows must appear before unflagged rows.
    flagged_status = results["flagged"].eq("Y").astype(int)
    assert flagged_status.is_monotonic_decreasing

    # Within each group, known estimates descend and missing values come last.
    for _, group in results.groupby("flagged", sort=False):
        amounts = group["money_at_risk"]
        known = amounts.dropna()
        assert known.is_monotonic_decreasing
        if amounts.isna().any():
            first_missing = amounts.isna().to_numpy().argmax()
            assert amounts.iloc[first_missing:].isna().all()

