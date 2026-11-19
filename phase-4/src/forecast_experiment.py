from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

HORIZON = 6
BACKTEST_END = pd.Timestamp("2026-03-01")
Z80 = 1.2815515655

# --------------------------------------------------
# 1. Load company-level monthly history
# --------------------------------------------------

history = pd.read_excel(
    DATA,
    sheet_name="monthly_headcount_history"
)

history["month"] = pd.to_datetime(history["month"])

company = (
    history[history["department_id"] == "ALL"]
    .sort_values("month")
    .set_index("month")
    .asfreq("MS")
)

series_data = {
    "headcount_end": company["headcount_end"].astype(float),
    "exits": company["exits"].astype(float),
}

train = company.loc[:BACKTEST_END]
actual = company.loc["2026-04-01":"2026-09-01"]

if len(train) != 75 or len(actual) != 6:
    raise ValueError(
        f"Unexpected split: training={len(train)}, "
        f"backtest={len(actual)}"
    )

# --------------------------------------------------
# 2. Seasonal trend baseline for headcount
# --------------------------------------------------

def headcount_baseline(values, future_index):
    recent = values.tail(36)

    x = np.arange(len(recent), dtype=float)
    y = recent.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    fitted = intercept + slope * x
    residuals = y - fitted

    seasonal = pd.Series(
        residuals,
        index=recent.index
    ).groupby(recent.index.month).mean()

    future_x = np.arange(
        len(recent),
        len(recent) + len(future_index),
        dtype=float
    )

    trend = intercept + slope * future_x

    seasonal_adjustment = np.array([
        seasonal.get(month, 0.0)
        for month in future_index.month
    ])

    prediction = trend + 0.5 * seasonal_adjustment

    residual_std = max(
        float(np.std(residuals, ddof=1)),
        1.0
    )

    return (
        prediction,
        prediction - Z80 * residual_std,
        prediction + Z80 * residual_std
    )


# --------------------------------------------------
# 3. Same-month seasonal baseline for exits
# --------------------------------------------------

def exits_baseline(values, future_index):
    recent = values.tail(36)

    monthly_average = (
        recent.groupby(recent.index.month).mean()
    )

    prediction = np.array([
        monthly_average.get(month, recent.mean())
        for month in future_index.month
    ])

    fitted = np.array([
        monthly_average.get(month, recent.mean())
        for month in recent.index.month
    ])

    residuals = recent.to_numpy() - fitted

    residual_std = max(
        float(np.std(residuals, ddof=1)),
        0.5
    )

    prediction = np.maximum(prediction, 0)

    return (
        prediction,
        np.maximum(prediction - Z80 * residual_std, 0),
        prediction + Z80 * residual_std
    )


# --------------------------------------------------
# 4. SARIMAX model
# --------------------------------------------------

def sarimax_forecast(values, future_index, metric):
    if metric == "headcount_end":
        order = (1, 1, 1)
        seasonal_order = (1, 0, 0, 12)
        trend = "t"
    else:
        order = (1, 0, 0)
        seasonal_order = (1, 0, 0, 12)
        trend = "c"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(
            disp=False,
            maxiter=300
        )

    forecast = fitted.get_forecast(steps=len(future_index))

    prediction = np.asarray(forecast.predicted_mean)
    interval = np.asarray(
        forecast.conf_int(alpha=0.20)
    )

    lower = interval[:, 0]
    upper = interval[:, 1]

    if metric == "exits":
        prediction = np.maximum(prediction, 0)
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)

    return prediction, lower, upper


# --------------------------------------------------
# 5. Forecast evaluation metrics
# --------------------------------------------------

def evaluate(actual_values, predicted_values):
    actual_values = np.asarray(actual_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)

    errors = actual_values - predicted_values

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    denominator = np.sum(np.abs(actual_values))
    wape = (
        np.sum(np.abs(errors)) / denominator * 100
        if denominator > 0 else np.nan
    )

    return mae, rmse, wape


# --------------------------------------------------
# 6. Six-month backtest
# --------------------------------------------------

future_backtest = pd.date_range(
    "2026-04-01",
    periods=HORIZON,
    freq="MS"
)

backtest_rows = []
selected_models = {}

for metric, values in series_data.items():

    training_values = values.loc[:BACKTEST_END]
    actual_values = values.loc[future_backtest].to_numpy()

    if metric == "headcount_end":
        baseline = headcount_baseline
    else:
        baseline = exits_baseline

    baseline_result = baseline(
        training_values,
        future_backtest
    )

    sarimax_result = sarimax_forecast(
        training_values,
        future_backtest,
        metric
    )

    for model_name, result in [
        ("Seasonal baseline", baseline_result),
        ("SARIMAX", sarimax_result)
    ]:
        prediction, lower, upper = result

        mae, rmse, wape = evaluate(
            actual_values,
            prediction
        )

        backtest_rows.append({
            "metric": metric,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "WAPE_pct": wape
        })

    baseline_mae = evaluate(
        actual_values,
        baseline_result[0]
    )[0]

    sarimax_mae = evaluate(
        actual_values,
        sarimax_result[0]
    )[0]

    selected_models[metric] = (
        "Seasonal baseline"
        if baseline_mae <= sarimax_mae
        else "SARIMAX"
    )

# --------------------------------------------------
# 7. Save backtest results
# --------------------------------------------------

comparison = pd.DataFrame(backtest_rows)

comparison.to_csv(
    REPORTS / "backtest_comparison.csv",
    index=False
)

print("\n========== SIX-MONTH BACKTEST ==========")
print(comparison.round(3).to_string(index=False))

print("\n========== MODEL SELECTION ==========")

for metric, model_name in selected_models.items():
    print(f"{metric}: {model_name}")

# --------------------------------------------------
# 8. Workforce event review
# --------------------------------------------------

events = pd.read_excel(
    DATA,
    sheet_name="workforce_events"
)

events["start_date"] = pd.to_datetime(events["start_date"])
events["end_date"] = pd.to_datetime(events["end_date"])

print("\n========== COMPANY-WIDE EVENTS ==========")

company_events = events[
    events["department_id"] == "ALL"
]

for _, event in company_events.iterrows():
    print(
        f"{event['event_id']} | "
        f"{event['start_date'].date()} to "
        f"{event['end_date'].date()} | "
        f"{event['event_name']}"
    )

print("\nNo future company-wide workforce events are listed.")
print("Future event adjustments will not be invented.")

print("\nBacktesting experiment completed.")
print("Report saved: reports/backtest_comparison.csv")
