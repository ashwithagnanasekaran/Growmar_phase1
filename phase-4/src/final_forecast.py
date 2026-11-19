from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

HORIZON = 6
Z80 = 1.2815515655
TRAINED_THROUGH = "2026-09"
MODEL_VERSION = "phase4-forecast-1.0.0"

# --------------------------------------------------
# 1. Load and validate history
# --------------------------------------------------

history = pd.read_excel(DATA, sheet_name="monthly_headcount_history")
history["month"] = pd.to_datetime(history["month"])

if history.duplicated(["department_id", "month"]).any():
    raise ValueError("Duplicate department-month records found.")

future_index = pd.date_range(
    "2026-10-01", periods=HORIZON, freq="MS"
)

company = (
    history[history["department_id"] == "ALL"]
    .sort_values("month")
    .set_index("month")
    .asfreq("MS")
)

if company.index.max() != pd.Timestamp("2026-09-01"):
    raise ValueError("Expected company history through September 2026.")

if company[["headcount_end", "exits"]].isna().any().any():
    raise ValueError("Missing company-level target values.")

# Match the department coverage in sample_forecast.
department_ids = ["ALL"] + [
    f"D{i:02d}" for i in range(1, 11)
]

# --------------------------------------------------
# 2. Headcount seasonal trend baseline
# --------------------------------------------------

def headcount_baseline(values, future_dates):
    values = values.dropna().astype(float).tail(36)

    if len(values) != 36:
        raise ValueError("Headcount model requires 36 recent months.")

    x = np.arange(len(values), dtype=float)
    y = values.to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted

    seasonal = pd.Series(
        residuals, index=values.index
    ).groupby(values.index.month).mean()

    future_x = np.arange(
        len(values), len(values) + len(future_dates), dtype=float
    )

    trend = intercept + slope * future_x
    seasonal_adjustment = np.array([
        seasonal.get(month, 0.0)
        for month in future_dates.month
    ])

    prediction = trend + 0.5 * seasonal_adjustment

    residual_std = max(float(np.std(residuals, ddof=1)), 1.0)

    lower = prediction - Z80 * residual_std
    upper = prediction + Z80 * residual_std

    return prediction, lower, upper


# --------------------------------------------------
# 3. Company-wide exits SARIMAX
# --------------------------------------------------

def forecast_exits(values, future_dates):
    values = values.dropna().astype(float)

    model = SARIMAX(
        values,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False, maxiter=300)

    result = fitted.get_forecast(steps=len(future_dates))

    prediction = np.asarray(result.predicted_mean, dtype=float)
    interval = np.array(
        result.conf_int(alpha=0.20), dtype=float, copy=True
    )

    lower = interval[:, 0]
    upper = interval[:, 1]

    prediction = np.maximum(prediction, 0)
    lower = np.maximum(lower, 0)
    upper = np.maximum(upper, 0)

    return prediction, lower, upper


# --------------------------------------------------
# 4. Generate headcount forecasts
# --------------------------------------------------

rows = []
headcount_forecasts = {}

for department_id in department_ids:
    department = (
        history[history["department_id"] == department_id]
        .sort_values("month")
        .set_index("month")
        .asfreq("MS")
    )

    if department["headcount_end"].isna().any():
        raise ValueError(
            f"Missing headcount history for {department_id}."
        )

    prediction, lower, upper = headcount_baseline(
        department["headcount_end"], future_index
    )

    headcount_forecasts[department_id] = {
        "prediction": prediction,
        "lower": lower,
        "upper": upper
    }

    for i, month in enumerate(future_index):
        rows.append({
            "metric": "headcount_end",
            "department_id": department_id,
            "forecast_month": month,
            "forecast_value": prediction[i],
            "lower_80": lower[i],
            "upper_80": upper[i],
            "method": "36m linear trend + half-strength seasonality"
        })


# --------------------------------------------------
# 5. Forecast company-wide exits
# --------------------------------------------------

exit_prediction, exit_lower, exit_upper = forecast_exits(
    company["exits"], future_index
)

for i, month in enumerate(future_index):
    rows.append({
        "metric": "exits",
        "department_id": "ALL",
        "forecast_month": month,
        "forecast_value": exit_prediction[i],
        "lower_80": exit_lower[i],
        "upper_80": exit_upper[i],
        "method": "SARIMAX(1,0,0)x(1,0,0,12)"
    })


# --------------------------------------------------
# 6. Calculate company-wide attrition forecast
# --------------------------------------------------

company_headcount = headcount_forecasts["ALL"]

previous_actual_headcount = float(
    company.loc[pd.Timestamp("2026-09-01"), "headcount_end"]
)

for i, month in enumerate(future_index):
    if i == 0:
        start_headcount = previous_actual_headcount
        start_lower = previous_actual_headcount
        start_upper = previous_actual_headcount
    else:
        start_headcount = company_headcount["prediction"][i - 1]
        start_lower = company_headcount["lower"][i - 1]
        start_upper = company_headcount["upper"][i - 1]

    rate = (
        exit_prediction[i] / start_headcount * 100
        if start_headcount > 0 else 0.0
    )

    rate_lower = (
        exit_lower[i] / start_upper * 100
        if start_upper > 0 else 0.0
    )

    rate_upper = (
        exit_upper[i] / start_lower * 100
        if start_lower > 0 else 0.0
    )

    rows.append({
        "metric": "attrition_rate_monthly_pct",
        "department_id": "ALL",
        "forecast_month": month,
        "forecast_value": rate,
        "lower_80": rate_lower,
        "upper_80": rate_upper,
        "method": "SARIMAX exits / forecast starting headcount"
    })


# --------------------------------------------------
# 7. Build exact sample-compatible output
# --------------------------------------------------

forecast = pd.DataFrame(rows)

forecast.insert(
    0, "forecast_id",
    [f"FC-P4-{i:04d}" for i in range(1, len(forecast) + 1)]
)

forecast["trained_through"] = TRAINED_THROUGH
forecast["model_version"] = MODEL_VERSION

columns = [
    "forecast_id",
    "metric",
    "department_id",
    "forecast_month",
    "forecast_value",
    "lower_80",
    "upper_80",
    "method",
    "trained_through",
    "model_version"
]

forecast = forecast[columns].sort_values(
    ["metric", "department_id", "forecast_month"]
).reset_index(drop=True)

forecast["forecast_value"] = forecast["forecast_value"].round(3)
forecast["lower_80"] = forecast["lower_80"].round(3)
forecast["upper_80"] = forecast["upper_80"].round(3)

if len(forecast) != 78:
    raise ValueError(f"Expected 78 forecast rows, got {len(forecast)}.")

if forecast[columns].isna().any().any():
    raise ValueError("Forecast output contains missing values.")

if not (
    forecast["lower_80"] <= forecast["forecast_value"]
).all():
    raise ValueError("Some lower bounds exceed forecast values.")

if not (
    forecast["forecast_value"] <= forecast["upper_80"]
).all():
    raise ValueError("Some forecast values exceed upper bounds.")

output_path = REPORTS / "final_forecast.csv"
forecast.to_csv(output_path, index=False)


# --------------------------------------------------
# 8. Review workforce events
# --------------------------------------------------

events = pd.read_excel(DATA, sheet_name="workforce_events")
events["start_date"] = pd.to_datetime(events["start_date"])
events["end_date"] = pd.to_datetime(events["end_date"])

events["historical_or_future"] = np.where(
    events["start_date"] <= pd.Timestamp("2026-09-30"),
    "Historical event reviewed",
    "Future event"
)

events.to_csv(REPORTS / "forecast_workforce_events_review.csv", index=False)


# --------------------------------------------------
# 9. Create company headcount forecast chart
# --------------------------------------------------

historical_dates = company.index
historical_values = company["headcount_end"].astype(float)

plt.figure(figsize=(12, 6))
plt.plot(
    historical_dates, historical_values,
    label="Actual company headcount"
)

plt.plot(
    future_index, company_headcount["prediction"],
    marker="o", label="Six-month forecast"
)

plt.fill_between(
    future_index,
    company_headcount["lower"],
    company_headcount["upper"],
    alpha=0.20,
    label="80% forecast interval"
)

plt.axvline(
    pd.Timestamp("2026-09-01"),
    linestyle="--",
    label="Forecast origin"
)

plt.title("Company Headcount: Actual and Six-Month Forecast")
plt.xlabel("Month")
plt.ylabel("Headcount")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(REPORTS / "final_headcount_forecast.png", dpi=160)
plt.close()


# --------------------------------------------------
# 10. Create company exits forecast chart
# --------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(
    company.index, company["exits"],
    label="Actual monthly exits"
)

plt.plot(
    future_index, exit_prediction,
    marker="o", label="SARIMAX exits forecast"
)

plt.fill_between(
    future_index, exit_lower, exit_upper,
    alpha=0.20, label="80% forecast interval"
)

plt.axvline(
    pd.Timestamp("2026-09-01"),
    linestyle="--",
    label="Forecast origin"
)

plt.title("Company Monthly Exits: Actual and Six-Month Forecast")
plt.xlabel("Month")
plt.ylabel("Number of exits")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(REPORTS / "final_exits_forecast.png", dpi=160)
plt.close()


# --------------------------------------------------
# 11. Summary and completion checks
# --------------------------------------------------

print("\n========== FINAL FORECAST SUMMARY ==========")
print(f"Forecast months: {future_index[0].date()} to {future_index[-1].date()}")
print(f"Training data through: {TRAINED_THROUGH}")
print(f"Rows generated: {len(forecast)}")

print("\nRows by metric:")
print(forecast.groupby("metric").size().to_string())

print("\nRows by department:")
print(
    forecast[forecast["metric"] == "headcount_end"]
    .groupby("department_id").size().to_string()
)

print("\nCompany-level forecast:")
print(
    forecast[forecast["department_id"] == "ALL"][
        ["metric", "forecast_month", "forecast_value", "lower_80", "upper_80"]
    ].to_string(index=False)
)

print("\nSaved files:")
print(output_path)
print(REPORTS / "forecast_workforce_events_review.csv")
print(REPORTS / "final_headcount_forecast.png")
print(REPORTS / "final_exits_forecast.png")

print("\nFinal forecast generation completed successfully.")
