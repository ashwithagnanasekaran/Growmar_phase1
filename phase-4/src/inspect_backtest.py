from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")
OUT = Path("reports")

df = pd.read_excel(DATA, sheet_name="monthly_headcount_history")
df["month"] = pd.to_datetime(df["month"])

company = (
    df[df["department_id"] == "ALL"]
    .sort_values("month")
    .set_index("month")
    .asfreq("MS")
)

train_end = "2026-03-01"
future = pd.date_range("2026-04-01", periods=6, freq="MS")

def baseline(values, future, metric):
    recent = values.tail(36)
    x = np.arange(len(recent), dtype=float)
    y = recent.to_numpy(dtype=float)

    if metric == "headcount_end":
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        residuals = y - fitted
        seasonal = pd.Series(
            residuals, index=recent.index
        ).groupby(recent.index.month).mean()

        future_x = np.arange(len(recent), len(recent) + 6)
        prediction = (
            intercept + slope * future_x
            + 0.5 * np.array([
                seasonal.get(m, 0) for m in future.month
            ])
        )
    else:
        seasonal = recent.groupby(recent.index.month).mean()
        prediction = np.array([
            seasonal.get(m, recent.mean()) for m in future.month
        ])
        fitted = np.array([
            seasonal.get(m, recent.mean())
            for m in recent.index.month
        ])
        residuals = y - fitted

    std = max(float(np.std(residuals, ddof=1)), 0.5)

    return (
        prediction,
        np.maximum(prediction - 1.2815515655 * std, 0),
        prediction + 1.2815515655 * std
    )

def sarimax(values, future, metric):
    if metric == "headcount_end":
        order, seasonal_order, trend = (1, 1, 1), (1, 0, 0, 12), "t"
    else:
        order, seasonal_order, trend = (1, 0, 0), (1, 0, 0, 12), "c"

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
        fitted = model.fit(disp=False, maxiter=300)

    result = fitted.get_forecast(steps=6)
    prediction = np.asarray(result.predicted_mean)
    interval = np.array(result.conf_int(alpha=0.20), copy=True)

    if metric == "exits":
        prediction = np.maximum(prediction, 0)
        interval[:, 0] = np.maximum(interval[:, 0], 0)
        interval[:, 1] = np.maximum(interval[:, 1], 0)

    return prediction, interval[:, 0], interval[:, 1]

rows = []

for metric in ["headcount_end", "exits"]:
    values = company[metric].astype(float)
    training = values.loc[:train_end]
    actual = values.loc[future].to_numpy()

    for model_name, result in [
        ("Seasonal baseline", baseline(training, future, metric)),
        ("SARIMAX", sarimax(training, future, metric))
    ]:
        prediction, lower, upper = result

        for i, month in enumerate(future):
            rows.append({
                "metric": metric,
                "model": model_name,
                "month": month,
                "actual": actual[i],
                "predicted": prediction[i],
                "absolute_error": abs(actual[i] - prediction[i]),
                "lower_80": lower[i],
                "upper_80": upper[i],
                "inside_80_interval": (
                    lower[i] <= actual[i] <= upper[i]
                )
            })

result = pd.DataFrame(rows)
result.to_csv(OUT / "backtest_monthly_predictions.csv", index=False)

for metric in ["headcount_end", "exits"]:
    print(f"\n===== {metric.upper()} =====")
    subset = result[result["metric"] == metric]
    print(subset.round(2).to_string(index=False))

print("\n===== 80% INTERVAL COVERAGE =====")
coverage = (
    result.groupby(["metric", "model"])["inside_80_interval"]
    .mean().mul(100).round(1)
)
print(coverage.to_string())

print("\nSaved: reports/backtest_monthly_predictions.csv")

