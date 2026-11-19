from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")

history = pd.read_excel(
    DATA_PATH,
    sheet_name="monthly_headcount_history"
)

events = pd.read_excel(
    DATA_PATH,
    sheet_name="workforce_events"
)

history["month"] = pd.to_datetime(history["month"])

company = (
    history[history["department_id"] == "ALL"]
    .sort_values("month")
    .copy()
)

company["net_change"] = (
    company["headcount_end"] - company["headcount_start"]
)

company["year"] = company["month"].dt.year
company["calendar_month"] = company["month"].dt.month

print("\n========== 1. OVERALL HEADCOUNT TREND ==========")

first = company.iloc[0]
last = company.iloc[-1]

total_growth = last["headcount_end"] - first["headcount_end"]
growth_pct = total_growth / first["headcount_end"] * 100

print(f"Starting headcount: {first['headcount_end']}")
print(f"Latest headcount:   {last['headcount_end']}")
print(f"Absolute growth:    {total_growth}")
print(f"Percentage growth:  {growth_pct:.2f}%")

print("\n========== 2. YEARLY HEADCOUNT ==========")

yearly = (
    company.groupby("year")
    .agg(
        average_headcount=("headcount_end", "mean"),
        year_end_headcount=("headcount_end", "last"),
        total_hires=("hires", "sum"),
        total_exits=("exits", "sum"),
        average_attrition=("attrition_rate_monthly_pct", "mean")
    )
    .round(2)
)

print(yearly.to_string())

print("\n========== 3. MONTHLY SEASONALITY ==========")

seasonality = (
    company.groupby("calendar_month")
    .agg(
        average_hires=("hires", "mean"),
        average_exits=("exits", "mean"),
        average_net_change=("net_change", "mean"),
        average_attrition=("attrition_rate_monthly_pct", "mean")
    )
    .round(2)
)

print(seasonality.to_string())

print("\n========== 4. UNUSUAL MONTHS ==========")

# Identify unusual monthly net changes using the IQR method.
q1 = company["net_change"].quantile(0.25)
q3 = company["net_change"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

unusual = company[
    (company["net_change"] < lower)
    | (company["net_change"] > upper)
]

print("Net change lower threshold:", round(lower, 2))
print("Net change upper threshold:", round(upper, 2))

print(
    unusual[
        ["month", "headcount_start", "headcount_end",
         "hires", "exits", "net_change"]
    ].to_string(index=False)
)

print("\n========== 5. WORKFORCE EVENTS ==========")

events["start_date"] = pd.to_datetime(events["start_date"])
events["end_date"] = pd.to_datetime(events["end_date"])

for _, event in events.iterrows():
    print(
        f"{event['event_id']} | "
        f"{event['start_date'].date()} to {event['end_date'].date()} | "
        f"{event['department_id']} | "
        f"{event['event_name']}"
    )

print("\n========== 6. LATEST 12 MONTHS ==========")

print(
    company.tail(12)[
        ["month", "headcount_end", "hires",
         "exits", "net_change", "attrition_rate_monthly_pct"]
    ].to_string(index=False)
)

print("\nHistorical analysis completed.")
