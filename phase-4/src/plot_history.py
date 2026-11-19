from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

history = pd.read_excel(DATA, sheet_name="monthly_headcount_history")
history["month"] = pd.to_datetime(history["month"])

company = (
    history[history["department_id"] == "ALL"]
    .sort_values("month")
    .copy()
)

company["net_change"] = (
    company["headcount_end"] - company["headcount_start"]
)

# Chart 1: Headcount trend
plt.figure(figsize=(12, 6))
plt.plot(company["month"], company["headcount_end"],
         label="Month-end headcount")
plt.title("CortexHR Company Headcount (2020-2026)")
plt.xlabel("Month")
plt.ylabel("Employees")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "historical_headcount_trend.png", dpi=150)
plt.close()

# Chart 2: Hires and exits
plt.figure(figsize=(12, 6))
plt.plot(company["month"], company["hires"], label="Hires")
plt.plot(company["month"], company["exits"], label="Exits")
plt.title("Monthly Hires and Exits (2020-2026)")
plt.xlabel("Month")
plt.ylabel("Employees")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "historical_hires_exits.png", dpi=150)
plt.close()

# Chart 3: Average seasonality
company["calendar_month"] = company["month"].dt.month

seasonality = company.groupby("calendar_month").agg(
    average_hires=("hires", "mean"),
    average_exits=("exits", "mean"),
    average_net_change=("net_change", "mean")
)

plt.figure(figsize=(12, 6))
plt.plot(seasonality.index, seasonality["average_hires"],
         marker="o", label="Average hires")
plt.plot(seasonality.index, seasonality["average_exits"],
         marker="o", label="Average exits")
plt.plot(seasonality.index, seasonality["average_net_change"],
         marker="o", label="Average net change")
plt.title("Average Monthly Workforce Seasonality")
plt.xlabel("Calendar month")
plt.ylabel("Average employees")
plt.xticks(range(1, 13))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "historical_monthly_seasonality.png", dpi=150)
plt.close()

seasonality.to_csv(OUT / "historical_seasonality.csv")

print("Historical charts created:")
for file in sorted(OUT.glob("historical_*")):
    print(file)
