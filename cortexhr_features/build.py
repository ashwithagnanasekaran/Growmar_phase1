"""Build the reusable CortexHR feature pipeline."""

import json
from pathlib import Path

import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.features.monthly_aggregation import aggregate_monthly_features

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
ARTIFACT_PATH = ROOT / "artifacts" / "features" / "features-1.0.0.joblib"
METADATA_PATH = Path(__file__).resolve().parent / "metadata.json"


def main():
    print("Loading CortexHR dataset...")

    df = pd.read_excel(
        DATA_PATH, sheet_name="attrition_dataset"
    )
    attendance_df = pd.read_excel(
        DATA_PATH, sheet_name="attendance_monthly_features"
    )
    payroll_df = pd.read_excel(
        DATA_PATH, sheet_name="payroll_monthly_features"
    )
    df = aggregate_monthly_features(
        df, attendance_df, payroll_df
    )
    print(f"Rows after monthly aggregation: {len(df)}")

    # Fit only on the training portion to avoid data leakage.
    from sklearn.model_selection import train_test_split

    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    pipeline.save(ARTIFACT_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["artifact_path"] = str(ARTIFACT_PATH.relative_to(ROOT))
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Pipeline saved to: {ARTIFACT_PATH}")
    print("Feature pipeline build completed.")


if __name__ == "__main__":
    main()