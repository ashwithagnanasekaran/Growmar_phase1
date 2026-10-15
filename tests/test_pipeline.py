
"""Integration tests for the reusable CortexHR feature pipeline."""

from pathlib import Path

import pandas as pd

from src.features.pipeline import FeaturePipeline


# Project paths
DATA_PATH = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")
ARTIFACT_PATH = Path("artifacts/features/features-1.0.0.joblib")


def main():
    # 1. Check that the dataset exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 2. Load a sample for pipeline testing.
    print("Loading dataset...")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="attrition_dataset",
    )

    sample = df.head(200).copy()

    if sample.empty:
        raise ValueError("The dataset sample is empty.")

    print("Sample shape:", sample.shape)

    # 3. Fit the pipeline and transform the sample.
    pipeline = FeaturePipeline()
    pipeline.fit(sample)

    print("Pipeline fitted successfully.")

    transformed_before = pipeline.transform(sample)

    print("Transformed shape:", transformed_before.shape)

    # 4. Test that the row count is preserved.
    assert len(transformed_before) == len(sample), (
        "Output row count does not match input row count."
    )

    # 5. Test that features were generated.
    assert transformed_before.shape[1] > 0, (
        "No features were produced."
    )

    # 6. Test that no missing values remain.
    missing_count = int(
        transformed_before.isna().sum().sum()
    )

    print("Missing values after transform:", missing_count)

    assert missing_count == 0, (
        "Missing values remain after preprocessing."
    )

    # 7. Test that feature names are unique.
    assert transformed_before.columns.is_unique, (
        "Output feature names are not unique."
    )

    print("Shape, missing-value, and feature-name tests passed.")

    # 8. Save the fitted pipeline.
    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline.save(ARTIFACT_PATH)

    # 9. Load the saved pipeline.
    loaded_pipeline = FeaturePipeline.load(ARTIFACT_PATH)

    # 10. Transform the same sample again.
    transformed_after = loaded_pipeline.transform(sample)

    # 11. Confirm that saving and loading preserves the output.
    pd.testing.assert_frame_equal(
        transformed_before,
        transformed_after,
    )

    print("Save/load consistency test passed.")
    print("Pipeline version:", loaded_pipeline.version)
    print("Artifact saved to:", ARTIFACT_PATH)
    print("All pipeline tests passed!")


if __name__ == "__main__":
    main()
