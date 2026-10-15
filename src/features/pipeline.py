"""Reusable preprocessing pipeline for CortexHR."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features.engineering import engineer_features
from pathlib import Path
import joblib
# Columns that must never be used as model inputs.
TARGET_COLUMNS = [
    "left_within_12m",
    "voluntary_left_within_12m",
]

LEAKAGE_COLUMNS = [
    "exit_date",
    "resignation_date",
    "exit_reason",
    "exit_interview_done",
    "fnf_settlement_amount",
]

IDENTIFIER_COLUMNS = [
    "row_id",
    "employee_id",
    "observation_date",
    "source",
]

EXCLUDED_COLUMNS = (
    TARGET_COLUMNS
    + LEAKAGE_COLUMNS
    + IDENTIFIER_COLUMNS
)
FEATURE_VERSION = "features-1.0.0"


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical columns."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
def prepare_features(df: pd.DataFrame):
    """Engineer features and identify safe model input columns."""

    # Create a copy with engineered features.
    featured_df = engineer_features(df)

    # Keep only columns that are safe model inputs.
    feature_columns = [
        column
        for column in featured_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = featured_df[feature_columns].copy()

    # Identify numeric and categorical columns.
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return X, numeric_features, categorical_features 
class FeaturePipeline:
    """Fit, save, load, and reuse the feature preprocessing pipeline."""

    def __init__(self):
        self.preprocessor = None
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None
        self.version = FEATURE_VERSION

    def fit(self, df: pd.DataFrame):
        """Learn preprocessing rules from the supplied training data."""

        X, numeric_features, categorical_features = prepare_features(df)

        self.feature_columns = X.columns.tolist()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.preprocessor = build_preprocessing_pipeline(
            numeric_features,
            categorical_features,
        )

        self.preprocessor.fit(X)

        return self

    def transform(self, df: pd.DataFrame):
        """Apply the previously learned rules to new data."""

        if self.preprocessor is None:
            raise RuntimeError("Fit the feature pipeline before transforming data.")

        X, _, _ = prepare_features(df)

        X = X.reindex(columns=self.feature_columns)

        transformed = self.preprocessor.transform(X)

        feature_names = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            transformed,
            columns=feature_names,
            index=df.index,
        )

    def save(self, path: str | Path):
        """Save the fitted pipeline and its version."""

        if self.preprocessor is None:
            raise RuntimeError("Fit the feature pipeline before saving it.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        """Load a previously saved feature pipeline."""

        pipeline = joblib.load(path)

        if not isinstance(pipeline, cls):
            raise TypeError("The saved file is not a FeaturePipeline.")

        return pipeline