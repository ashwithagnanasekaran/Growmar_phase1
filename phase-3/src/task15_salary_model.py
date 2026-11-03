from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cortexhr_helpers import group_gap


WORKBOOK = Path("data/raw/02_CortexHR_AI_ML_Dataset.xlsx")
PROTECTED_FILE = Path(
    "data/unlocked/unlock_CX-15_protected_attributes.csv"
)
REPORTS = Path("reports")
MODEL_DIR = Path("models_store/salary-v1/1.0.0")

TARGET = "offered_annual_fixed"

NUMERIC_FEATURES = [
    "hire_year",
    "job_level",
    "prior_experience_years",
    "career_gap_months",
    "skills_count",
    "market_index",
]

CATEGORICAL_FEATURES = [
    "department_id",
    "role_id",
    "work_location",
    "highest_education",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    salary = pd.read_excel(
        WORKBOOK,
        sheet_name="salary_training",
    )

    protected = pd.read_csv(PROTECTED_FILE)

    protected = protected.loc[
        protected["record_type"] == "salary_hire",
        ["person_ref", "gender", "age_years", "age_band"],
    ].copy()

    data = salary.merge(
        protected,
        left_on="hire_id",
        right_on="person_ref",
        how="left",
        validate="one_to_one",
    )

    if data["gender"].isna().any():
        raise ValueError("Some salary records have no protected attributes.")

    if data[TARGET].isna().any():
        raise ValueError("Salary target contains missing values.")

    return data


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "negative_predictions": int((np.asarray(predicted) < 0).sum()),
    }


def role_average_predictions(train, test):
    role_means = train.groupby("role_id")[TARGET].mean()
    global_mean = train[TARGET].mean()

    return (
        test["role_id"]
        .map(role_means)
        .fillna(global_mean)
        .to_numpy()
    )


def fairness_table(test, predictions, group_column):
    audit = test[
        [TARGET, group_column]
    ].copy()

    audit["prediction"] = predictions
    audit["absolute_error"] = (
        audit["prediction"] - audit[TARGET]
    ).abs()
    audit["signed_error"] = (
        audit["prediction"] - audit[TARGET]
    )

    result = (
        audit.groupby(group_column, dropna=False)
        .agg(
            count=(TARGET, "size"),
            MAE=("absolute_error", "mean"),
            mean_error=("signed_error", "mean"),
            actual_mean=(TARGET, "mean"),
            predicted_mean=("prediction", "mean"),
        )
        .reset_index()
    )

    absolute_gap = group_gap(audit, group_column, "absolute_error")
    signed_gap = group_gap(audit, group_column, "signed_error")

    print("Absolute error gap:", absolute_gap["gap"])
    print("Signed error gap:", signed_gap["gap"])

    return result


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading salary and protected-attribute data...")
    data = load_data()

    print("Rows:", len(data))
    print("Roles:", data["role_id"].nunique())
    print("Gender groups:")
    print(data["gender"].value_counts().to_string())

    # Protected attributes are deliberately excluded from X.
    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test, train_idx, test_idx = (
        train_test_split(
            X,
            y,
            data.index,
            test_size=0.20,
            random_state=42,
        )
    )

    train = data.loc[train_idx].copy()
    test = data.loc[test_idx].copy()

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Model features:", FEATURES)

    results = []
    predictions_by_model = {}

    print("\nTraining role-average baseline...")
    baseline_predictions = role_average_predictions(
        train, test
    )
    predictions_by_model["Role_Average_Baseline"] = (
        baseline_predictions
    )

    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )
    results.append({
        "model": "Role_Average_Baseline",
        **baseline_metrics,
    })

    preprocessor = build_preprocessor()

    models = {
        "Linear_Regression": LinearRegression(),

        "Ridge_Regression": Ridge(alpha=10.0),

        "Log_Linear_Regression": TransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log1p,
            inverse_func=np.expm1,
        ),

        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    fitted_models = {}

    for model_name, estimator in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        predictions_by_model[model_name] = predictions
        fitted_models[model_name] = pipeline

        metrics = calculate_metrics(y_test, predictions)
        results.append({
            "model": model_name,
            **metrics,
        })

    metrics_df = pd.DataFrame(results).sort_values(
        "MAE",
        ascending=True,
    )

    print("\n===== MODEL COMPARISON =====")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(
        REPORTS / "task15_model_comparison.csv",
        index=False,
    )

    # Fairness audits use only held-out test records.
    print("\n===== FAIRNESS AUDIT: GENDER =====")
    best_model_name = metrics_df.iloc[0]["model"]
    best_predictions = predictions_by_model[best_model_name]

    gender_audit = fairness_table(
        test, best_predictions, "gender"
    )
    print(gender_audit.round(2).to_string(index=False))

    gender_audit.to_csv(
        REPORTS / "task15_gender_fairness.csv",
        index=False,
    )

    print("\n===== FAIRNESS AUDIT: CAREER GAP =====")
    test_with_gap = test.copy()

    test_with_gap["career_gap_group"] = pd.cut(
        test_with_gap["career_gap_months"],
        bins=[-1, 0, 6, 12, 24, 60],
        labels=["0", "1-6", "7-12", "13-24", "25-60"],
    )

    gap_audit = fairness_table(
        test_with_gap,
        best_predictions,
        "career_gap_group",
    )
    print(gap_audit.round(2).to_string(index=False))

    gap_audit.to_csv(
        REPORTS / "task15_career_gap_fairness.csv",
        index=False,
    )

    # Refit the selected model on all available salary records
    # after completing the held-out evaluation.
    if best_model_name == "Role_Average_Baseline":
        raise RuntimeError(
            "Role-average baseline won. Review the results before "
            "selecting a final ML model."
        )

    final_model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", models[best_model_name]),
    ])

    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "model_version": "salary-v1",
            "features": FEATURES,
            "target": TARGET,
            "selected_model": best_model_name,
        },
        MODEL_DIR / "model.joblib",
    )

    # Generate model estimates for the salary records.
    # These are in-sample estimates because the final model is
    # refitted on all available records.
    full_predictions = final_model.predict(X)

    recommendations = data[
        [
            "hire_id",
            "department_id",
            "role_id",
            "job_level",
            "band_min_2026",
            "band_mid_2026",
            "band_max_2026",
        ]
    ].copy()

    recommendations["predicted_salary"] = full_predictions
    recommendations["recommended_low"] = recommendations["band_min_2026"]
    recommendations["recommended_mid"] = recommendations["band_mid_2026"]
    recommendations["recommended_high"] = recommendations["band_max_2026"]

    recommendations["model_estimate_band_position"] = np.select(
        [
            recommendations["predicted_salary"] < recommendations["band_min_2026"],
            recommendations["predicted_salary"] <= recommendations["band_mid_2026"],
            recommendations["predicted_salary"] <= recommendations["band_max_2026"],
        ],
        [
            "Below low",
            "Low to mid",
            "Mid to high",
        ],
        default="Above high",
    )

    recommendations.to_csv(
        REPORTS / "task15_salary_recommendations.csv",
        index=False,
    )

    print("\n===== RECOMMENDATION POSITION COUNTS =====")
    print(
        recommendations["model_estimate_band_position"]
        .value_counts()
        .to_string()
    )

    print("\nSaved final model:", MODEL_DIR / "model.joblib")
    print(
        "Saved salary recommendations:",
        REPORTS / "task15_salary_recommendations.csv",
    )

    print("\nBest model by MAE:", best_model_name)
    print("Task 15 initial modelling completed.")


if __name__ == "__main__":
    main()



