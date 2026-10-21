

from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.pipeline import (
    FeaturePipeline,
    prepare_features,
    build_preprocessing_pipeline,
)
from src.features.monthly_aggregation import aggregate_monthly_features


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
MODEL_DIR = ROOT / "models_store" / "attrition_baseline" / "1.0.0"
REPORT_DIR = ROOT / "reports"

TARGET = "left_within_12m"
RANDOM_STATE = 42
RECALL_TARGET = 0.60


class RawFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Make feature engineering and preprocessing cloneable for sklearn CV."""

    def fit(self, X, y=None):
        featured_X, numeric_features, categorical_features = prepare_features(X)
        self.feature_columns_ = featured_X.columns.tolist()
        self.preprocessor_ = build_preprocessing_pipeline(
            numeric_features, categorical_features
        )
        self.preprocessor_.fit(featured_X)
        return self

    def transform(self, X):
        if not hasattr(self, "preprocessor_"):
            raise RuntimeError(
                "Fit RawFeaturePreprocessor before transforming data."
            )

        featured_X, _, _ = prepare_features(X)
        featured_X = featured_X.reindex(columns=self.feature_columns_)
        return self.preprocessor_.transform(featured_X)


def main():
    # 1. Load the dataset and split into training and test sets.
    print("Loading dataset...")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="attrition_dataset",
    )

    attendance_df = pd.read_excel(
        DATA_PATH,
        sheet_name="attendance_monthly_features",
    )

    payroll_df = pd.read_excel(
        DATA_PATH,
        sheet_name="payroll_monthly_features",
    )

    df = aggregate_monthly_features(
        df, attendance_df, payroll_df
    )
    print(f"Rows after monthly aggregation: {len(df)}")

    y = df[TARGET].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    print("Preparing features...")

    X_train = train_df.drop(columns=[TARGET]).copy()
    X_test = test_df.drop(columns=[TARGET]).copy()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Define the three baseline models.
    models = {
        "Dummy (always stays)": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree (depth 4)": DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    # 3. Configure stratified five-fold cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    fitted_models = {}
    oof_probabilities_by_model = {}

    print("Running five-fold cross-validation...")

    # 4. Evaluate each model with fold-specific preprocessing.
    for name, model in models.items():
        fold_scores = {
            "roc_auc": [],
            "pr_auc": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        for train_idx, val_idx in cv.split(X_train, y_train):
            fold_train = X_train.iloc[train_idx]
            fold_val = X_train.iloc[val_idx]

            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Fit preprocessing only on the fold's training rows.
            fold_pipeline = FeaturePipeline()
            fold_pipeline.fit(fold_train)

            X_fold_train = fold_pipeline.transform(fold_train)
            X_fold_val = fold_pipeline.transform(fold_val)

            fold_classifier = clone(model)
            fold_classifier.fit(X_fold_train, y_fold_train)

            predictions = fold_classifier.predict(X_fold_val)

            fold_scores["precision"].append(
                precision_score(
                    y_fold_val,
                    predictions,
                    zero_division=0,
                )
            )

            fold_scores["recall"].append(
                recall_score(
                    y_fold_val,
                    predictions,
                    zero_division=0,
                )
            )

            fold_scores["f1"].append(
                f1_score(
                    y_fold_val,
                    predictions,
                    zero_division=0,
                )
            )

            probabilities = fold_classifier.predict_proba(
                X_fold_val
            )[:, 1]

            if y_fold_val.nunique() == 2:
                fold_scores["roc_auc"].append(
                    roc_auc_score(y_fold_val, probabilities)
                )
            else:
                fold_scores["roc_auc"].append(float("nan"))

            fold_scores["pr_auc"].append(
                average_precision_score(
                    y_fold_val,
                    probabilities,
                )
            )

        # Generate one out-of-fold probability per training row.
        # Preprocessing is included inside the estimator and is fitted
        # independently within each cross-validation training fold.
        cv_estimator = SklearnPipeline(
            steps=[
                ("preprocessor", RawFeaturePreprocessor()),
                ("classifier", clone(model)),
            ]
        )

        oof_probabilities_by_model[name] = cross_val_predict(
            cv_estimator,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        # Calculate the mean and standard deviation of each metric.
        row = {"model": name}

        for metric, values in fold_scores.items():
            row[f"{metric}_mean"] = pd.Series(values).mean()
            row[f"{metric}_std"] = pd.Series(values).std(ddof=0)

        results.append(row)

        # Fit the final pipeline on the complete training split.
        final_pipeline = FeaturePipeline()
        final_pipeline.fit(X_train)

        X_train_processed = final_pipeline.transform(X_train)

        final_classifier = clone(model)
        final_classifier.fit(X_train_processed, y_train)

        fitted_models[name] = {
            "feature_pipeline": final_pipeline,
            "classifier": final_classifier,
        }

    # 5. Save cross-validation metrics.
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "baseline_metrics.csv",
        index=False,
    )

    # 6. Select the supervised model with the highest mean PR-AUC.
    supervised = results_df[
        results_df["model"].isin(
            [
                "Logistic Regression",
                "Decision Tree (depth 4)",
            ]
        )
    ]

    best_name = supervised.sort_values(
        "pr_auc_mean",
        ascending=False,
    ).iloc[0]["model"]

    best_model = fitted_models[best_name]

    # 7. Select a threshold using cross_val_predict probabilities.
    print(
        "Selecting threshold from cross_val_predict "
        "out-of-fold predictions..."
    )

    oof_probabilities = oof_probabilities_by_model[best_name]

    eligible_thresholds = []

    for threshold in sorted(set(oof_probabilities), reverse=True):
        predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        recall = recall_score(
            y_train,
            predictions,
            zero_division=0,
        )

        precision = precision_score(
            y_train,
            predictions,
            zero_division=0,
        )

        if recall >= RECALL_TARGET:
            eligible_thresholds.append(
                (precision, recall, threshold)
            )

    if not eligible_thresholds:
        raise RuntimeError(
            "No threshold achieved the required recall. "
            "Check the training labels and model probabilities."
        )

    # Maximize precision while maintaining at least 60% recall.
    selected_precision, selected_recall, selected_threshold = max(
        eligible_thresholds,
        key=lambda item: (item[0], item[2]),
    )

    # 8. Evaluate the selected model on the held-out test set.
    print("Evaluating on the held-out test set...")

    test_pipeline = best_model["feature_pipeline"]
    test_classifier = best_model["classifier"]

    X_test_processed = test_pipeline.transform(X_test)

    test_probabilities = test_classifier.predict_proba(
        X_test_processed
    )[:, 1]

    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    # 9. Save the confusion matrix.
    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Stayed", "Left"],
    ).plot()

    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "baseline_confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    # 10. Save the ROC curve.
    fpr, tpr, _ = roc_curve(
        y_test,
        test_probabilities,
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {test_roc_auc:.3f}",
    )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - {best_name}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "baseline_roc_curve.png",
        dpi=150,
    )

    plt.close()

    # 11. Save fitted preprocessing pipelines and models.
    for name, model_bundle in fitted_models.items():
        safe_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name.lower(),
        ).strip("_")

        joblib.dump(
            model_bundle,
            MODEL_DIR / f"{safe_name}.joblib",
        )

    metadata = {
        "version": "1.0.0",
        "target": TARGET,
        "selected_model": best_name,
        "threshold": float(selected_threshold),
        "threshold_selection_method": "cross_val_predict_out_of_fold_predictions",
        "out_of_fold_precision": float(selected_precision),
        "out_of_fold_recall": float(selected_recall),
        "recall_target": RECALL_TARGET,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "random_state": RANDOM_STATE,
        "cross_validation_folds": 5,
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
    }

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Load the project-provided benchmark sheet.
    benchmark_df = pd.read_excel(
        DATA_PATH,
        sheet_name="baseline_benchmarks",
    )

    benchmark_lookup = benchmark_df.set_index("model")

    comparison_rows = []

    benchmark_names = {
        "Dummy (always stays)": "Dummy (always stays)",
        "Logistic Regression": (
            "Logistic Regression (class_weight=balanced)"
        ),
        "Decision Tree (depth 4)": (
            "Decision Tree (depth 4, balanced)"
        ),
    }

    for model_name, benchmark_name in benchmark_names.items():
        actual_row = results_df.loc[
            results_df["model"] == model_name
        ].iloc[0]

        benchmark_row = benchmark_lookup.loc[benchmark_name]

        comparison_rows.append(
            {
                "model": model_name,
                "cv_roc_auc": float(actual_row["roc_auc_mean"]),
                "benchmark_roc_auc": float(
                    benchmark_row["roc_auc_mean"]
                ),
                "difference": float(
                    actual_row["roc_auc_mean"]
                    - benchmark_row["roc_auc_mean"]
                ),
                "cv_pr_auc": float(actual_row["pr_auc_mean"]),
                "benchmark_pr_auc": float(
                    benchmark_row["pr_auc_mean"]
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        REPORT_DIR / "baseline_benchmark_comparison.csv",
        index=False,
    )

    # Five-sentence interpretation based on the actual results.
    logistic_auc = float(
        results_df.loc[
            results_df["model"] == "Logistic Regression",
            "roc_auc_mean",
        ].iloc[0]
    )

    tree_auc = float(
        results_df.loc[
            results_df["model"] == "Decision Tree (depth 4)",
            "roc_auc_mean",
        ].iloc[0]
    )

    dummy_auc = float(
        results_df.loc[
            results_df["model"] == "Dummy (always stays)",
            "roc_auc_mean",
        ].iloc[0]
    )

    logistic_benchmark = float(
        benchmark_lookup.loc[
            "Logistic Regression (class_weight=balanced)",
            "roc_auc_mean",
        ]
    )

    tree_benchmark = float(
        benchmark_lookup.loc[
            "Decision Tree (depth 4, balanced)",
            "roc_auc_mean",
        ]
    )

    dummy_accuracy = float(
        benchmark_lookup.loc[
            "Dummy (always stays)",
            "accuracy_mean",
        ]
    )

    comparison_note = [
        (
            f"The DummyClassifier achieved ROC-AUC {dummy_auc:.4f}; "
            f"the reference sheet reports majority-class accuracy "
            f"of {dummy_accuracy:.1%}, but the model catches no "
            "employees who leave."
        ),
        (
            f"Logistic Regression achieved mean cross-validation "
            f"ROC-AUC {logistic_auc:.4f}, compared with the workbook "
            f"reference of {logistic_benchmark:.3f}."
        ),
        (
            f"The depth-4 Decision Tree achieved mean cross-validation "
            f"ROC-AUC {tree_auc:.4f}, compared with the workbook "
            f"reference of {tree_benchmark:.3f}."
        ),
        (
            "The differences may reflect data splits, preprocessing, "
            "or evaluation details; identical folds and setups have "
            "not been confirmed."
        ),
        (
            f"Logistic Regression's held-out test ROC-AUC was "
            f"{test_roc_auc:.4f}, below its cross-validation mean, "
            "so further validation is needed before real-world HR use."
        ),
    ]

    # 12. Generate the baseline report.
    report = [
        "# Task 3 - Attrition Baseline Report",
        "",
        f"- Dataset rows: {len(df)}",
        f"- Training rows: {len(train_df)}",
        f"- Held-out test rows: {len(test_df)}",
        f"- Target: `{TARGET}`",
        "- Cross-validation: stratified five-fold CV on training data",
        "- Preprocessing is fitted independently within each CV fold.",
        "- Out-of-fold probabilities generated with `cross_val_predict`.",
        "",
        "## Cross-validation results",
        "",
        results_df.to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## Comparison with the workbook benchmark",
        "",
        comparison_df.to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## Five-sentence interpretation",
        "",
        *comparison_note,
        "",
        "## Final held-out test evaluation",
        "",
        f"- Selected model: **{best_name}**.",
        "- Selection criterion: highest mean cross-validated PR-AUC "
        "among the two supervised models.",
        f"- Selected threshold: **{selected_threshold:.4f}**.",
        "- Threshold selection: cross_val_predict out-of-fold "
        "training probabilities.",
        f"- Out-of-fold training precision: {selected_precision:.4f}.",
        f"- Out-of-fold training recall: {selected_recall:.4f}.",
        f"- Test ROC-AUC: {test_roc_auc:.4f}.",
        f"- Test PR-AUC: {test_pr_auc:.4f}.",
        f"- Test precision: {test_precision:.4f}.",
        f"- Test recall: {test_recall:.4f}.",
        f"- Test F1: {test_f1:.4f}.",
        "",
        "## Confusion matrix",
        "",
        "Rows represent actual labels; columns represent predicted labels.",
        "Label order: Stayed (0), Left (1).",
        "",
        "```text",
        str(matrix),
        "```",
        "",
        "See `baseline_confusion_matrix.png` and "
        "`baseline_roc_curve.png`.",
        "",
        "## HR interpretation",
        "",
        "The model estimates which employees may leave within 12 months. "
        "The threshold was selected using out-of-fold training predictions "
        "to target at least 60% recall. Higher recall may identify more "
        "employees who leave, but may also flag employees who stay.",
        "",
        "Predictions indicate patterns, not certainty about individual "
        "intentions. HR should review results fairly, protect employee "
        "privacy, and never use predictions as the sole basis for "
        "employment decisions.",
        "",
        "## Limitations",
        "",
        "This is a baseline model, not a validated production system. "
        "Cross-validation scores summarize performance on training folds. "
        "The final test set is held out for evaluation. Future data, "
        "fairness, and operational performance should be assessed before "
        "using the model in real HR decisions.",
        "",
    ]

    (REPORT_DIR / "baseline-report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # 13. Print the run summary.
    print("\nCross-validation results:")
    print(results_df.to_string(index=False))

    print("\nBenchmark comparison:")
    print(comparison_df.to_string(index=False))

    print(f"\nSelected model: {best_name}")
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(f"Out-of-fold precision: {selected_precision:.4f}")
    print(f"Out-of-fold recall: {selected_recall:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"\nReport saved: {REPORT_DIR / 'baseline-report.md'}")
    print(
        "Benchmark comparison saved: "
        f"{REPORT_DIR / 'baseline_benchmark_comparison.csv'}"
    )
    print(f"Models saved: {MODEL_DIR}")
    print("Task 3 baseline run completed.")


if __name__ == "__main__":
    main()