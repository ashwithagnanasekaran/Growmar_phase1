from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve

from src.features.monthly_aggregation import aggregate_monthly_features
from src.features.pipeline import FeaturePipeline

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models_store" / "attrition-v2" / "1.0.0"

TARGET = "left_within_12m"
RANDOM_STATE = 42

REPORTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    attrition = pd.read_excel(WORKBOOK, sheet_name="attrition_dataset")
    attendance = pd.read_excel(
        WORKBOOK, sheet_name="attendance_monthly_features"
    )
    payroll = pd.read_excel(
        WORKBOOK, sheet_name="payroll_monthly_features"
    )

    data = aggregate_monthly_features(
        attrition, attendance, payroll
    )

    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    return data


def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def run_cross_validation(name, estimator, train_df, y_train, cv):
    fold_results = []
    oof_probabilities = np.zeros(len(train_df), dtype=float)

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv.split(train_df, y_train), start=1
    ):
        fold_train = train_df.iloc[train_idx]
        fold_valid = train_df.iloc[valid_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        feature_pipeline = FeaturePipeline()
        feature_pipeline.fit(fold_train)

        X_fold_train = feature_pipeline.transform(fold_train)
        X_fold_valid = feature_pipeline.transform(fold_valid)

        model = clone(estimator)
        model.fit(X_fold_train, y_fold_train)

        probabilities = model.predict_proba(X_fold_valid)[:, 1]
        oof_probabilities[valid_idx] = probabilities

        metrics = evaluate_probabilities(
            y_fold_valid, probabilities
        )
        metrics["model"] = name
        metrics["fold"] = fold_number
        fold_results.append(metrics)

        print(
            f"{name} | Fold {fold_number}/5 | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    return pd.DataFrame(fold_results), oof_probabilities


def choose_recall_threshold(y_true, probabilities, target_recall=0.60):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    valid = np.where(recall[:-1] >= target_recall)[0]

    if len(valid) == 0:
        return 0.5

    best_index = valid[np.argmax(precision[valid])]
    return float(thresholds[best_index])


def main():
    print("Loading and aggregating the dataset...")
    data = load_dataset()

    y = data[TARGET].astype(int)
    train_df, test_df = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Training positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    baseline_models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    xgb_configs = {
        "XGBoost_A": {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
        },
        "XGBoost_B": {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.05,
        },
        "XGBoost_C": {
            "n_estimators": 200,
            "max_depth": 2,
            "learning_rate": 0.08,
        },
    }

    all_results = []
    tuning_results = []
    selected_oof = None
    selected_name = None
    selected_config = None
    best_pr_auc = -1.0

    print("\n--- BASELINE CROSS-VALIDATION ---")
    for name, estimator in baseline_models.items():
        fold_df, _ = run_cross_validation(
            name, estimator, train_df, y_train, cv
        )
        all_results.append(fold_df)

    print("\n--- XGBOOST TUNING ---")
    for name, config in xgb_configs.items():
        estimator = XGBClassifier(
            **config,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
        )

        fold_df, oof_probabilities = run_cross_validation(
            name, estimator, train_df, y_train, cv
        )
        all_results.append(fold_df)

        mean_pr_auc = float(fold_df["pr_auc"].mean())
        tuning_results.append({
            "configuration": name,
            **config,
            "mean_pr_auc": mean_pr_auc,
            "mean_roc_auc": float(fold_df["roc_auc"].mean()),
            "mean_brier_score": float(fold_df["brier_score"].mean()),
        })

        if mean_pr_auc > best_pr_auc:
            best_pr_auc = mean_pr_auc
            selected_name = name
            selected_config = config
            selected_oof = oof_probabilities.copy()

    fold_results = pd.concat(all_results, ignore_index=True)
    fold_results.to_csv(
        REPORTS / "task13_cv_fold_metrics.csv", index=False
    )

    cv_summary = (
        fold_results.groupby("model")
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_pr_auc", ascending=False)
    )

    cv_summary.to_csv(
        REPORTS / "task13_cv_summary.csv", index=False
    )

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(
        REPORTS / "task13_tuning_log.csv", index=False
    )

    print("\n--- CV SUMMARY ---")
    print(cv_summary.to_string(index=False))

    print("\n--- SELECTED CONFIGURATION ---")
    print(selected_name, selected_config)

    threshold = choose_recall_threshold(
        y_train, selected_oof, target_recall=0.60
    )
    print(f"Selected threshold for >=60% training OOF recall: {threshold:.4f}")

    final_feature_pipeline = FeaturePipeline()
    final_feature_pipeline.fit(train_df)

    X_train = final_feature_pipeline.transform(train_df)
    X_test = final_feature_pipeline.transform(test_df)

    final_model = XGBClassifier(
        **selected_config,
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )

    final_model.fit(X_train, y_train)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_probabilities(
        y_test, test_probabilities, threshold
    )
    test_metrics["model"] = "Selected XGBoost"
    test_metrics["configuration"] = selected_name
    test_metrics["threshold"] = threshold

    pd.DataFrame([test_metrics]).to_csv(
        REPORTS / "task13_test_metrics.csv", index=False
    )

    print("\n--- HELD-OUT TEST METRICS ---")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # Calibration is assessed on untouched test predictions.
    brier = brier_score_loss(y_test, test_probabilities)
    observed, predicted = calibration_curve(
        y_test,
        test_probabilities,
        n_bins=8,
        strategy="quantile",
    )

    plt.figure(figsize=(7, 6))
    plt.plot(predicted, observed, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Task 13: XGBoost Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        REPORTS / "task13_calibration_curve.png", dpi=160
    )
    plt.close()

    # Fit the benchmark table's published values alongside our same-fold results.
    benchmark_df = pd.read_excel(
        WORKBOOK, sheet_name="baseline_benchmarks"
    )
    comparison = cv_summary.merge(
        benchmark_df,
        left_on="model",
        right_on="model",
        how="left",
        suffixes=("_phase3", "_benchmark"),
    )
    comparison.to_csv(
        REPORTS / "task13_benchmark_comparison.csv", index=False
    )

    model_bundle = {
        "model": final_model,
        "feature_pipeline": final_feature_pipeline,
        "target": TARGET,
        "threshold": threshold,
        "model_version": "attrition-v2",
        "selected_configuration": selected_config,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_bundle, MODEL_DIR / "model.joblib")

    metadata = {
        "model_version": "attrition-v2",
        "selected_configuration_name": selected_name,
        "selected_configuration": selected_config,
        "threshold": threshold,
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "cv_folds": 5,
        "random_state": RANDOM_STATE,
        "cv_selection_metric": "mean PR-AUC",
        "test_metrics": test_metrics,
        "test_brier_score": float(brier),
        "calibration": "assessed; no post-hoc calibration applied",
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    report = [
        "# Task 13 – Attrition Model V2",
        "",
        "## Method",
        "- Same 80/20 stratified split and random seed as Phase 1.",
        "- Five-fold stratified CV on the training split.",
        "- Fold-specific preprocessing to prevent validation leakage.",
        "- Model selection based on mean PR-AUC.",
        "- Threshold selected from training out-of-fold predictions to target at least 60% recall.",
        "",
        "## Selected model",
        f"- Configuration: {selected_name}",
        f"- Parameters: `{selected_config}`",
        f"- Decision threshold: {threshold:.4f}",
        "",
        "## Held-out test results",
    ]
    report.extend(
        f"- {key}: {value}" for key, value in test_metrics.items()
    )
    report.extend([
        f"- Test Brier score: {brier:.4f}",
        "- Calibration was assessed on the held-out test set. No post-hoc calibration was applied in this run.",
        "",
        "## Important limitations",
        "- The model is evaluated on one stratified held-out split.",
        "- The workbook benchmark values are reference results, not a replacement for same-fold comparisons.",
        "- A higher ranking metric does not establish fairness or causal validity.",
        "- Predictions require human review and must not be used as automatic employment decisions.",
        "",
        "## Evidence files",
        "- `task13_cv_fold_metrics.csv`",
        "- `task13_cv_summary.csv`",
        "- `task13_tuning_log.csv`",
        "- `task13_test_metrics.csv`",
        "- `task13_benchmark_comparison.csv`",
        "- `task13_calibration_curve.png`",
    ])
    (REPORTS / "task13_attrition_v2_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSaved model:", MODEL_DIR / "model.joblib")
    print("Saved metadata:", MODEL_DIR / "metadata.json")
    print("Task 13 training run completed.")


if __name__ == "__main__":
    main()
