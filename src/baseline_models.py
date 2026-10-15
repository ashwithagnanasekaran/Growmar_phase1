
"""Train and evaluate baseline employee attrition models."""

from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.tree import DecisionTreeClassifier

from src.features.pipeline import FeaturePipeline


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "02_CortexHR_AI_ML_Dataset.xlsx"
MODEL_DIR = ROOT / "models_store" / "attrition_baseline" / "1.0.0"
REPORT_DIR = ROOT / "reports"
TARGET = "left_within_12m"
RANDOM_STATE = 42
RECALL_TARGET = 0.60


def main():
    # 1. Load dataset and create a stratified train/test split.
    print("Loading dataset...")
    df = pd.read_excel(
        DATA_PATH,
        sheet_name="attrition_dataset",
    )

    y = df[TARGET].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # 2. Fit feature preprocessing on training data only.
    print("Preparing features...")
    feature_pipeline = FeaturePipeline()
    feature_pipeline.fit(train_df)

    X_train = feature_pipeline.transform(train_df)
    X_test = feature_pipeline.transform(test_df)

    # 3. Define the three baseline models.
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Compare models using stratified five-fold cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
    }

    results = []
    fitted_models = {}

    print("Running five-fold cross-validation...")

    for name, model in models.items():
        scores = cross_validate(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

        # Fit each model on the complete training set.
        model.fit(X_train, y_train)
        fitted_models[name] = model

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "baseline_metrics.csv",
        index=False,
    )

    # 5. Select the supervised model with the highest CV PR-AUC.
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

    # 6. Choose the decision threshold using out-of-fold training
    # probabilities rather than predictions on data used to fit
    # the same model.
    print("Selecting threshold from out-of-fold predictions...")

    oof_probabilities = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

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

    # Maximize precision while satisfying the recall requirement.
    selected_precision, selected_recall, selected_threshold = max(
        eligible_thresholds,
        key=lambda item: (item[0], item[2]),
    )

    # 7. Evaluate the selected model exactly once on the held-out test set.
    print("Evaluating on the held-out test set...")

    test_probabilities = best_model.predict_proba(X_test)[:, 1]

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

    # 8. Save the confusion matrix.
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

    # 9. Save the ROC curve.
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

    # 10. Save all fitted models for reproducibility.
    for name, model in fitted_models.items():
        safe_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name.lower(),
        ).strip("_")

        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib",
        )

    metadata = {
        "version": "1.0.0",
        "target": TARGET,
        "selected_model": best_name,
        "threshold": float(selected_threshold),
        "threshold_selection_method": "out_of_fold_training_predictions",
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

    # 11. Generate the baseline report for the HR team.
    report = [
        "# Task 3 - Attrition Baseline Report",
        "",
        f"- Dataset rows: {len(df)}",
        f"- Training rows: {len(train_df)}",
        f"- Held-out test rows: {len(test_df)}",
        f"- Target: `{TARGET}`",
        "- Cross-validation: stratified five-fold CV on training data",
        "",
        "## Cross-validation results",
        "",
        results_df.to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## Final held-out test evaluation",
        "",
        f"- Selected model: **{best_name}**.",
        "- Selection criterion: highest mean cross-validated PR-AUC "
        "among the two supervised models.",
        f"- Selected threshold: **{selected_threshold:.4f}**.",
        "- Threshold selection: out-of-fold training predictions.",
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
        "The decision threshold was selected using out-of-fold training "
        "predictions to target at least 60% recall. Higher recall may "
        "help HR identify more employees who could leave, but can also "
        "flag employees who stay.",
        "",
        "These predictions indicate patterns, not certainty about an "
        "individual's intentions. HR should review results fairly, "
        "protect employee privacy, and never use model predictions as "
        "the sole basis for employment decisions.",
        "",
        "## Limitations",
        "",
        "This is a baseline model, not a validated production system. "
        "Preprocessing was fitted on the training split before model "
        "cross-validation, so cross-validation scores may be optimistic. "
        "A stricter evaluation should fit preprocessing separately "
        "within each cross-validation fold. Performance and fairness "
        "should be checked on future data before operational use.",
        "",
    ]

    (REPORT_DIR / "baseline-report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # 12. Print the run summary.
    print("\nCross-validation results:")
    print(results_df.to_string(index=False))

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
    print(f"Models saved: {MODEL_DIR}")
    print("Task 3 baseline run completed.")


if __name__ == "__main__":
    main()

