from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models_store" / "chatbot" / "1.0.0" / "intent_model.joblib"
REPORT_DIR = ROOT / "reports" / "chatbot"

TRAIN_PATH = DATA_DIR / "chatbot_nlu_training.csv"
TEST_PATH = DATA_DIR / "chatbot_test_set.csv"
FAQ_PATH = DATA_DIR / "hr_faq.csv"
RESPONSES_PATH = DATA_DIR / "chatbot_responses.csv"


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def main():
    for path in [TRAIN_PATH, TEST_PATH, FAQ_PATH, RESPONSES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    faq = pd.read_csv(FAQ_PATH)
    responses = pd.read_csv(RESPONSES_PATH)

    required_train = {"text", "intent"}
    required_test = {"text", "expected_intent"}

    if not required_train.issubset(train.columns):
        raise ValueError("Training CSV must contain text and intent.")
    if not required_test.issubset(test.columns):
        raise ValueError("Test CSV must contain text and expected_intent.")
    if len(train) != 858 or train["intent"].nunique() != 33:
        raise ValueError(
            f"Unexpected training data: {len(train)} rows, "
            f"{train['intent'].nunique()} intents."
        )
    if train["text"].isna().any() or train["intent"].isna().any():
        raise ValueError("Training data contains missing text or labels.")
    if test["text"].isna().any() or test["expected_intent"].isna().any():
        raise ValueError("Test data contains missing text or labels.")

    X_train = train["text"].map(clean_text)
    y_train = train["intent"]
    X_test = test["text"].map(clean_text)
    y_test = test["expected_intent"]

    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                strip_accents="unicode",
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            )),
        ])),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    labels = sorted(set(y_test) | set(predictions))
    report = classification_report(
        y_test, predictions, labels=labels,
        zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "version": "1.0.0",
        "confidence_threshold": 0.45,
        "training_rows": len(train),
        "intent_count": int(train["intent"].nunique()),
    }, MODEL_PATH)

    pd.DataFrame(
        matrix, index=labels, columns=labels
    ).rename_axis("actual_intent").to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )

    pd.DataFrame(report).T.to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    errors = test.copy()
    errors["predicted_intent"] = predictions
    errors["correct"] = errors["expected_intent"] == predictions
    errors[~errors["correct"]].to_csv(
        REPORT_DIR / "misclassified_examples.csv", index=False
    )

    (REPORT_DIR / "evaluation_summary.md").write_text(
        f"# CX-09 Intent Classifier Evaluation\n\n"
        f"- Training rows: {len(train)}\n"
        f"- Training intents: {train['intent'].nunique()}\n"
        f"- Unseen test rows: {len(test)}\n"
        f"- Test accuracy: {accuracy:.4f}\n"
        f"- Model: TF-IDF + Logistic Regression\n"
        f"- Confidence threshold: 0.45 (initial setting; tune using validation data)\n"
        f"- Model version: 1.0.0\n\n"
        f"Accuracy is measured on the supplied test set. Review "
        f"misclassified_examples.csv and the confusion matrix before "
        f"treating this as production-ready.\n",
        encoding="utf-8",
    )

    print("CX-09 training and evaluation completed.")
    print("Training rows:", len(train))
    print("Training intents:", train["intent"].nunique())
    print("Unseen test rows:", len(test))
    print(f"Test accuracy: {accuracy:.4f}")
    print("Misclassified rows:", int((predictions != y_test).sum()))
    print("FAQ entries:", len(faq))
    print("Response definitions:", len(responses))
    print("Saved model:", MODEL_PATH)
    print("Evaluation report:", REPORT_DIR / "evaluation_summary.md")
    print("Confusion matrix:", REPORT_DIR / "confusion_matrix.csv")
    print("Classification report:", REPORT_DIR / "classification_report.csv")
    print("Misclassified examples:", REPORT_DIR / "misclassified_examples.csv")


if __name__ == "__main__":
    main()

