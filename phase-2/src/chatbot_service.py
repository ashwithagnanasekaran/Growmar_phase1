from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models_store/chatbot/1.0.0/intent_model.joblib"
RESPONSES_PATH = ROOT / "data/raw/chatbot_responses.csv"
FAQ_PATH = ROOT / "data/raw/hr_faq.csv"

CONFIDENCE_THRESHOLD = 0.45
FAQ_THRESHOLD = 0.50
FALLBACK = (
    "I'm not confident I understood your question. "
    "Please rephrase it or contact HR for assistance."
)

_bundle = None


def load_resources():
    global _bundle

    if _bundle is not None:
        return _bundle

    model_bundle = joblib.load(MODEL_PATH)
    responses = pd.read_csv(RESPONSES_PATH).fillna("")
    faq = pd.read_csv(FAQ_PATH).fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        strip_accents="unicode",
    )
    faq_matrix = vectorizer.fit_transform(faq["question"].astype(str))

    _bundle = {
        "model": model_bundle["model"],
        "responses": responses,
        "faq": faq,
        "vectorizer": vectorizer,
        "faq_matrix": faq_matrix,
    }
    return _bundle


def answer_message(message):
    text = str(message).strip()

    if not text:
        return {
            "intent": "fallback",
            "confidence": 0.0,
            "answer": FALLBACK,
            "fallback": True,
        }

    bundle = load_resources()
    model = bundle["model"]

    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_
    best_index = int(probabilities.argmax())
    intent = str(classes[best_index])
    confidence = float(probabilities[best_index])

    # Always refuse another person's salary before looking up an FAQ.
    if intent == "salary_of_other":
        return {
            "intent": intent,
            "confidence": confidence,
            "answer": (
                "I can't provide another employee's salary or payslip. "
                "Please use the authorized HR or payroll process."
            ),
            "fallback": False,
            "refused": True,
        }

    # A greeting should not fail only because its probability is low.
    if intent == "greet":
        match = bundle["responses"][
            bundle["responses"]["intent"] == "greet"
        ]
        greeting = (
            str(match.iloc[0]["response_template"])
            if not match.empty
            else "Hello! How can I help with your HR question?"
        )
        return {
            "intent": intent,
            "confidence": confidence,
            "answer": greeting,
            "fallback": False,
            "refused": False,
        }

    faq = bundle["faq"]
    normalized = text.lower()

    # The FAQ has specific leave topics, but no general leave-policy entry.
    # Ask for clarification instead of guessing a specific policy.
    if normalized in {
        "what is the leave policy",
        "what is your leave policy",
        "tell me the leave policy",
        "explain the leave policy",
    }:
        return {
            "intent": "hr_policy_leave",
            "confidence": confidence,
            "answer": (
                "Which leave policy would you like to know about: "
                "casual leave, sick leave, earned leave, leave without pay, "
                "or maternity leave?"
            ),
            "fallback": False,
            "refused": False,
        }

    # The supplied FAQ answers when salary is paid. Recognize
    # payslip-date wording that may differ from the FAQ question.
    if "payslip" in normalized and any(
        word in normalized for word in ("when", "date", "available", "ready")
    ):
        payment_rows = faq[
            faq["question"].str.contains(
                "when is salary paid", case=False, na=False
            )
        ]
        if not payment_rows.empty:
            return {
                "intent": intent,
                "confidence": confidence,
                "answer": str(payment_rows.iloc[0]["answer"]),
                "fallback": False,
                "refused": False,
            }

    # Reject obvious unrelated topics before searching the FAQ.
    unrelated_markers = (
        "weather", "mars", "football", "cricket", "recipe",
        "movie review", "stock market", "video game",
    )
    if any(term in normalized for term in unrelated_markers):
        return {
            "intent": "out_of_scope",
            "confidence": confidence,
            "answer": FALLBACK,
            "fallback": True,
            "refused": False,
        }

    query_vector = bundle["vectorizer"].transform([text])
    similarities = cosine_similarity(
        query_vector, bundle["faq_matrix"]
    )[0]
    faq_index = int(similarities.argmax())
    faq_score = float(similarities[faq_index])

    # Only use an FAQ when the question has a meaningful lexical
    # match. Prevent unrelated questions from receiving policy answers.
    faq_question = str(faq.iloc[faq_index]["question"]).lower()
    query_terms = set(normalized.split())
    faq_terms = set(faq_question.replace("?", "").split())
    shared_terms = query_terms & faq_terms
    meaningful_terms = shared_terms - {
        "a", "an", "the", "is", "are", "do", "does", "how",
        "what", "when", "who", "i", "my", "we", "to", "of",
        "for", "and", "in", "on", "have", "get", "me",
    }

    if faq_score >= FAQ_THRESHOLD and meaningful_terms:
        return {
            "intent": intent,
            "confidence": confidence,
            "answer": str(faq.iloc[faq_index]["answer"]),
            "fallback": False,
            "refused": False,
        }

    # These intents have fixed, non-sensitive responses in the
    # response table and should remain useful at low confidence.
    if intent == "out_of_scope" or (
        confidence < CONFIDENCE_THRESHOLD
        and intent not in {"chitchat", "help"}
    ):
        return {
            "intent": intent,
            "confidence": confidence,
            "answer": FALLBACK,
            "fallback": True,
            "refused": False,
        }

    matched = bundle["responses"][
        bundle["responses"]["intent"] == intent
    ]

    if matched.empty:
        answer = FALLBACK
    else:
        answer = str(matched.iloc[0]["response_template"])

    # Do not expose unresolved personal-data templates such as
    # {month}, {status}, {cl}, {sl}, or {el} as if they were answers.
    if "{" in answer or not answer.strip():
        answer = FALLBACK

    return {
        "intent": intent,
        "confidence": confidence,
        "answer": answer,
        "fallback": answer == FALLBACK,
        "refused": False,
    }
