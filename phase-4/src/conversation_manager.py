import time

from src.chatbot_service import answer_message


FOLLOW_UP_RULES = [
    ("show details", "payroll_flags"),
    ("what about ", "salary_band_query"),
    ("for support", "headcount_forecast"),
    ("for engineering", "attrition_risk_query"),
    ("and in ", None),
    ("why is it high", None),
    ("and sick leave", "leave_balance"),
    ("any unusual days", "unusual_attendance"),
]

NEUTRAL_INTENTS = {"thanks", "goodbye"}


def resolve_follow_up(message, previous_intent=None, predicted_intent=None):
    text = str(message).strip()
    normalized = text.lower()

    if not text:
        return {
            "message": text,
            "intent": None,
            "used_context": False,
        }

    # Resolve short follow-ups using the previous conversation intent.
    if previous_intent == "leave_balance" and normalized in {
        "can you check it for me",
        "can you check it for me?",
        "check it for me",
        "what about my balance",
        "what about my leave",
        "and my leave balance",
    }:
        return {"message": text, "intent": "leave_balance", "used_context": True}

    # Recognize specific leave-type follow-ups after a leave-balance question.
    if (
        previous_intent == "leave_balance"
        and any(
            term in normalized
            for term in (
                "sick leave",
                "casual leave",
                "earned leave",
                "sick",
                "casual",
                "earned",
            )
        )
    ):
        return {"message": text, "intent": "leave_balance", "used_context": True}

    # Specific follow-up rules from the supplied dialogue examples.
    if normalized.startswith("show details") and previous_intent == "payroll_flags":
        return {"message": text, "intent": "payroll_flags", "used_context": True}

    if normalized.startswith("what about ") and previous_intent == "salary_band_query":
        return {"message": text, "intent": "salary_band_query", "used_context": True}

    if normalized.startswith("for support") and previous_intent == "headcount_forecast":
        return {"message": text, "intent": "headcount_forecast", "used_context": True}

    if normalized.startswith("and in ") and previous_intent == "attrition_risk_query":
        return {"message": text, "intent": "attrition_risk_query", "used_context": True}

    if normalized.startswith("why is it high") and previous_intent == "attrition_risk_query":
        return {"message": text, "intent": "attrition_risk_query", "used_context": True}

    if normalized.startswith("and sick leave") and previous_intent == "leave_balance":
        return {"message": text, "intent": "leave_balance", "used_context": True}

    if normalized.startswith("any unusual days") and previous_intent == "attendance_summary":
        return {"message": text, "intent": "unusual_attendance", "used_context": True}

    if predicted_intent is None:
        predicted_intent = answer_message(text)["intent"]

    return {
        "message": text,
        "intent": predicted_intent,
        "used_context": False,
    }


class ConversationManager:
    def __init__(self, session_ttl_seconds=1800, max_sessions=1000):
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")

        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")

        self.sessions = {}
        self.session_ttl_seconds = session_ttl_seconds
        self.max_sessions = max_sessions

    def _remove_expired_sessions(self, now):
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if now - session["last_activity"] >= self.session_ttl_seconds
        ]

        for session_id in expired_sessions:
            del self.sessions[session_id]

    def handle_message(self, session_id, message, employee_id=None, role=None):
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id is required")

        now = time.monotonic()
        self._remove_expired_sessions(now)

        session_id = str(session_id)
        session = self.sessions.get(session_id)

        if session is None:
            if len(self.sessions) >= self.max_sessions:
                oldest_session_id = min(
                    self.sessions,
                    key=lambda key: self.sessions[key]["last_activity"],
                )
                del self.sessions[oldest_session_id]

            session = {
                "previous_intent": None,
                "last_message": None,
                "last_activity": now,
            }
            self.sessions[session_id] = session
        else:
            session["last_activity"] = now

        result = answer_message(message)

        resolved = resolve_follow_up(
            message,
            previous_intent=session["previous_intent"],
            predicted_intent=result["intent"],
        )

        # Handle the authenticated user's manager lookup.
        if resolved["intent"] == "my_manager":
            result["intent"] = "my_manager"
            result["used_context"] = resolved["used_context"]

            if not employee_id or not role:
                result["answer"] = "Please sign in to view your own HR information."
                result["fallback"] = True
            else:
                from src.hr_mock_services import get_own_manager

                manager = get_own_manager(employee_id, role)
                if manager is None:
                    result["answer"] = "I couldn't find your manager record. Please contact HR."
                    result["fallback"] = True
                else:
                    result["answer"] = (
                        f"Your manager is {manager['manager_name']}. "
                        f"You can contact them at {manager['manager_email']}."
                    )
                    result["fallback"] = False

            result["refused"] = False

        # Handle leave balance before the generic follow-up placeholder.
        elif resolved["intent"] == "leave_balance":
            result["intent"] = "leave_balance"
            result["used_context"] = resolved["used_context"]

            if not employee_id or not role:
                result["answer"] = "Please sign in to view your own leave balance."
                result["fallback"] = True
            else:
                result["answer"] = answer_leave_balance_follow_up(
                    employee_id, role, message
                )
                result["fallback"] = False
            result["refused"] = False

        elif resolved["used_context"]:
            result["intent"] = resolved["intent"]
            result["used_context"] = True
            result["fallback"] = True
            result["answer"] = (
                "I understood this as a follow-up to your previous "
                f"'{resolved['intent']}' request. The corresponding "
                "service response is not connected yet."
            )
        else:
            result["used_context"] = False

        if (
            result["intent"] not in NEUTRAL_INTENTS
            and not result.get("refused", False)
        ):
            session["previous_intent"] = result["intent"]
            session["last_message"] = str(message)

        return result


def answer_leave_balance_follow_up(employee_id, role, message):
    """Answer a leave-balance question using the protected mock service."""
    from src.hr_mock_services import get_leave_balance

    normalized = str(message).lower()
    balance = get_leave_balance(employee_id, role)

    if balance is None:
        return "I couldn't find a leave-balance record for your account."

    if "sick" in normalized:
        return f"Your sick leave balance is {balance['sl_balance']} days."

    if "earned" in normalized or " el" in f" {normalized}":
        return f"Your earned leave balance is {balance['el_balance']} days."

    if "casual" in normalized or " cl" in f" {normalized}":
        return f"Your casual leave balance is {balance['cl_balance']} days."

    return (
        f"Your balances are: casual leave {balance['cl_balance']} days, "
        f"sick leave {balance['sl_balance']} days, "
        f"earned leave {balance['el_balance']} days."
    )
