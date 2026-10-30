from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.payroll_detector import build_features, apply_rules
from src.chatbot_service import answer_message
from src.conversation_manager import ConversationManager


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "models_store" / "payroll" / "1.1.0"
    / "payroll_detector.joblib"
)

MODEL_BUNDLE = None
conversation_manager = ConversationManager()


USERS_FILE = ROOT / "data" / "raw" / "demo_users.csv"


def authenticate_chat_user(x_demo_user: str | None):
    if not x_demo_user or not x_demo_user.strip():
        raise HTTPException(
            status_code=401,
            detail="X-Demo-User header is required.",
        )

    users = pd.read_csv(USERS_FILE).fillna("")
    matches = users[
        users["username"].astype(str).str.lower()
        == x_demo_user.strip().lower()
    ]

    if matches.empty:
        raise HTTPException(status_code=401, detail="Unknown demo user.")

    row = matches.iloc[0]
    return str(row["employee_id"]), str(row["role"]).upper()


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class PayrollInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payroll_item_id: str = Field(min_length=1)
    employee_id: str = Field(min_length=1)
    month: str = Field(min_length=1)
    monthly_fixed_gross: float = Field(ge=0)
    gross_pay: float = Field(ge=0)
    total_deductions: float = Field(ge=0)
    net_pay: float
    overtime_pay: float = Field(ge=0)
    variable_pay: float = Field(ge=0)
    lop_days: float = Field(ge=0)
    overtime_hours_paid: float = Field(ge=0)


class AttendanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attendance_feature_id: str = Field(min_length=1)
    employee_id: str = Field(min_length=1)
    month: str = Field(min_length=1)
    scheduled_days: int = Field(ge=0)
    present_days: int = Field(ge=0)
    unpaid_absence_days: int = Field(ge=0)
    longest_unpaid_streak_days: int = Field(ge=0)
    late_arrivals: int = Field(ge=0)
    attendance_rate: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Payroll model not found: {MODEL_PATH}. "
            "Run python -m src.payroll_detector first."
        )

    MODEL_BUNDLE = joblib.load(MODEL_PATH)
    yield
    MODEL_BUNDLE = None


app = FastAPI(
    title="GrowMar Workforce Anomaly API",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "payroll_model_file_exists": MODEL_PATH.exists(),
        "payroll_model_version": "1.1.0",
    }


@app.post("/v1/anomaly/payroll")
def payroll_anomaly(payload: PayrollInput):
    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503, detail="Payroll model is not loaded."
        )

    record = pd.DataFrame([payload.model_dump()])

    # Supply explicit neutral defaults for fields absent from the API schema.
    record["run_id"] = "API_SINGLE_RECORD"
    record["overtime_eligible"] = "N"
    record["pf_employee"] = 0.0

    features = build_features(record)
    model = MODEL_BUNDLE["model"]

    prediction = int(model.predict(features)[0])
    score = float(-model.score_samples(features)[0])

    rule_flags, rule_reasons = apply_rules(record)
    reasons = rule_reasons[0]

    if prediction == -1:
        reasons.append("ISOLATION_FOREST_ANOMALY")

    return {
        "record_id": payload.payroll_item_id,
        "employee_id": payload.employee_id,
        "period": payload.month,
        "flagged": bool(bool(rule_flags.iloc[0]) or prediction == -1),
        "model_flag": bool(prediction == -1),
        "anomaly_score": score,
        "reason_codes": reasons or ["NO_RULE_TRIGGER"],
        "model_version": "iforest-payroll-1.1.0",
    }

@app.post("/v1/anomaly/attendance")
def attendance_anomaly(payload: AttendanceInput):
    if payload.present_days > payload.scheduled_days:
        raise HTTPException(
            status_code=422,
            detail="present_days cannot exceed scheduled_days.",
        )

    reasons = []

    if payload.scheduled_days > 0 and payload.present_days == 0:
        reasons.append("NO_ATTENDANCE")
    if payload.unpaid_absence_days > 0:
        reasons.append("UNPAID_ABSENCE")
    if payload.longest_unpaid_streak_days >= 3:
        reasons.append("LONG_UNPAID_STREAK")
    if payload.attendance_rate < 0.75:
        reasons.append("LOW_ATTENDANCE_RATE")
    if payload.late_arrivals >= 5:
        reasons.append("FREQUENT_LATE_ARRIVALS")

    return {
        "record_id": payload.attendance_feature_id,
        "employee_id": payload.employee_id,
        "period": payload.month,
        "flagged": bool(reasons),
        "reason_codes": reasons or ["NO_RULE_TRIGGER"],
        "model_version": "attendance-rules-0.1.0",
        "method": "transparent_rules",
    }


@app.post("/v1/chat")
def chat(
    payload: ChatInput,
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
):
    employee_id, role = authenticate_chat_user(x_demo_user)
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(
            status_code=422,
            detail="session_id cannot be blank.",
        )

    try:
        return conversation_manager.handle_message(
            session_id=session_id,
            message=payload.message,
            employee_id=employee_id,
            role=role,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000)




