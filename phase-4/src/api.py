from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import json
import logging
import math
import os
import hmac
import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import chatbot_service
from src.conversation_manager import ConversationManager
from src.payroll_detector import FEATURE_COLUMNS, build_features, apply_rules
from src import task11_resume_ranking


logger = logging.getLogger("growmar.api")


# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

PUBLIC_DEMO = os.getenv("GROWMAR_PUBLIC_DEMO", "false").strip().lower() in {
    "1", "true", "yes"
}



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load isolated resources for public or internal operation."""

    models_dir = BASE_DIR / "models_store"
    raw_dir = BASE_DIR / "data" / "raw"

    if PUBLIC_DEMO:
        public_models = models_dir / "public_demo"

        attendance_dir = public_models / "attendance" / "1.0.0"
        attrition_dir = public_models / "attrition" / "1.0.0"
        payroll_dir = public_models / "payroll" / "1.0.0"
        salary_dir = public_models / "salary" / "1.0.0"

        with open(attendance_dir / "metadata.json", encoding="utf-8") as file:
            attendance_metadata = json.load(file)

        with open(attrition_dir / "metadata.json", encoding="utf-8") as file:
            attrition_metadata = json.load(file)

        attendance_model = joblib.load(
            attendance_dir / "isolation_forest.joblib"
        )
        attrition_bundle = joblib.load(
            attrition_dir / "logistic_regression.joblib"
        )
        payroll_bundle = joblib.load(
            payroll_dir / "payroll_detector.joblib"
        )
        salary_bundle = joblib.load(
            salary_dir / "model.joblib"
        )

        # No employee-specific demo users or chatbot ML model in public mode.
        chatbot_bundle = None
        demo_users = None

        payroll_history = pd.read_csv(
            BASE_DIR / "data" / "public_demo" / "payroll_history.csv"
        )
        forecast_data = pd.read_csv(
            BASE_DIR / "reports" / "public_demo" / "final_forecast.csv"
        )

    else:
        with open(
            models_dir / "attrition" / "1.0.0" / "metadata.json",
            encoding="utf-8"
        ) as file:
            attrition_metadata = json.load(file)

        with open(
            models_dir / "attendance" / "1.0.0" / "metadata.json",
            encoding="utf-8"
        ) as file:
            attendance_metadata = json.load(file)

        attendance_model = joblib.load(
            models_dir / "attendance" / "1.0.0" / "isolation_forest.joblib"
        )
        attrition_bundle = joblib.load(
            models_dir / "attrition" / "1.0.0" / "logistic_regression.joblib"
        )
        payroll_bundle = joblib.load(
            models_dir / "payroll" / "1.1.0" / "payroll_detector.joblib"
        )
        salary_bundle = joblib.load(
            models_dir / "salary-v1" / "1.0.0" / "model.joblib"
        )
        chatbot_bundle = joblib.load(
            models_dir / "chatbot" / "1.0.0" / "intent_model.joblib"
        )

        payroll_history = pd.read_csv(
            raw_dir / "payroll_monthly_features.csv"
        )
        demo_users = pd.read_csv(raw_dir / "demo_users.csv")
        forecast_data = pd.read_csv(
            BASE_DIR / "reports" / "final_forecast.csv"
        )

    # Resume loader has its own public synthetic-data branch.
    resumes, jobs, resume_labels = task11_resume_ranking.load_data()

    # Public mode loads FAQ-only resources, not the chatbot ML model.
    chatbot_service.load_resources()

    conversation_manager = (
        None if PUBLIC_DEMO else ConversationManager()
    )

    app.state.models = {
        "attendance": attendance_model,
        "attrition": attrition_bundle,
        "payroll": payroll_bundle["model"],
        "salary": salary_bundle["model"],
        "chatbot": (
            None if PUBLIC_DEMO else chatbot_bundle["model"]
        ),
    }

    app.state.metadata = {
        "attendance_version": attendance_metadata["version"],
        "attendance_threshold": attendance_metadata["threshold"],
        "attendance_features": attendance_metadata["feature_columns"],
        "attrition_version": attrition_metadata["version"],
        "attrition_threshold": attrition_metadata["threshold"],
        "payroll_version": payroll_bundle["version"],
        "salary_version": salary_bundle["model_version"],
        "chatbot_version": (
            "public-faq-1.0.0"
            if PUBLIC_DEMO else chatbot_bundle["version"]
        ),
    }

    app.state.data = {
        "payroll_history": payroll_history,
        "demo_users": demo_users,
        "forecast": forecast_data,
        "resumes": resumes,
        "jobs": jobs,
        "resume_labels": resume_labels,
    }

    app.state.conversation_manager = conversation_manager

    print(
        "GrowMar API: public synthetic resources loaded."
        if PUBLIC_DEMO
        else "GrowMar API: internal resources loaded."
    )

    yield

    print("GrowMar API: application shutting down.")


app = FastAPI(
    title="GrowMar CortexHR API",
    description="Versioned AI and ML services for the CortexHR platform.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Require an API key for every endpoint except the health check."""

    if request.url.path == "/v1/health":
        return await call_next(request)

    # Test-only bypass. Never enable this in a public deployment.
    if os.getenv("GROWMAR_TESTING", "").strip().lower() == "true":
        return await call_next(request)

    expected_key = os.getenv("GROWMAR_API_KEY")
    supplied_key = request.headers.get("X-API-Key", "")

    if not expected_key:
        return JSONResponse(
            status_code=503,
            content={"detail": "API authentication is not configured."},
        )

    if not hmac.compare_digest(supplied_key, expected_key):
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid API key."},
        )

    return await call_next(request)


@app.get("/v1/health")
def health_check():
    """Check whether all required API service resources are initialized."""

    models = getattr(app.state, "models", {})
    metadata = getattr(app.state, "metadata", {})
    data = getattr(app.state, "data", {})
    conversation_manager = getattr(
        app.state, "conversation_manager", None
    )

    services = {
        "attendance": (
            models.get("attendance") is not None
            and metadata.get("attendance_features") is not None
        ),
        "attrition": (
            models.get("attrition") is not None
            and metadata.get("attrition_threshold") is not None
        ),
        "payroll": (
            models.get("payroll") is not None
            and data.get("payroll_history") is not None
        ),
        "salary": models.get("salary") is not None,
        "chatbot": (
            chatbot_service._bundle is not None
            if PUBLIC_DEMO
            else (
                models.get("chatbot") is not None
                and data.get("demo_users") is not None
                and conversation_manager is not None
            )
        ),
        "forecast": data.get("forecast") is not None,
        "resume": (
            data.get("resumes") is not None
            and data.get("jobs") is not None
        ),
    }

    service_status = {
        name: "loaded" if available else "unavailable"
        for name, available in services.items()
    }

    all_ready = all(services.values())

    if not all_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "api_version": "1.0.0",
                "services": service_status,
            },
        )

    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "services": service_status,
    }

@app.get("/v1/forecast")
def get_forecast(
    metric: str | None = None,
    department_id: str | None = None,
):
    """Retrieve workforce forecasts with optional filters."""

    forecast = app.state.data["forecast"].copy()

    if metric is not None:
        if metric not in forecast["metric"].unique():
            raise HTTPException(
                status_code=400,
                detail="Unsupported forecast metric."
            )
        forecast = forecast[
            forecast["metric"] == metric
        ]

    if department_id is not None:
        if department_id not in forecast["department_id"].unique():
            raise HTTPException(
                status_code=400,
                detail="Unknown department_id."
            )
        forecast = forecast[
            forecast["department_id"] == department_id
        ]

    forecast = forecast.sort_values(
        ["metric", "department_id", "forecast_month"]
    )

    return {
        "count": len(forecast),
        "filters": {
            "metric": metric,
            "department_id": department_id,
        },
        "forecasts": forecast.where(
            pd.notna(forecast), None
        ).to_dict(orient="records"),
    }


class AttritionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description="Employee model features. Do not include identifiers or target fields."
    )


@app.post("/v1/attrition/predict")
def predict_attrition(request: AttritionRequest):
    """Estimate attrition risk using the saved model and threshold."""

    forbidden_fields = {
        "employee_id",
        "row_id",
        "observation_date",
        "source",
        "left_within_12m",
        "voluntary_left_within_12m",
        "exit_date",
        "resignation_date",
        "exit_reason",
        "exit_interview_done",
        "fnf_settlement_amount",
    }

    supplied_fields = set(request.features.keys())
    prohibited = supplied_fields.intersection(forbidden_fields)

    if prohibited:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Identifiers, target labels and leakage fields are not accepted.",
                "prohibited_fields": sorted(prohibited),
            },
        )

    if not supplied_fields:
        raise HTTPException(
            status_code=422,
            detail="At least one attrition feature must be supplied.",
        )

    try:
        bundle = app.state.models["attrition"]
        feature_pipeline = bundle["feature_pipeline"]
        classifier = bundle["classifier"]

        expected_fields = set(feature_pipeline.feature_columns)

        engineered_fields = {
            "attendance_rate_feature",
            "late_arrival_count_feature",
            "overtime_hours_feature",
            "tenure_years_feature",
            "tenure_years",
            "months_since_last_revision_feature",
            "raise_overdue",
            "attendance_gap",
            "overtime_load",
        }

        aggregate_fields = {
            name for name in expected_fields
            if (
                name.startswith("attendance_")
                or name.startswith("payroll_")
            ) and name.endswith("_mean_3m")
        }

        required_fields = (
            expected_fields - engineered_fields - aggregate_fields
        )

        missing_fields = required_fields - supplied_fields
        allowed_fields = required_fields | aggregate_fields
        unknown_fields = supplied_fields - allowed_fields

        if missing_fields or unknown_fields:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "The supplied attrition features do not match the API input schema.",
                    "missing_features": sorted(missing_fields),
                    "unknown_features": sorted(unknown_fields),
                },
            )

        input_df = pd.DataFrame([request.features])
        transformed = feature_pipeline.transform(input_df)

        probability = float(classifier.predict_proba(transformed)[0][1])
        threshold = float(app.state.metadata["attrition_threshold"])
        predicted_class = int(probability >= threshold)

        return {
            "attrition_probability": probability,
            "threshold": threshold,
            "predicted_class": predicted_class,
            "prediction_label": (
                "higher_model_estimated_risk"
                if predicted_class == 1
                else "lower_model_estimated_risk"
            ),
            "target": "left_within_12m",
            "model_version": app.state.metadata["attrition_version"],
            "human_review_required": True,
            "notice": (
                "This is a statistical decision-support estimate, "
                "not a determination about an employee. "
                "Do not use it alone for employment decisions."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Attrition prediction failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during attrition prediction.",
        ) from exc
 
class AttendanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheduled_days: float = Field(gt=0, allow_inf_nan=False)
    present_days: float = Field(ge=0, allow_inf_nan=False)
    wfh_days: float = Field(ge=0, allow_inf_nan=False)
    half_days_unpaid: float = Field(ge=0, allow_inf_nan=False)
    paid_leave_days: float = Field(ge=0, allow_inf_nan=False)
    unpaid_absence_days: float = Field(ge=0, allow_inf_nan=False)
    longest_unpaid_streak_days: float = Field(ge=0, allow_inf_nan=False)
    late_arrivals: float = Field(ge=0, allow_inf_nan=False)
    avg_hours_worked: float = Field(ge=0, allow_inf_nan=False)
    overtime_hours: float = Field(ge=0, allow_inf_nan=False)
    attendance_rate: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_attendance_consistency(self):
        if self.present_days > self.scheduled_days:
            raise ValueError("present_days cannot exceed scheduled_days.")
        if self.paid_leave_days > self.scheduled_days:
            raise ValueError("paid_leave_days cannot exceed scheduled_days.")
        if self.unpaid_absence_days > self.scheduled_days:
            raise ValueError("unpaid_absence_days cannot exceed scheduled_days.")
        if self.longest_unpaid_streak_days > self.unpaid_absence_days:
            raise ValueError(
                "longest_unpaid_streak_days cannot exceed unpaid_absence_days."
            )
        if self.late_arrivals > self.present_days:
            raise ValueError("late_arrivals cannot exceed present_days.")
        if self.half_days_unpaid > self.scheduled_days:
            raise ValueError("half_days_unpaid cannot exceed scheduled_days.")
        return self


@app.post("/v1/anomaly/attendance")
def detect_attendance_anomaly(request: AttendanceRequest):
    """Detect unusual attendance patterns using the saved Isolation Forest."""

    try:
        feature_columns = app.state.metadata["attendance_features"]

        input_df = pd.DataFrame(
            [[getattr(request, column) for column in feature_columns]],
            columns=feature_columns,
        )

        model = app.state.models["attendance"]

        anomaly_score = float(-model.decision_function(input_df)[0])
        threshold = float(app.state.metadata["attendance_threshold"])

        predicted_anomaly = anomaly_score >= threshold

        protected_by_paid_leave = (
            request.scheduled_days > 0
            and request.paid_leave_days >= 0.50 * request.scheduled_days
            and request.unpaid_absence_days == 0
        )

        if protected_by_paid_leave:
            predicted_anomaly = False

        return {
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "predicted_anomaly": bool(predicted_anomaly),
            "protected_by_paid_leave": bool(protected_by_paid_leave),
            "model_version": app.state.metadata["attendance_version"],
            "human_review_required": True,
            "notice": (
                "This is an anomaly-detection signal, not proof of misconduct. "
                "Review the attendance record and relevant context before taking action."
            ),
        }

    except Exception as exc:
        logger.exception("Attendance anomaly detection failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during attendance analysis.",
        ) from exc

class PayrollAnomalyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payroll_item_id: str = Field(min_length=1)
    employee_id: str = Field(min_length=1)
    month: str = Field(min_length=7)
    run_id: str = Field(min_length=1)

    monthly_fixed_gross: float = Field(ge=0, allow_inf_nan=False)
    gross_pay: float = Field(ge=0, allow_inf_nan=False)
    total_deductions: float = Field(ge=0, allow_inf_nan=False)
    net_pay: float = Field(allow_inf_nan=False)
    overtime_pay: float = Field(ge=0, allow_inf_nan=False)
    variable_pay: float = Field(ge=0, allow_inf_nan=False)
    lop_days: float = Field(ge=0, allow_inf_nan=False)
    overtime_hours_paid: float = Field(ge=0, allow_inf_nan=False)
    overtime_eligible: str = Field(pattern="^[YN]$")
    pf_employee: float = Field(ge=0, allow_inf_nan=False)


@app.post("/v1/anomaly/payroll")
def detect_payroll_anomaly(request: PayrollAnomalyRequest):
    """Detect payroll anomalies using the saved model and historical rules."""

    try:
        record = request.model_dump()

        try:
            record["month"] = pd.to_datetime(
                record["month"], errors="raise"
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail="month must be a valid date.",
            )

        history = app.state.data["payroll_history"].copy()

        if record["payroll_item_id"] in history["payroll_item_id"].astype(str).values:
            raise HTTPException(
                status_code=409,
                detail="This payroll_item_id already exists in the historical dataset.",
            )

        combined = pd.concat(
            [history, pd.DataFrame([record])],
            ignore_index=True,
        )

        combined["_month_sort"] = pd.to_datetime(
            combined["month"], errors="raise"
        )

        combined = combined.sort_values(
            ["employee_id", "_month_sort", "payroll_item_id"]
        ).reset_index(drop=True)

        current_index = combined.index[
            combined["payroll_item_id"].astype(str)
            == record["payroll_item_id"]
        ][0]

        current_record = combined.loc[[current_index]]

        features = build_features(current_record)[FEATURE_COLUMNS]

        model = app.state.models["payroll"]

        model_flag = bool(model.predict(features)[0] == -1)
        anomaly_score = float(-model.score_samples(features)[0])

        rule_flags, reasons = apply_rules(combined)

        rule_flag = bool(rule_flags.loc[current_index])
        reason_codes = reasons[current_index]

        if model_flag:
            reason_codes.append("ISOLATION_FOREST_ANOMALY")

        return {
            "payroll_item_id": record["payroll_item_id"],
            "employee_id": record["employee_id"],
            "anomaly_score": anomaly_score,
            "model_flag": model_flag,
            "rule_flag": rule_flag,
            "anomaly_flag": bool(model_flag or rule_flag),
            "reason_codes": (
                reason_codes if reason_codes else ["NO_RULE_TRIGGER"]
            ),
            "model_version": app.state.metadata["payroll_version"],
            "human_review_required": True,
            "notice": (
                "These are payroll review signals, not confirmed payroll errors. "
                "Verify source records before making corrections."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Payroll anomaly detection failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during payroll analysis.",
        ) from exc

class SalaryPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hire_year: float = Field(allow_inf_nan=False)
    job_level: float = Field(allow_inf_nan=False)
    prior_experience_years: float = Field(ge=0, allow_inf_nan=False)
    career_gap_months: float = Field(ge=0, allow_inf_nan=False)
    skills_count: float = Field(ge=0, allow_inf_nan=False)
    market_index: float = Field(allow_inf_nan=False)

    department_id: str = Field(min_length=1)
    role_id: str = Field(min_length=1)
    work_location: str = Field(min_length=1)
    highest_education: str = Field(min_length=1)


@app.post("/v1/salary/predict")
def predict_salary(request: SalaryPredictionRequest):
    """Estimate annual fixed salary using the saved salary model."""

    try:
        features = pd.DataFrame([request.model_dump()])

        model = app.state.models["salary"]
        predicted_salary = float(model.predict(features)[0])

        if not math.isfinite(predicted_salary):
            raise ValueError("Salary model returned an invalid prediction.")

        if predicted_salary < 0:
            raise ValueError("Salary model returned a negative prediction.")

        return {
            "estimated_annual_fixed_salary": round(predicted_salary, 2),
            "target": "offered_annual_fixed",
            "model_version": app.state.metadata["salary_version"],
            "human_review_required": True,
            "notice": (
                "This is a model-estimated salary, not an approved compensation offer. "
                "Review role, experience, market conditions and applicable pay policies "
                "before making compensation decisions."
            ),
        }

    except Exception as exc:
        logger.exception("Salary prediction failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during salary prediction.",
        ) from exc


class ResumeRankingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    top_n: int = Field(default=10, ge=1, le=50)


@app.post("/v1/resume/rank")
def rank_resumes(request: ResumeRankingRequest):
    """Rank resumes against a selected job description."""

    try:
        resumes = app.state.data["resumes"]
        jobs = app.state.data["jobs"]

        selected_job = jobs[
            jobs["job_id"].astype(str) == request.job_id
        ]

        if selected_job.empty:
            raise HTTPException(
                status_code=404,
                detail="The requested job_id was not found.",
            )

        rankings = task11_resume_ranking.build_rankings(
            resumes,
            selected_job,
        )

        top_candidates = rankings.head(request.top_n)

        candidates = []

        for _, row in top_candidates.iterrows():
            candidates.append({
                "resume_id": str(row["resume_id"]),
                "job_id": str(row["job_id"]),
                "job_title": str(row["job_title"]),
                "overall_rank": int(row["overall_rank"]),
                "total_experience_years": float(
                    row["total_experience_years"]
                ),
                "required_skill_coverage": float(
                    row["required_skill_coverage"]
                ),
                "nice_skill_coverage": float(
                    row["nice_skill_coverage"]
                ),
                "tfidf_similarity": float(
                    row["tfidf_similarity"]
                ),
                "experience_fit": float(
                    row["experience_fit"]
                ),
                "ranking_score": float(row["ranking_score"]),
            })

        return {
            "job_id": request.job_id,
            "job_title": str(selected_job.iloc[0]["job_title"]),
            "candidates_returned": len(candidates),
            "ranking_method": "Phase 3 weighted resume ranking",
            "ranking_weights": task11_resume_ranking.WEIGHTS,
            "candidates": candidates,
            "human_review_required": True,
            "notice": (
                "Rankings are decision-support recommendations, not "
                "hiring decisions. Review candidate qualifications "
                "individually. Do not automatically reject candidates "
                "based on ranking position."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Resume ranking failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during resume ranking.",
        ) from exc


class ChatbotMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


@app.post("/v1/chatbot/message")
def chatbot_message(
    request: ChatbotMessageRequest,
    x_demo_user: str | None = Header(
        default=None, alias="X-Demo-User"
    ),
):
    """Handle an authenticated HR chatbot conversation."""

    try:
        if PUBLIC_DEMO:
            result = chatbot_service.answer_public_faq(
                request.message.strip()
            )
            return {
                "session_id": request.session_id,
                "authenticated": False,
                "mode": "public_faq_only",
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "answer": result.get("answer"),
                "fallback": result.get("fallback", False),
                "refused": result.get("refused", False),
                "used_context": False,
                "human_review_required": True,
                "notice": (
                    "This public demonstration answers general HR FAQs only. "
                    "It cannot access personal employee records."
                ),
            }

        if not x_demo_user:
            raise HTTPException(
                status_code=422,
                detail="The X-Demo-User header is required in local mode.",
            )

        users = app.state.data["demo_users"]

        matches = users[
            users["username"].astype(str).str.strip().str.casefold()
            == x_demo_user.strip().casefold()
        ]

        if matches.empty:
            raise HTTPException(
                status_code=401,
                detail="Unknown demo user. Authentication failed.",
            )

        user = matches.iloc[0]

        # Identity and role come only from the trusted server-side CSV.
        employee_id = str(user["employee_id"])
        role = str(user["role"]).upper()

        # Namespace sessions by username to prevent cross-user session reuse.
        safe_session_id = (
            f"{x_demo_user.strip().casefold()}:{request.session_id}"
        )

        result = app.state.conversation_manager.handle_message(
            session_id=safe_session_id,
            message=request.message.strip(),
            employee_id=employee_id,
            role=role,
        )

        return {
            "session_id": request.session_id,
            "authenticated": True,
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "answer": result.get("answer"),
            "fallback": result.get("fallback", False),
            "refused": result.get("refused", False),
            "used_context": result.get("used_context", False),
            "human_review_required": True,
            "notice": (
                "HR chatbot responses are informational. "
                "Verify important employment and payroll information "
                "with the appropriate HR team."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chatbot request failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the chatbot request.",
        ) from exc
