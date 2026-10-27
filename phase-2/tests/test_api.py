from fastapi.testclient import TestClient

from src.api import app

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_demo_users = pd.read_csv(ROOT / "data" / "raw" / "demo_users.csv").fillna("")
DEMO_USERNAME = str(_demo_users.iloc[0]["username"])
CHAT_HEADERS = {"X-Demo-User": DEMO_USERNAME}


def test_health():
    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_payroll_valid_request():
    payload = {
        "payroll_item_id": "PI-TEST-001",
        "employee_id": "EMP-TEST-001",
        "month": "2026-09-01",
        "monthly_fixed_gross": 50000,
        "gross_pay": 52000,
        "total_deductions": 5000,
        "net_pay": 47000,
        "overtime_pay": 0,
        "variable_pay": 2000,
        "lop_days": 0,
        "overtime_hours_paid": 0,
    }

    with TestClient(app) as client:
        response = client.post("/v1/anomaly/payroll", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["record_id"] == "PI-TEST-001"
    assert isinstance(body["flagged"], bool)
    assert body["model_version"] == "iforest-payroll-1.0.0"


def test_payroll_missing_required_field_returns_422():
    payload = {
        "payroll_item_id": "PI-TEST-002",
        "employee_id": "EMP-TEST-002",
        "month": "2026-09-01",
    }

    with TestClient(app) as client:
        response = client.post("/v1/anomaly/payroll", json=payload)

    assert response.status_code == 422


def test_payroll_negative_gross_returns_422():
    payload = {
        "payroll_item_id": "PI-TEST-003",
        "employee_id": "EMP-TEST-003",
        "month": "2026-09-01",
        "monthly_fixed_gross": 50000,
        "gross_pay": -1,
        "total_deductions": 5000,
        "net_pay": -5001,
        "overtime_pay": 0,
        "variable_pay": 0,
        "lop_days": 0,
        "overtime_hours_paid": 0,
    }

    with TestClient(app) as client:
        response = client.post("/v1/anomaly/payroll", json=payload)

    assert response.status_code == 422


def test_attendance_flags_low_attendance():
    payload = {
        "attendance_feature_id": "AF-TEST-001",
        "employee_id": "EMP-TEST-001",
        "month": "2026-09-01",
        "scheduled_days": 20,
        "present_days": 10,
        "unpaid_absence_days": 8,
        "longest_unpaid_streak_days": 4,
        "late_arrivals": 6,
        "attendance_rate": 0.5,
    }

    with TestClient(app) as client:
        response = client.post("/v1/anomaly/attendance", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["flagged"] is True
    assert "LOW_ATTENDANCE_RATE" in body["reason_codes"]


def test_attendance_rejects_present_days_above_schedule():
    payload = {
        "attendance_feature_id": "AF-TEST-002",
        "employee_id": "EMP-TEST-002",
        "month": "2026-09-01",
        "scheduled_days": 20,
        "present_days": 21,
        "unpaid_absence_days": 0,
        "longest_unpaid_streak_days": 0,
        "late_arrivals": 0,
        "attendance_rate": 1.0,
    }

    with TestClient(app) as client:
        response = client.post("/v1/anomaly/attendance", json=payload)

    assert response.status_code == 422
from src.api import conversation_manager


def test_chat_greeting():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"session_id": "api-greeting-test", "message": "hello"},
            headers=CHAT_HEADERS,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "greet"
    assert "answer" in body


def test_chat_follow_up_uses_session_context():
    with TestClient(app) as client:
        first = client.post(
            "/v1/chat",
            json={
                "session_id": "api-follow-up-test",
                "message": "how many casual leaves do I have left",
            },
            headers=CHAT_HEADERS,
        )
        second = client.post(
            "/v1/chat",
            json={
                "session_id": "api-follow-up-test",
                "message": "and sick leave?",
            },
            headers=CHAT_HEADERS,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["intent"] == "leave_balance"
    assert second.json()["used_context"] is True

def test_chat_rejects_blank_session_id():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"session_id": "   ", "message": "hello"},
            headers=CHAT_HEADERS,
        )

    assert response.status_code == 422


def test_chat_rejects_missing_authentication():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"session_id": "api-no-auth-test", "message": "hello"},
        )

    assert response.status_code == 401

