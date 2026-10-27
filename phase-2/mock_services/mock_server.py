from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query

from src.hr_mock_services import (
    AccessDeniedError,
    get_leave_balance,
    get_latest_leave_request,
    get_own_payslip,
    get_employee_salary,
)

ROOT = Path(__file__).resolve().parents[1]
USERS_FILE = ROOT / "data" / "raw" / "demo_users.csv"

app = FastAPI(title="GrowMar Mock HR Service", version="1.0.0")


def authenticate(x_demo_user: str | None):
    if not x_demo_user or not x_demo_user.strip():
        raise HTTPException(status_code=401, detail="X-Demo-User header is required.")

    users = pd.read_csv(USERS_FILE).fillna("")
    matches = users[
        users["username"].astype(str).str.lower() == x_demo_user.strip().lower()
    ]

    if matches.empty:
        raise HTTPException(status_code=401, detail="Unknown demo user.")

    row = matches.iloc[0]
    return str(row["employee_id"]), str(row["role"]).upper()


def run_service(operation, employee_id, role):
    try:
        return operation(employee_id, role)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "service": "mock-hr"}


@app.get("/api/v1/leave/balance")
def leave_balance(
    employee: str = Query(default="me"),
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
):
    employee_id, role = authenticate(x_demo_user)
    if employee not in ("me", employee_id):
        raise HTTPException(status_code=403, detail="Only your own leave balance is available.")
    result = run_service(lambda eid, r: get_leave_balance(eid, r), employee_id, role)
    if result is None:
        raise HTTPException(status_code=404, detail="Leave balance not found.")
    return result


@app.get("/api/v1/leave/requests")
def leave_requests(
    employee: str = Query(default="me"),
    limit: int = Query(default=1, ge=1, le=10),
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
):
    employee_id, role = authenticate(x_demo_user)
    if employee not in ("me", employee_id):
        raise HTTPException(status_code=403, detail="Only your own leave requests are available.")
    result = run_service(
        lambda eid, r: get_latest_leave_request(eid, r), employee_id, role
    )
    if result is None:
        return {"items": []}
    return {"items": [result][:limit]}


@app.get("/api/v1/payslips/me/{month}")
def payslip(
    month: str,
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
):
    employee_id, role = authenticate(x_demo_user)
    result = run_service(
        lambda eid, r: get_own_payslip(eid, r, period_start=month),
        employee_id,
        role,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Payslip not found for that period.")
    return result


@app.get("/api/v1/employees/me")
def my_employee_record(
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
):
    employee_id, role = authenticate(x_demo_user)
    employees = pd.read_csv(ROOT / "data" / "raw" / "employees.csv").fillna("")
    rows = employees[employees["employee_id"].astype(str) == employee_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Employee record not found.")
    return rows.iloc[0].to_dict()


@app.get("/api/v1/holidays")
def holidays():
    data = pd.read_csv(ROOT / "data" / "raw" / "holidays.csv").fillna("")
    return {"items": data.to_dict(orient="records")}


@app.get("/api/v1/salary/other")
def refuse_other_salary(
    employee_id: str = Query(...),
    x_demo_user: str | None = Header(default=None, alias="X-Demo-User"),
):
    authenticate(x_demo_user)
    try:
        get_employee_salary(employee_id)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    raise HTTPException(status_code=403, detail="Salary disclosure is prohibited.")
