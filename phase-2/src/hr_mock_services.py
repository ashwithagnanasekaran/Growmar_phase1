from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

USERS_FILE = DATA / "demo_users.csv"
BALANCES_FILE = DATA / "leave_balances.csv"
REQUESTS_FILE = DATA / "leave_requests.csv"
PAYSLIPS_FILE = DATA / "payslips.csv"

ALLOWED_ROLES = {"EMPLOYEE", "HR", "ADMIN"}


class AccessDeniedError(Exception):
    """Raised when the caller is not permitted to access a record."""


def _load_csv(filename):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Required HR data file not found: {path}")
    return pd.read_csv(path).fillna("")


def _load_actor(employee_id, role):
    if role not in ALLOWED_ROLES:
        raise AccessDeniedError("Your role is not authorized for this service.")

    users = _load_csv("demo_users.csv")
    matches = users[users["employee_id"].astype(str) == str(employee_id)]

    if matches.empty:
        raise AccessDeniedError("The authenticated user was not found.")

    actual_role = str(matches.iloc[0]["role"]).upper()
    if actual_role != role.upper():
        raise AccessDeniedError("The supplied role does not match the user record.")

    return actual_role


def get_leave_balance(employee_id, role, target_employee_id=None):
    """Return leave balances, enforcing self-service for employees."""
    actual_role = _load_actor(employee_id, role)
    target = str(target_employee_id or employee_id)

    if actual_role == "EMPLOYEE" and target != str(employee_id):
        raise AccessDeniedError("Employees may only view their own leave balance.")

    balances = _load_csv("leave_balances.csv")
    rows = balances[balances["employee_id"].astype(str) == target]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


def get_latest_leave_request(employee_id, role, target_employee_id=None):
    """Return the latest leave request permitted for the caller."""
    actual_role = _load_actor(employee_id, role)
    target = str(target_employee_id or employee_id)

    if actual_role == "EMPLOYEE" and target != str(employee_id):
        raise AccessDeniedError("Employees may only view their own leave requests.")

    requests = _load_csv("leave_requests.csv")
    rows = requests[requests["employee_id"].astype(str) == target].copy()

    if rows.empty:
        return None

    rows["applied_on"] = pd.to_datetime(rows["applied_on"], errors="coerce")
    rows = rows.sort_values("applied_on", ascending=False)
    return rows.iloc[0].to_dict()


def get_own_payslip(employee_id, role, period_start=None):
    """Return only the authenticated employee's payslip."""
    _load_actor(employee_id, role)

    payslips = _load_csv("payslips.csv")
    rows = payslips[payslips["employee_id"].astype(str) == str(employee_id)]

    if period_start:
        rows = rows[rows["period_start"].astype(str) == str(period_start)]

    if rows.empty:
        return None

    rows = rows.sort_values("period_start", ascending=False)
    return rows.iloc[0].to_dict()


def get_employee_salary(*args, **kwargs):
    """Never disclose another employee's salary through the chatbot."""
    raise AccessDeniedError(
        "The chatbot cannot disclose another employee's salary or payslip."
    )

def get_own_manager(employee_id, role):
    """Return the authenticated employee's manager details only."""
    _load_actor(employee_id, role)

    employees = _load_csv("employees.csv")
    employee_rows = employees[
        employees["employee_id"].astype(str) == str(employee_id)
    ]

    if employee_rows.empty:
        return None

    manager_id = str(employee_rows.iloc[0]["manager_id"]).strip()
    if not manager_id:
        return None

    manager_rows = employees[
        employees["employee_id"].astype(str) == manager_id
    ]

    if manager_rows.empty:
        return None

    manager = manager_rows.iloc[0]
    return {
        "manager_id": manager_id,
        "manager_name": str(manager["full_name"]),
        "manager_email": str(manager["work_email"]),
    }
