import pytest

from src.hr_mock_services import (
    AccessDeniedError,
    get_leave_balance,
    get_latest_leave_request,
    get_own_payslip,
    get_employee_salary,
)


def test_employee_can_read_own_leave_balance():
    result = get_leave_balance("EMP-0002", "EMPLOYEE")
    assert result is not None
    assert result["employee_id"] == "EMP-0002"


def test_employee_cannot_read_another_leave_balance():
    with pytest.raises(AccessDeniedError):
        get_leave_balance(
            "EMP-0002", "EMPLOYEE", target_employee_id="EMP-0004"
        )


def test_employee_can_read_own_latest_leave_request():
    result = get_latest_leave_request("EMP-0002", "EMPLOYEE")
    if result is not None:
        assert result["employee_id"] == "EMP-0002"


def test_employee_can_read_own_payslip():
    result = get_own_payslip("EMP-0002", "EMPLOYEE")
    if result is not None:
        assert result["employee_id"] == "EMP-0002"


def test_role_mismatch_is_denied():
    with pytest.raises(AccessDeniedError):
        get_leave_balance("EMP-0002", "ADMIN")


def test_other_employee_salary_is_always_refused():
    with pytest.raises(AccessDeniedError):
        get_employee_salary("EMP-0002", "ADMIN", "EMP-0004")
