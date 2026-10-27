import pytest

from src.conversation_manager import ConversationManager
from src.hr_mock_services import AccessDeniedError


def test_leave_follow_up_returns_employee_balance():
    manager = ConversationManager()

    manager.handle_message(
        "employee-session",
        "how many casual leaves do I have left",
        employee_id="EMP-0002",
        role="EMPLOYEE",
    )
    result = manager.handle_message(
        "employee-session",
        "and sick leave?",
        employee_id="EMP-0002",
        role="EMPLOYEE",
    )

    assert result["intent"] == "leave_balance"
    assert result["used_context"] is True
    assert "sick leave balance is 12 days" in result["answer"]


def test_leave_balance_requires_identity():
    manager = ConversationManager()
    result = manager.handle_message(
        "anonymous-session",
        "how many casual leaves do I have left",
    )
    assert "sign in" in result["answer"].lower()


def test_forged_role_is_rejected():
    manager = ConversationManager()

    with pytest.raises(AccessDeniedError):
        manager.handle_message(
            "forged-role-session",
            "how many casual leaves do I have left",
            employee_id="EMP-0002",
            role="ADMIN",
        )
