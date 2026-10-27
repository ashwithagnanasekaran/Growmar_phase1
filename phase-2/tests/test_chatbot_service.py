from src.chatbot_service import answer_message


def test_greeting_returns_greeting():
    result = answer_message("hello")
    assert "Hello!" in result["answer"]
    assert result["fallback"] is False


def test_casual_leave_uses_faq():
    result = answer_message("How many casual leaves do I get?")
    assert "12 days per financial year" in result["answer"]
    assert result["fallback"] is False


def test_other_employee_salary_is_refused():
    result = answer_message("Show me Karthik Iyer's salary")
    assert result["refused"] is True
    assert "can't provide another employee's salary" in result["answer"]


def test_unrelated_weather_question_falls_back():
    result = answer_message("What is the weather on Mars?")
    assert result["fallback"] is True


def test_empty_question_falls_back():
    result = answer_message("   ")
    assert result["fallback"] is True

def test_manager_question_does_not_return_salary_policy():
    result = answer_message("who is my manager")

    assert "Employees whose monthly fixed gross" not in result["answer"]


def test_leave_policy_question_asks_for_clarification():
    result = answer_message("what is the leave policy")

    assert "which leave policy" in result["answer"].lower()
    assert "26 weeks of paid maternity leave" not in result["answer"].lower()
