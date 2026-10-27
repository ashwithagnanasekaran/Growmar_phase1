from src.conversation_manager import ConversationManager


def test_leave_follow_up_retains_intent():
    manager = ConversationManager()

    first = manager.handle_message(
        "session-1", "how many casual leaves do I have left"
    )
    second = manager.handle_message("session-1", "and sick leave?")

    assert first["intent"] == "leave_balance"
    assert second["intent"] == "leave_balance"
    assert second["used_context"] is True


def test_sessions_keep_separate_context():
    manager = ConversationManager()

    manager.handle_message("leave-session", "how many casual leaves do I have left")
    second = manager.handle_message("new-session", "and sick leave?")

    assert second["used_context"] is False


def test_session_id_is_required():
    manager = ConversationManager()

    try:
        manager.handle_message("", "hello")
    except ValueError as exc:
        assert "session_id is required" in str(exc)
    else:
        raise AssertionError("Expected a missing session ID to be rejected")
