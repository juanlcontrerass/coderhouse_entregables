import pytest

from react_agent.llm import get_llm


def test_missing_key_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm()


def test_placeholder_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "tu_openai_api_key")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm()


def test_model_can_be_overridden_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    monkeypatch.setenv("OPENAI_MODEL", "modelo-de-prueba")
    assert get_llm().model_name == "modelo-de-prueba"
