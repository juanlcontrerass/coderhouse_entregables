"""Creación del LLM de OpenAI."""

import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from react_agent.config import DEFAULT_MODEL, require_env


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        api_key=SecretStr(require_env("OPENAI_API_KEY")),
    )
