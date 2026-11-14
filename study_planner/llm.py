"""Thin chat-completion wrappers (Anthropic / OpenAI). Offline mode uses no LLM at all."""
from __future__ import annotations

import os
from typing import Any, Optional

from .tracing import traced


class LLMError(RuntimeError):
    pass


class BaseLLM:
    provider = "base"
    model = ""

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None, client: Any = None):
        self.model = model
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install anthropic to use the Anthropic provider") from exc
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic(api_key=key)
        self.client = client

    @traced("anthropic_messages", run_type="llm")
    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        text = "".join(
            getattr(block, "text", "") for block in getattr(response, "content", []) if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise LLMError("Anthropic returned an empty response")
        return text


class OpenAILLM(BaseLLM):
    provider = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None, client: Any = None):
        self.model = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install openai to use the OpenAI provider") from exc
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMError("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=key)
        self.client = client

    @traced("openai_chat", run_type="llm")
    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        text = (response.choices[0].message.content or "") if response.choices else ""
        if not text.strip():
            raise LLMError("OpenAI returned an empty response")
        return text


def make_llm(provider: str, model: str) -> Optional[BaseLLM]:
    """Return an LLM client, or None for offline mode."""
    provider = (provider or "offline").lower()
    if provider == "offline":
        return None
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAILLM(model)
    raise LLMError(f"Unknown LLM provider {provider!r} (use anthropic, openai or offline)")
