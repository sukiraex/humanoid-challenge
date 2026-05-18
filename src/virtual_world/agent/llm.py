"""LLM provider adapters for structured agent decisions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from virtual_world.agent.parser import parse_decision
from virtual_world.agent.prompts import build_messages, response_format_hint
from virtual_world.agent.schema import AgentDecision
from virtual_world.harness.action_space import ActionSpace
from virtual_world.harness.observation import Observation


class LLMClient(ABC):
    """Provider-agnostic interface for one decision per environment step."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return raw model text (expected to contain JSON)."""

    def decide(
        self,
        observation: Observation,
        action_space: ActionSpace,
        *,
        history: list[tuple[Observation, AgentDecision]] | None = None,
    ) -> AgentDecision:
        messages = build_messages(observation, action_space, history=history)
        raw = self.complete(messages)
        return parse_decision(raw, action_space)


class MockLLMClient(LLMClient):
    """Deterministic scripted client for tests and offline demos."""

    def __init__(self, script: list[str]) -> None:
        if not script:
            raise ValueError("MockLLMClient requires a non-empty action script")
        self._script = script
        self._index = 0
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        action_id = self._script[min(self._index, len(self._script) - 1)]
        if self._index < len(self._script):
            self._index += 1
        decision = AgentDecision(
            reasoning=f"Mock policy step {self._index}: executing {action_id}.",
            action_id=action_id,
        )
        return decision.model_dump_json()


class OpenAILLMClient(LLMClient):
    """OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI support requires: pip install 'virtual-world-agent[openai]'"
            ) from exc

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self._temperature = temperature

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned empty content")
        return content


class AnthropicLLMClient(LLMClient):
    """Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic support requires: pip install 'virtual-world-agent[anthropic]'"
            ) from exc

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-4-20250514")
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = ""
        user_messages: list[dict[str, Any]] = []
        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            else:
                user_messages.append(message)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=user_messages,  # type: ignore[arg-type]
        )
        parts = [block.text for block in response.content if block.type == "text"]
        if not parts:
            raise RuntimeError("Anthropic returned no text blocks")
        return parts[0]


def create_llm_client(
    provider: str,
    *,
    script: list[str] | None = None,
    model: str | None = None,
) -> LLMClient:
    """Factory for mock, openai, or anthropic clients."""
    normalized = provider.strip().lower()
    if normalized == "mock":
        if script is None:
            raise ValueError("Mock provider requires an action script")
        return MockLLMClient(script)
    if normalized == "openai":
        return OpenAILLMClient(model=model)
    if normalized in {"anthropic", "claude"}:
        return AnthropicLLMClient(model=model)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def response_schema_json() -> str:
    """Expose the decision schema for documentation and tooling."""
    from virtual_world.harness.serialization import to_json

    return to_json(response_format_hint())
