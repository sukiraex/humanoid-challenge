"""Structured output schema for LLM agent decisions."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AgentDecision(BaseModel):
    """One reasoning step and chosen action from the LLM."""

    reasoning: str = Field(
        description="Brief chain-of-thought explaining the situation and plan.",
        min_length=1,
    )
    action_id: str = Field(
        description="Exactly one valid action_id from the environment catalog.",
        min_length=1,
    )

    @field_validator("action_id")
    @classmethod
    def normalize_action_id(cls, value: str) -> str:
        return value.strip().lower()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
