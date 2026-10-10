"""LLM client contract shared by Ollama, Foundry Local and test doubles.

Plain chat accepts sanitized strings; the optional native capability accepts
JSON-compatible sanitized messages. Neither accepts a token vault or detector
result.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-generation interface."""

    model_name: str

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's completion for an already-sanitized prompt."""
        ...


class LLMHealth(str, Enum):
    """Backend health as reported by `/health` (no internal detail)."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MODEL_MISSING = "model_missing"
    UNKNOWN = "unknown"


@runtime_checkable
class SupportsHealthCheck(Protocol):
    """Optional capability for clients that can report backend health."""

    def check_health(self) -> LLMHealth:
        ...


from dataclasses import dataclass, field

@dataclass(frozen=True)
class ToolCall:
    name: str = field(repr=False)
    arguments: dict = field(repr=False)

@dataclass(frozen=True)
class ModelTurn:
    content: str = field(repr=False)
    tool_calls: tuple[ToolCall,...] = field(default_factory=tuple,repr=False)

@runtime_checkable
class SupportsToolCalling(Protocol):
    """Optional structured-chat capability; plain chat clients remain compatible."""
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...
