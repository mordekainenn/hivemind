from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolUse:
    name: str
    input: dict[str, Any]
    id: str = ""


@dataclass
class ToolResult:
    tool_use_id: str
    output: str
    error: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def provider_type(self) -> str: ...

    @property
    def supports_native_tools(self) -> bool: ...

    @property
    def default_model(self) -> str: ...

    async def execute(
        self,
        prompt: str,
        *,
        model: str = "",
        system_prompt: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        allowed_tools: list[str] | None = None,
        context_files: list[str] | None = None,
        working_dir: str = "",
    ) -> dict[str, Any]: ...

    async def execute_streaming(
        self,
        prompt: str,
        *,
        model: str = "",
        system_prompt: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        allowed_tools: list[str] | None = None,
        context_files: list[str] | None = None,
        working_dir: str = "",
    ) -> AsyncIterator[dict[str, Any]]: ...

    async def execute_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        *,
        model: str = "",
        system_prompt: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        context_files: list[str] | None = None,
        working_dir: str = "",
    ) -> tuple[str, list[ToolUse]]: ...

    async def list_models(self) -> list[str]: ...

    async def health_check(self) -> bool: ...


class ProviderError(RuntimeError):
    pass


class ModelNotFoundError(ProviderError):
    pass


class AuthenticationError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass
