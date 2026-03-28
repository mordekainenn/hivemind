from __future__ import annotations

import logging
import subprocess
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_runtime import RuntimeResponse, StreamEvent


HIVEMIND_TOOL_EXECUTORS = {
    "bash": lambda inp: _run_bash(inp.get("command", "")),
    "read_file": lambda inp: _read_file(inp.get("path", "")),
    "grep": lambda inp: _run_bash(
        f"grep -r {inp.get('pattern', '')} {inp.get('path', '.')} 2>/dev/null | head -50"
    ),
    "glob": lambda inp: _run_bash(
        f"find {inp.get('path', '.')} -maxdepth 3 -name '{inp.get('pattern', '*')}' 2>/dev/null | head -50"
    ),
}


def _run_bash(cmd: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        output = result.stdout if result.stdout else result.stderr
        return output if output else "(command completed with no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {str(e)}"


def _read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()[:10000]
    except Exception as e:
        return f"Error reading file: {e}"


class LLMRuntimeAdapter:
    name = "LLM Provider"
    provider_type = "llm"

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self._registry = None
        self._get_role_model = None

    def _ensure_registry(self):
        if self._registry is None:
            from src.llm_providers import get_provider_registry
            from src.llm_providers.registry import get_role_model_from_config

            self._registry = get_provider_registry()
            self._get_role_model = get_role_model_from_config

    async def execute(
        self,
        prompt: str,
        *,
        working_dir: str = "",
        role: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        allowed_tools: list[str] | None = None,
        system_prompt: str = "",
        context_files: list[str] | None = None,
    ) -> "RuntimeResponse":
        self._ensure_registry()

        from agent_runtime import RuntimeResponse, RuntimeStatus

        provider = self._registry.get(self.provider_name)
        model = self._get_role_model(role, self.provider_name) if role else provider.default_model

        result = await provider.execute(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            timeout=timeout,
            budget_usd=budget_usd,
            allowed_tools=allowed_tools,
            context_files=context_files,
            working_dir=working_dir,
        )

        status = RuntimeStatus.SUCCESS if result.get("status") == "success" else RuntimeStatus.ERROR

        if result.get("status") == "rate_limited":
            status = (
                RuntimeStatus.RATE_LIMITED
                if hasattr(RuntimeStatus, "RATE_LIMITED")
                else RuntimeStatus.ERROR
            )

        return RuntimeResponse(
            status=status,
            result_text=result.get("result_text", ""),
            cost_usd=result.get("cost_usd", 0.0),
            duration_seconds=result.get("duration_seconds", 0.0),
            tokens_in=result.get("tokens_in", 0),
            tokens_out=result.get("tokens_out", 0),
            error_message=result.get("error_message", ""),
            runtime_name=f"{provider.name}",
            raw=result,
        )

    async def execute_streaming(
        self,
        prompt: str,
        *,
        working_dir: str = "",
        role: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        allowed_tools: list[str] | None = None,
        system_prompt: str = "",
        context_files: list[str] | None = None,
    ) -> Any:
        self._ensure_registry()

        from agent_runtime import StreamEvent

        provider = self._registry.get(self.provider_name)
        model = self._get_role_model(role, self.provider_name) if role else provider.default_model

        try:
            async for event in provider.execute_streaming(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_turns=max_turns,
                timeout=timeout,
                budget_usd=budget_usd,
                allowed_tools=allowed_tools,
                context_files=context_files,
                working_dir=working_dir,
            ):
                if event.get("kind") == "text":
                    yield StreamEvent(
                        kind="text",
                        content=event.get("content", ""),
                        metadata={"model": event.get("model"), "provider": event.get("provider")},
                    )
                elif event.get("kind") == "error":
                    yield StreamEvent(
                        kind="error",
                        content=event.get("content", ""),
                        metadata={"model": event.get("model"), "provider": event.get("provider")},
                    )
                elif event.get("kind") == "done":
                    yield StreamEvent(
                        kind="done",
                        content="",
                        metadata={"model": event.get("model"), "provider": event.get("provider")},
                    )
        except Exception as e:
            yield StreamEvent(
                kind="error", content=str(e), metadata={"provider": self.provider_name}
            )

    async def health_check(self) -> bool:
        self._ensure_registry()
        try:
            provider = self._registry.get(self.provider_name)
            return await provider.health_check()
        except Exception as e: logger.exception(e)  # return False

    async def shutdown(self) -> None:
        pass

    async def execute_with_tools(
        self,
        prompt: str,
        allowed_tools: list[str] | None = None,
        *,
        working_dir: str = "",
        role: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        system_prompt: str = "",
        context_files: list[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Execute with tool calling support.

        For Ollama: Uses prompt-based tool loop
        For OpenAI/Anthropic: Uses native function calling
        """
        self._ensure_registry()

        from src.llm_providers import initialize_providers

        initialize_providers()

        from src.llm_providers.base import ToolDefinition

        if allowed_tools is None:
            allowed_tools = list(HIVEMIND_TOOL_EXECUTORS.keys())

        tools = [
            ToolDefinition(
                name=name,
                description=_get_tool_description(name),
                parameters=_get_tool_schema(name),
            )
            for name in allowed_tools
            if name in HIVEMIND_TOOL_EXECUTORS
        ]

        if not tools:
            result = await self.execute(
                prompt,
                working_dir=working_dir,
                role=role,
                max_turns=max_turns,
                timeout=timeout,
                budget_usd=budget_usd,
                allowed_tools=allowed_tools,
                system_prompt=system_prompt,
                context_files=context_files,
            )
            return result.result_text, []

        provider = self._registry.get(self.provider_name)
        model = self._get_role_model(role, self.provider_name) if role else provider.default_model

        tool_executor = HIVEMIND_TOOL_EXECUTORS

        async def execute_tool(tool_name: str, tool_input: dict) -> str:
            if tool_name in tool_executor:
                return tool_executor[tool_name](tool_input)
            return f"Unknown tool: {tool_name}"

        result_text, tool_calls = await provider.execute_with_tools(
            prompt=prompt,
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            timeout=timeout,
            budget_usd=budget_usd,
            context_files=context_files,
            working_dir=working_dir,
            tool_executor=execute_tool,
        )

        return result_text, [{"name": tc.name, "input": tc.input, "id": tc.id} for tc in tool_calls]


def _get_tool_description(name: str) -> str:
    desc = {
        "bash": "Execute a bash command in the terminal",
        "read_file": "Read contents of a file",
        "grep": "Search for a pattern in files",
        "glob": "Find files matching a glob pattern",
    }
    return desc.get(name, f"Tool: {name}")


def _get_tool_schema(name: str) -> dict:
    schemas = {
        "bash": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        "read_file": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "grep": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
        "glob": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    }
    return schemas.get(name, {"type": "object", "properties": {}})
