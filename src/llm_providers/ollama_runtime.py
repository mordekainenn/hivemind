from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator
from urllib import error, request

from .base import (
    LLMProvider,
    ProviderError,
    ModelNotFoundError,
    ToolDefinition,
    ToolUse,
)
from .cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)


HIVEMIND_TOOLS = {
    "bash": {
        "description": "Execute a bash command in the terminal",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to execute"}},
            "required": ["command"],
        },
    },
    "read_file": {
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    "grep": {
        "description": "Search for a pattern in files",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to search"},
                "path": {"type": "string", "description": "Directory to search"},
            },
            "required": ["pattern"],
        },
    },
    "glob": {
        "description": "Find files matching a pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Directory to search"},
            },
            "required": ["pattern"],
        },
    },
    "web_fetch": {
        "description": "Fetch content from a URL",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to fetch"}},
            "required": ["url"],
        },
    },
}


class OllamaClientError(ProviderError):
    pass


class OllamaRuntime:
    name = "Ollama"
    provider_type = "ollama"
    supports_native_tools = False
    default_model = "llama3:latest"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "llama3:latest",
        timeout: int = 180,
    ):
        self.host = host.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._cost_tracker = get_cost_tracker()

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(f"{self.host}{path}", data=data, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaClientError(f"Failed to reach Ollama at {self.host}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaClientError(f"Ollama request timed out after {self.timeout}s") from exc
        except json.JSONDecodeError as exc:
            raise OllamaClientError("Ollama returned invalid JSON") from exc

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
    ) -> dict[str, Any]:
        model = model or self.default_model
        start_time = time.monotonic()

        try:
            response = self._request(
                "POST",
                "/api/generate",
                {
                    "model": model,
                    "prompt": self._build_prompt(prompt, system_prompt, context_files),
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                        "num_predict": 2048,
                    },
                },
            )

            elapsed = time.monotonic() - start_time
            result_text = response.get("response", "").strip()

            tokens_in = len(prompt) // 4
            tokens_out = len(result_text) // 4
            cost = self._cost_tracker.record("ollama", model, tokens_in, tokens_out)

            return {
                "status": "success",
                "result_text": result_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "duration_seconds": elapsed,
                "model": model,
                "provider": "ollama",
            }

        except OllamaClientError as e:
            elapsed = time.monotonic() - start_time
            return {
                "status": "error",
                "error_message": str(e),
                "duration_seconds": elapsed,
                "model": model,
                "provider": "ollama",
            }
        except Exception as e:
            elapsed = time.monotonic() - start_time
            return {
                "status": "error",
                "error_message": f"Unexpected error: {str(e)}",
                "duration_seconds": elapsed,
                "model": model,
                "provider": "ollama",
            }

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
    ):
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": self._build_prompt(prompt, system_prompt, context_files),
            "stream": True,
            "options": {
                "num_ctx": 4096,
                "num_predict": 2048,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield {
                            "kind": "text",
                            "content": token,
                            "model": model,
                            "provider": "ollama",
                        }
                    if chunk.get("done"):
                        yield {"kind": "done", "content": "", "model": model, "provider": "ollama"}
                        break
        except Exception as e:
            yield {"kind": "error", "content": str(e), "model": model, "provider": "ollama"}

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
    ) -> tuple[str, list[ToolUse]]:
        tool_defs = {t.name: t for t in tools}

        tool_prompt = self._build_tool_prompt(tool_defs)
        full_prompt = f"{tool_prompt}\n\n{prompt}"

        response = await self.execute(
            full_prompt,
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            timeout=timeout,
            budget_usd=budget_usd,
            context_files=context_files,
            working_dir=working_dir,
        )

        tool_calls = self._parse_tool_calls(response.get("result_text", ""), tool_defs)

        return response.get("result_text", ""), tool_calls

    async def list_models(self) -> list[str]:
        try:
            payload = self._request("GET", "/api/tags")
            return [item["name"] for item in payload.get("models", [])]
        except OllamaClientError:
            return []

    async def health_check(self) -> bool:
        try:
            return len(await self.list_models()) >= 0
        except Exception:
            return False

    def _build_prompt(
        self, prompt: str, system_prompt: str, context_files: list[str] | None
    ) -> str:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if context_files:
            for f in context_files:
                parts.append(f"Context file: {f}")
        parts.append(f"User: {prompt}")
        return "\n\n".join(parts)

    def _build_tool_prompt(self, tools: dict[str, ToolDefinition]) -> str:
        if not tools:
            return ""

        tool_descriptions = []
        for name, tool in tools.items():
            params = ", ".join(f"{p}" for p in tool.parameters.get("properties", {}).keys())
            tool_descriptions.append(f"- {name}({params}): {tool.description}")

        return f"""You have access to the following tools:
{chr(10).join(tool_descriptions)}

To use a tool, output a JSON object in this format:
{{"tool": "tool_name", "input": {{"param1": "value1"}}}}"""

    def _parse_tool_calls(self, text: str, tools: dict[str, ToolDefinition]) -> list[ToolUse]:
        tool_calls = []

        import re

        pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}'
        matches = re.findall(pattern, text)

        for match in matches:
            if match in tools:
                try:
                    json_match = re.search(
                        r'\{"tool"\s*:\s*"' + re.escape(match) + r'"[^{}]*\}', text
                    )
                    if json_match:
                        tool_data = json.loads(json_match.group())
                        tool_calls.append(
                            ToolUse(
                                name=tool_data.get("tool", ""),
                                input=tool_data.get("input", {}),
                                id=tool_data.get("id", ""),
                            )
                        )
                except (json.JSONDecodeError, KeyError):
                    continue

        return tool_calls
