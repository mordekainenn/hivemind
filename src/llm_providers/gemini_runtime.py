from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, AsyncIterator

from .base import LLMProvider, ProviderError, ToolDefinition, ToolUse
from .cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiRuntime:
    name = "Google Gemini"
    provider_type = "gemini"
    supports_native_tools = True
    default_model = "gemini-1.5-pro"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1",
        default_model: str = "gemini-1.5-pro",
        timeout: int = 180,
    ):
        if not GEMINI_AVAILABLE:
            raise ProviderError("google-generativeai package not installed")

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ProviderError("Gemini API key not provided")

        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self._cost_tracker = get_cost_tracker()

        genai.configure(api_key=self.api_key)

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
        model_name = model or self.default_model
        start_time = time.monotonic()

        try:
            gen_model = genai.GenerativeModel(model_name)

            contents = []
            if system_prompt:
                contents.append(system_prompt)
            if context_files:
                for f in context_files:
                    contents.append(f"Context file: {f}")
            contents.append(prompt)

            response = gen_model.generate_content(contents)

            elapsed = time.monotonic() - start_time
            result_text = response.text

            tokens_in = prompt // 4
            tokens_out = len(result_text) // 4
            cost = self._cost_tracker.record("gemini", model_name, tokens_in, tokens_out)

            return {
                "status": "success",
                "result_text": result_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "duration_seconds": elapsed,
                "model": model_name,
                "provider": "gemini",
            }

        except Exception as e:
            elapsed = time.monotonic() - start_time
            error_str = str(e).lower()

            is_rate_limit = any(
                kw in error_str
                for kw in [
                    "rate limit",
                    "429",
                    "too many requests",
                    "throttl",
                    "rate_limit",
                    "quota",
                ]
            )
            retry_after = 60
            if "retry_after" in error_str or "retry" in error_str:
                match = re.search(r"retry.?after.*?(\d+)", error_str)
                if match:
                    retry_after = int(match.group(1))

            logger.error(f"Gemini error: {e}")
            return {
                "status": "rate_limited" if is_rate_limit else "error",
                "error_message": str(e),
                "duration_seconds": elapsed,
                "model": model_name,
                "provider": "gemini",
                "retry_after": retry_after if is_rate_limit else None,
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
    ) -> AsyncIterator[dict[str, Any]]:
        model_name = model or self.default_model

        gen_model = genai.GenerativeModel(model_name)

        contents = []
        if system_prompt:
            contents.append(system_prompt)
        if context_files:
            for f in context_files:
                contents.append(f"Context file: {f}")
        contents.append(prompt)

        try:
            response = gen_model.generate_content(contents, stream=True)

            for chunk in response:
                if chunk.text:
                    yield {
                        "kind": "text",
                        "content": chunk.text,
                        "model": model_name,
                        "provider": "gemini",
                    }

            yield {"kind": "done", "content": "", "model": model_name, "provider": "gemini"}

        except Exception as e:
            yield {"kind": "error", "content": str(e), "model": model_name, "provider": "gemini"}

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
        model_name = model or self.default_model

        gen_model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.7},
        )

        contents = []
        if system_prompt:
            contents.append(system_prompt)
        if context_files:
            for f in context_files:
                contents.append(f"Context file: {f}")
        contents.append(prompt)

        tool_defs = [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                ]
            }
            for tool in tools
        ]

        try:
            response = gen_model.generate_content(
                contents,
                tools=tool_defs,
            )

            result_text = response.text or ""

            tool_calls = []
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.function_call:
                                fc = part.function_call
                                tool_calls.append(
                                    ToolUse(
                                        name=fc.name,
                                        input=dict(fc.args) if fc.args else {},
                                        id="",
                                    )
                                )

            return result_text, tool_calls

        except Exception as e:
            logger.error(f"Gemini tool call error: {e}")
            return str(e), []

    async def list_models(self) -> list[str]:
        try:
            models = genai.list_models()
            return [m.name for m in models if "gemini" in m.name]
        except Exception:
            return [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-pro",
            ]

    async def health_check(self) -> bool:
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception as e: logger.exception(e)  # return False
