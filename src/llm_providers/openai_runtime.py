from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from .base import LLMProvider, ProviderError, ToolDefinition, ToolUse
from .cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIRuntime:
    name = "OpenAI"
    provider_type = "openai"
    supports_native_tools = True
    default_model = "gpt-4o"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o",
        timeout: int = 180,
    ):
        if not OPENAI_AVAILABLE:
            raise ProviderError("openai package not installed")

        self.api_key = api_key or openai.api_key
        if not self.api_key:
            raise ProviderError("OpenAI API key not provided")

        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self._cost_tracker = get_cost_tracker()

        openai.api_key = self.api_key
        if base_url:
            openai.base_url = base_url

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
            messages = self._build_messages(prompt, system_prompt, context_files)

            client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )

            elapsed = time.monotonic() - start_time
            result_text = response.choices[0].message.content or ""

            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            cost = self._cost_tracker.record("openai", model, tokens_in, tokens_out)

            return {
                "status": "success",
                "result_text": result_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "duration_seconds": elapsed,
                "model": model,
                "provider": "openai",
            }

        except Exception as e:
            elapsed = time.monotonic() - start_time
            error_str = str(e).lower()

            is_rate_limit = any(
                kw in error_str
                for kw in ["rate limit", "429", "too many requests", "throttl", "rate_limit"]
            )
            retry_after = 60
            if "retry_after" in error_str:
                try:
                    import re

                    match = re.search(r"retry.?after.*?(\d+)", error_str)
                    if match:
                        retry_after = int(match.group(1))
                except:
                    pass

            logger.error(f"OpenAI error: {e}")
            return {
                "status": "rate_limited" if is_rate_limit else "error",
                "error_message": str(e),
                "duration_seconds": elapsed,
                "model": model,
                "provider": "openai",
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
    ):
        model = model or self.default_model

        messages = self._build_messages(prompt, system_prompt, context_files)

        client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "kind": "text",
                        "content": chunk.choices[0].delta.content,
                        "model": model,
                        "provider": "openai",
                    }
            yield {"kind": "done", "content": "", "model": model, "provider": "openai"}

        except Exception as e:
            yield {"kind": "error", "content": str(e), "model": model, "provider": "openai"}

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
        model = model or self.default_model

        messages = self._build_messages(prompt, system_prompt, context_files)

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

        client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_defs,
                temperature=0.7,
            )

            message = response.choices[0].message
            result_text = message.content or ""

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolUse(
                            name=tc.function.name,
                            input=tc.function.arguments,
                            id=tc.id,
                        )
                    )

            return result_text, tool_calls

        except Exception as e:
            logger.error(f"OpenAI tool call error: {e}")
            return str(e), []

    async def list_models(self) -> list[str]:
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            response = await client.models.list()
            return [m.id for m in response.data if "gpt" in m.id or "o1" in m.id]
        except Exception:
            return []

    async def health_check(self) -> bool:
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False

    def _build_messages(
        self, prompt: str, system_prompt: str, context_files: list[str] | None
    ) -> list[dict[str, str]]:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context_files:
            context_text = "Context files:\n" + "\n".join(f"- {f}" for f in context_files)
            messages.append({"role": "system", "content": context_text})

        messages.append({"role": "user", "content": prompt})

        return messages
