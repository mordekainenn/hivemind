from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from .base import LLMProvider, ProviderError, ToolDefinition, ToolUse
from .cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicRuntime:
    name = "Anthropic"
    provider_type = "anthropic"
    supports_native_tools = True
    default_model = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        default_model: str = "claude-sonnet-4-20250514",
        timeout: int = 180,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ProviderError("anthropic package not installed")

        self.api_key = api_key or anthropic.api_key
        if not self.api_key:
            raise ProviderError("Anthropic API key not provided")

        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self._cost_tracker = get_cost_tracker()

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
            client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
            )

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if context_files:
                context_text = "Context files:\n" + "\n".join(f"- {f}" for f in context_files)
                messages.append({"role": "user", "content": context_text})

            messages.append({"role": "user", "content": prompt})

            response = await client.messages.create(
                model=model,
                messages=messages,
                max_tokens=2048,
            )

            elapsed = time.monotonic() - start_time
            result_text = "".join(block.text for block in response.content)

            tokens_in = response.usage.input_tokens if response.usage else 0
            tokens_out = response.usage.output_tokens if response.usage else 0
            cost = self._cost_tracker.record("anthropic", model, tokens_in, tokens_out)

            return {
                "status": "success",
                "result_text": result_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "duration_seconds": elapsed,
                "model": model,
                "provider": "anthropic",
            }

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"Anthropic error: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "duration_seconds": elapsed,
                "model": model,
                "provider": "anthropic",
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
        model = model or self.default_model

        client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context_files:
            context_text = "Context files:\n" + "\n".join(f"- {f}" for f in context_files)
            messages.append({"role": "user", "content": context_text})

        messages.append({"role": "user", "content": prompt})

        try:
            async with client.messages.stream(
                model=model,
                messages=messages,
                max_tokens=2048,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield {
                            "kind": "text",
                            "content": text,
                            "model": model,
                            "provider": "anthropic",
                        }

            yield {"kind": "done", "content": "", "model": model, "provider": "anthropic"}

        except Exception as e:
            yield {"kind": "error", "content": str(e), "model": model, "provider": "anthropic"}

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

        client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context_files:
            context_text = "Context files:\n" + "\n".join(f"- {f}" for f in context_files)
            messages.append({"role": "user", "content": context_text})

        messages.append({"role": "user", "content": prompt})

        tools_schema = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

        try:
            response = await client.messages.create(
                model=model,
                messages=messages,
                tools=tools_schema,
                max_tokens=2048,
            )

            result_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

            tool_calls = []
            for block in response.content:
                if hasattr(block, "name") and block.name:
                    tool_calls.append(
                        ToolUse(
                            name=block.name,
                            input=block.input,
                            id=block.id or "",
                        )
                    )

            return result_text, tool_calls

        except Exception as e:
            logger.error(f"Anthropic tool call error: {e}")
            return str(e), []

    async def list_models(self) -> list[str]:
        return [
            "claude-opus-4-20250514",
            "claude-opus-3-5-20241022",
            "claude-sonnet-4-20250514",
            "claude-sonnet-3-5-20241022",
            "claude-haiku-3-20240307",
        ]

    async def health_check(self) -> bool:
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False
