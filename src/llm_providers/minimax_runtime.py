from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MINIMAX_AVAILABLE = True
try:
    import anthropic
except ImportError:
    MINIMAX_AVAILABLE = False


class MinimaxRuntime:
    """MiniMax provider using Anthropic-compatible API.

    API Docs: https://platform.minimax.io/docs/coding-plan/opencode
    Endpoint: https://api.minimax.io/anthropic/v1
    """

    name = "MiniMax"
    provider_type = "minimax"
    supports_native_tools = True
    default_model = "MiniMax-M2.5"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.minimax.io/anthropic/v1",
        default_model: str = "MiniMax-M2.5",
        timeout: int = 180,
    ):
        if not MINIMAX_AVAILABLE:
            from src.llm_providers.base import ProviderError

            raise ProviderError("anthropic package not installed")

        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        if not self.api_key:
            from src.llm_providers.base import ProviderError

            raise ProviderError("MiniMax API key not provided")

        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout

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
        import time

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

            from src.llm_providers import get_cost_tracker

            tokens_in = response.usage.input_tokens if response.usage else 0
            tokens_out = response.usage.output_tokens if response.usage else 0
            cost = get_cost_tracker().record("minimax", model, tokens_in, tokens_out)

            return {
                "status": "success",
                "result_text": result_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "duration_seconds": elapsed,
                "model": model,
                "provider": "minimax",
            }

        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"MiniMax error: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "duration_seconds": elapsed,
                "model": model,
                "provider": "minimax",
            }

    async def execute_streaming(self, prompt: str, **kwargs):
        from typing import AsyncIterator

        model = kwargs.get("model", self.default_model)

        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=kwargs.get("timeout", self.timeout),
        )

        messages = [{"role": "user", "content": prompt}]

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
                            "provider": "minimax",
                        }
            yield {"kind": "done", "content": "", "model": model, "provider": "minimax"}
        except Exception as e:
            yield {"kind": "error", "content": str(e), "model": model, "provider": "minimax"}

    async def execute_with_tools(
        self,
        prompt: str,
        tools: list,
        *,
        model: str = "",
        system_prompt: str = "",
        max_turns: int = 100,
        timeout: int = 900,
        budget_usd: float = 50.0,
        context_files: list[str] | None = None,
        working_dir: str = "",
        tool_executor: callable = None,
    ) -> tuple[str, list]:
        from src.llm_providers.base import ToolUse

        model = model or self.default_model

        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tools_schema = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

        response = await client.messages.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            max_tokens=2048,
        )

        result_text = "".join(block.text for block in response.content if hasattr(block, "text"))

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

                if tool_executor:
                    try:
                        result = await tool_executor(block.name, block.input)
                        messages.append({"role": "user", "content": f"[{block.name}]: {result}"})

                        response2 = await client.messages.create(
                            model=model,
                            messages=messages,
                            max_tokens=2048,
                        )
                        result_text += "\n" + "".join(
                            b.text for b in response2.content if hasattr(b, "text")
                        )
                    except Exception as e:
                        result_text += f"\n[Tool error: {e}]"

        return result_text, tool_calls

    async def list_models(self) -> list[str]:
        return [
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2.1",
            "MiniMax-M2.1-lightning",
        ]

    async def health_check(self) -> bool:
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False
