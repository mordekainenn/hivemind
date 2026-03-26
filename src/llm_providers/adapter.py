from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_runtime import RuntimeResponse, StreamEvent


class LLMRuntimeAdapter:
    name = "LLM Provider"
    provider_type = "llm"

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self._registry = None
        self._get_role_model = None

    def _ensure_registry(self):
        if self._registry is None:
            from src.llm_providers import get_provider_registry, get_role_model

            self._registry = get_provider_registry()
            self._get_role_model = get_role_model

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

        return RuntimeResponse(
            status=status,
            result_text=result.get("result_text", ""),
            cost_usd=result.get("cost_usd", 0.0),
            duration_seconds=result.get("duration_seconds", 0.0),
            tokens_in=result.get("tokens_in", 0),
            tokens_out=result.get("tokens_out", 0),
            error_message=result.get("error_message", ""),
            runtime_name=f"{provider.name}",
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
        except Exception:
            return False

    async def shutdown(self) -> None:
        pass
