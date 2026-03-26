from __future__ import annotations

import logging
from typing import Any

from .base import LLMProvider, ProviderError
from .config import LLM_PROVIDER_CONFIGS, get_role_model, get_role_runtime

logger = logging.getLogger(__name__)


def get_role_runtime_from_config(role: str) -> str:
    """Get runtime for a role from config.py AGENT_REGISTRY."""
    try:
        from config import AGENT_REGISTRY

        if role in AGENT_REGISTRY:
            return AGENT_REGISTRY[role].runtime
    except ImportError:
        pass
    return get_role_runtime(role)


def get_role_model_from_config(role: str, runtime: str) -> str:
    """Get model for a role from config.py AGENT_REGISTRY."""
    try:
        from config import AGENT_REGISTRY

        if role in AGENT_REGISTRY:
            model = AGENT_REGISTRY[role].model
            if model:
                return model
    except ImportError:
        pass
    return get_role_model(role, runtime)


class LLMProviderRegistry:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._initialized: bool = False

    def register(self, name: str, provider: LLMProvider) -> None:
        self._providers[name] = provider
        logger.info(f"Registered LLM provider: {name}")

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise ProviderError(f"Unknown provider: {name!r}")
        return self._providers[name]

    def get_for_role(self, role: str) -> LLMProvider:
        runtime = get_role_runtime(role)
        return self.get(runtime)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def is_available(self, name: str) -> bool:
        if name not in self._providers:
            return False
        return True

    async def health_check_all(self) -> dict[str, bool]:
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                results[name] = False
        return results


_global_registry: LLMProviderRegistry | None = None


def get_provider_registry() -> LLMProviderRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = LLMProviderRegistry()
    return _global_registry


def initialize_providers() -> None:
    registry = get_provider_registry()

    if "ollama" in LLM_PROVIDER_CONFIGS:
        from .ollama_runtime import OllamaRuntime

        config = LLM_PROVIDER_CONFIGS["ollama"]
        registry.register(
            "ollama",
            OllamaRuntime(
                host=config.base_url,
                default_model=config.default_model,
                timeout=config.timeout,
            ),
        )

    if "openai" in LLM_PROVIDER_CONFIGS:
        from .openai_runtime import OpenAIRuntime

        config = LLM_PROVIDER_CONFIGS["openai"]
        registry.register(
            "openai",
            OpenAIRuntime(
                api_key=config.api_key,
                base_url=config.base_url,
                default_model=config.default_model,
                timeout=config.timeout,
            ),
        )

    if "anthropic" in LLM_PROVIDER_CONFIGS:
        from .anthropic_runtime import AnthropicRuntime

        config = LLM_PROVIDER_CONFIGS["anthropic"]
        registry.register(
            "anthropic",
            AnthropicRuntime(
                api_key=config.api_key,
                base_url=config.base_url,
                default_model=config.default_model,
                timeout=config.timeout,
            ),
        )

    if "gemini" in LLM_PROVIDER_CONFIGS:
        from .gemini_runtime import GeminiRuntime

        config = LLM_PROVIDER_CONFIGS["gemini"]
        registry.register(
            "gemini",
            GeminiRuntime(
                api_key=config.api_key,
                base_url=config.base_url,
                default_model=config.default_model,
                timeout=config.timeout,
            ),
        )

    logger.info(f"Initialized {len(registry.list_providers())} LLM providers")


def get_role_runtime_name(role: str) -> str:
    return get_role_runtime(role)


def get_role_model_for_role(role: str) -> str:
    runtime = get_role_runtime(role)
    return get_role_model(role, runtime)
