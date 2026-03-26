from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderConfig:
    name: str
    provider_type: str
    enabled: bool = True
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    timeout: int = 180
    custom_config: dict[str, Any] = field(default_factory=dict)


LLM_PROVIDER_CONFIGS: dict[str, ProviderConfig] = {}


def load_provider_configs() -> dict[str, ProviderConfig]:
    configs = {}

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if os.getenv("OLLAMA_ENABLED", "true").lower() == "true":
        configs["ollama"] = ProviderConfig(
            name="Ollama",
            provider_type="ollama",
            enabled=True,
            base_url=ollama_host,
            default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3:latest"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "180")),
        )

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        configs["openai"] = ProviderConfig(
            name="OpenAI",
            provider_type="openai",
            enabled=True,
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "180")),
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        configs["anthropic"] = ProviderConfig(
            name="Anthropic",
            provider_type="anthropic",
            enabled=True,
            api_key=anthropic_key,
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            default_model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-20250514"),
            timeout=int(os.getenv("ANTHROPIC_TIMEOUT", "180")),
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        configs["gemini"] = ProviderConfig(
            name="Google Gemini",
            provider_type="gemini",
            enabled=True,
            api_key=gemini_key,
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1"),
            default_model=os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-pro"),
            timeout=int(os.getenv("GEMINI_TIMEOUT", "180")),
        )

    return configs


LLM_PROVIDER_CONFIGS = load_provider_configs()


ROLE_RUNTIME_MAP: dict[str, str] = {}
_ROLE_MAP_RAW = os.getenv("AGENT_RUNTIME_MAP", "{}")
try:
    ROLE_RUNTIME_MAP = json.loads(_ROLE_MAP_RAW)
except (json.JSONDecodeError, TypeError):
    pass

DEFAULT_RUNTIME = os.getenv("AGENT_RUNTIME_DEFAULT", "claude_code")


ROLE_MODEL_MAP: dict[str, dict[str, str]] = {
    "ollama": {},
    "openai": {},
    "anthropic": {},
    "gemini": {},
}

_OLLAMA_MODEL_RAW = os.getenv("OLLAMA_MODEL_MAP", "{}")
try:
    ROLE_MODEL_MAP["ollama"] = json.loads(_OLLAMA_MODEL_RAW)
except (json.JSONDecodeError, TypeError):
    pass

_OPENAI_MODEL_RAW = os.getenv("OPENAI_MODEL_MAP", "{}")
try:
    ROLE_MODEL_MAP["openai"] = json.loads(_OPENAI_MODEL_RAW)
except (json.JSONDecodeError, TypeError):
    pass

_ANTHROPIC_MODEL_RAW = os.getenv("ANTHROPIC_MODEL_MAP", "{}")
try:
    ROLE_MODEL_MAP["anthropic"] = json.loads(_ANTHROPIC_MODEL_RAW)
except (json.JSONDecodeError, TypeError):
    pass

_GEMINI_MODEL_RAW = os.getenv("GEMINI_MODEL_MAP", "{}")
try:
    ROLE_MODEL_MAP["gemini"] = json.loads(_GEMINI_MODEL_RAW)
except (json.JSONDecodeError, TypeError):
    pass


def get_role_runtime(role: str) -> str:
    return ROLE_RUNTIME_MAP.get(role, DEFAULT_RUNTIME)


def get_role_model(role: str, runtime: str) -> str:
    if runtime in ROLE_MODEL_MAP:
        model = ROLE_MODEL_MAP[runtime].get(role, "")
        if model:
            return model
    if runtime in LLM_PROVIDER_CONFIGS:
        return LLM_PROVIDER_CONFIGS[runtime].default_model
    return ""
