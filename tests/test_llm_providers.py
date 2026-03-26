"""Tests for the LLM providers module."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import os


class TestCostTracker(unittest.TestCase):
    """Test cost tracking functionality."""

    def test_estimate_cost_openai_gpt4(self):
        from src.llm_providers.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.estimate_cost("openai", "gpt-4o", 1000, 500)

        self.assertGreater(cost, 0)
        self.assertEqual(cost, (1000 / 1_000_000 * 5.0) + (500 / 1_000_000 * 15.0))

    def test_estimate_cost_ollama_free(self):
        from src.llm_providers.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.estimate_cost("ollama", "llama3", 1000, 500)

        self.assertEqual(cost, 0.0)

    def test_estimate_cost_unknown_provider(self):
        from src.llm_providers.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.estimate_cost("unknown", "model", 1000, 500)

        self.assertEqual(cost, 0.0)

    def test_record_updates_totals(self):
        from src.llm_providers.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.record("openai", "gpt-4o", 1000, 500)

        self.assertGreater(cost, 0)
        self.assertEqual(tracker.get_session_total(), cost)
        self.assertEqual(tracker.get_provider_breakdown()["openai"], cost)

    def test_reset_clears_data(self):
        from src.llm_providers.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", 1000, 500)
        tracker.reset()

        self.assertEqual(tracker.get_session_total(), 0)
        self.assertEqual(len(tracker.get_recent_costs()), 0)


class TestProviderConfig(unittest.TestCase):
    """Test provider configuration from environment variables."""

    def test_ollama_defaults(self):
        with patch.dict(os.environ, {"OLLAMA_ENABLED": "true"}, clear=False):
            from src.llm_providers import config

            reload(config)

            self.assertIn("ollama", config.LLM_PROVIDER_CONFIGS)
            self.assertEqual(config.LLM_PROVIDER_CONFIGS["ollama"].provider_type, "ollama")
            self.assertEqual(config.LLM_PROVIDER_CONFIGS["ollama"].default_model, "llama3:latest")

    def test_openai_requires_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            from src.llm_providers import config

            reload(config)

            self.assertIn("openai", config.LLM_PROVIDER_CONFIGS)
            self.assertEqual(config.LLM_PROVIDER_CONFIGS["openai"].api_key, "test-key")

    def test_openai_not_added_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.llm_providers import config

            reload(config)

            self.assertNotIn("openai", config.LLM_PROVIDER_CONFIGS)

    def test_get_role_runtime_default(self):
        from src.llm_providers.config import get_role_runtime

        runtime = get_role_runtime("unknown_role")
        self.assertEqual(runtime, "claude_code")

    def test_get_role_runtime_override(self):
        with patch.dict(os.environ, {"AGENT_RUNTIME_MAP": '{"pm": "ollama"}'}, clear=False):
            from src.llm_providers import config

            reload(config)

            runtime = get_role_runtime("pm")
            self.assertEqual(runtime, "ollama")


class TestToolDefinitions(unittest.TestCase):
    """Test tool definition classes."""

    def test_tool_definition_creation(self):
        from src.llm_providers.base import ToolDefinition, ToolUse, ToolResult

        tool = ToolDefinition(
            name="bash",
            description="Execute bash command",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        )

        self.assertEqual(tool.name, "bash")
        self.assertEqual(tool.description, "Execute bash command")

    def test_tool_use_creation(self):
        from src.llm_providers.base import ToolUse

        tool_use = ToolUse(name="bash", input={"command": "ls -la"}, id="call_123")

        self.assertEqual(tool_use.name, "bash")
        self.assertEqual(tool_use.input["command"], "ls -la")
        self.assertEqual(tool_use.id, "call_123")


class TestOllamaRuntime(unittest.TestCase):
    """Test Ollama runtime implementation."""

    @patch("src.llm_providers.ollama_runtime.request")
    def test_execute_success(self, mock_request):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Test output"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_request.urlopen.return_value = mock_response

        from src.llm_providers.ollama_runtime import OllamaRuntime

        runtime = OllamaRuntime(host="http://localhost:11434", default_model="llama3")

        import asyncio

        result = asyncio.run(runtime.execute("test prompt"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_text"], "Test output")
        self.assertEqual(result["model"], "llama3")
        self.assertEqual(result["provider"], "ollama")

    @patch("src.llm_providers.ollama_runtime.request")
    def test_list_models(self, mock_request):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_request.urlopen.return_value = mock_response

        from src.llm_providers.ollama_runtime import OllamaRuntime

        runtime = OllamaRuntime()

        import asyncio

        models = asyncio.run(runtime.list_models())

        self.assertEqual(models, ["llama3", "mistral"])


class TestProviderErrorClasses(unittest.TestCase):
    """Test custom exception classes."""

    def test_provider_error(self):
        from src.llm_providers.base import ProviderError

        with self.assertRaises(ProviderError):
            raise ProviderError("Test error")

    def test_model_not_found_error(self):
        from src.llm_providers.base import ModelNotFoundError

        with self.assertRaises(ModelNotFoundError):
            raise ModelNotFoundError("Model not found")

    def test_authentication_error(self):
        from src.llm_providers.base import AuthenticationError

        with self.assertRaises(AuthenticationError):
            raise AuthenticationError("Invalid API key")

    def test_rate_limit_error(self):
        from src.llm_providers.base import RateLimitError

        with self.assertRaises(RateLimitError):
            raise RateLimitError("Rate limited")


class TestRegistry(unittest.TestCase):
    """Test provider registry."""

    def test_registry_creation(self):
        from src.llm_providers.registry import LLMProviderRegistry

        registry = LLMProviderRegistry()

        self.assertEqual(registry.list_providers(), [])

    def test_registry_register(self):
        from src.llm_providers.registry import LLMProviderRegistry

        mock_provider = MagicMock()
        mock_provider.name = "Test Provider"

        registry = LLMProviderRegistry()
        registry.register("test", mock_provider)

        self.assertIn("test", registry.list_providers())
        self.assertEqual(registry.get("test"), mock_provider)

    def test_registry_get_unknown(self):
        from src.llm_providers.base import ProviderError
        from src.llm_providers.registry import LLMProviderRegistry

        registry = LLMProviderRegistry()

        with self.assertRaises(ProviderError):
            registry.get("unknown")


def reload(module):
    import importlib

    importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
