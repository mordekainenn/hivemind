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

    def test_agent_registry_has_runtime_field(self):
        from config import AGENT_REGISTRY

        pm_config = AGENT_REGISTRY["pm"]

        self.assertEqual(pm_config.runtime, "claude_code")


class TestProviderConfigEnv(unittest.TestCase):
    """Test provider config from environment variables."""

    def test_ollama_host_from_env(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom:11434"}, clear=True):
            import importlib
            import src.llm_providers.config as config_module

            importlib.reload(config_module)

            self.assertIn("ollama", config_module.LLM_PROVIDER_CONFIGS)
            self.assertEqual(
                config_module.LLM_PROVIDER_CONFIGS["ollama"].base_url, "http://custom:11434"
            )

    def test_openai_model_from_env(self):
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_DEFAULT_MODEL": "gpt-4"}, clear=True
        ):
            import importlib
            import src.llm_providers.config as config_module

            importlib.reload(config_module)

            self.assertIn("openai", config_module.LLM_PROVIDER_CONFIGS)
            self.assertEqual(config_module.LLM_PROVIDER_CONFIGS["openai"].default_model, "gpt-4")


class TestRegistryConfig(unittest.TestCase):
    """Test registry config functions."""

    def test_get_role_runtime_from_config(self):
        from src.llm_providers.registry import get_role_runtime_from_config

        runtime = get_role_runtime_from_config("pm")

        self.assertIn(
            runtime,
            ["claude_code", "ollama", "openai", "anthropic", "gemini", "openclaw", "bash", "http"],
        )

    def test_get_role_model_from_config(self):
        from src.llm_providers.registry import get_role_model_from_config

        model = get_role_model_from_config("pm", "claude_code")

        self.assertIsInstance(model, str)


def reload(module):
    import importlib

    importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
