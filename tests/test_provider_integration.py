"""Integration tests for LLM provider connectivity and agentic task execution.

Run with: python -m pytest tests/test_provider_integration.py -v
"""

import asyncio
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProviderConnectivity:
    """Test that configured providers are accessible."""

    def test_claude_code_cli_available(self):
        """Test that Claude Code CLI is installed."""
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, "Claude CLI should be available"
        print(f"\nClaude Code version: {result.stdout.strip()}")

    def test_openai_api_key_configured(self):
        """Test that OpenAI API key is configured."""
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY should be configured"
        assert len(api_key) > 10, "OPENAI_API_KEY should be a valid key"

    def test_anthropic_api_key_configured(self):
        """Test that Anthropic API key is configured."""
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("ANTHROPIC_API_KEY")
        assert api_key, "ANTHROPIC_API_KEY should be configured"
        assert len(api_key) > 10, "ANTHROPIC_API_KEY should be a valid key"

    def test_ollama_configured(self):
        """Test that Ollama is configured."""
        from dotenv import load_dotenv

        load_dotenv()

        enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        if not enabled:
            pytest.skip("Ollama not enabled")

        # Check if Ollama is running
        try:
            result = subprocess.run(
                ["curl", "-s", f"{host}/api/tags"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                pytest.skip("Ollama not running")
            print(f"\nOllama available at: {host}")
        except Exception:
            pytest.skip("Ollama not accessible")


class TestOpenAIExecution:
    """Test OpenAI provider execution."""

    @pytest.mark.asyncio
    async def test_openai_simple_task(self):
        """Test that OpenAI can execute a simple task."""
        from dotenv import load_dotenv

        load_dotenv()

        from src.llm_providers.openai_runtime import OpenAIRuntime

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not configured")

        runtime = OpenAIRuntime(api_key=api_key)

        response = await runtime.execute(
            prompt="What is 5 + 7? Just answer with the number.",
            max_turns=1,
        )

        # Check if we got rate limited or quota exceeded
        status = response.get("status")
        if status == "rate_limited" or status == "error":
            error_msg = response.get("error_message", response.get("error", "Unknown"))
            pytest.skip(f"OpenAI quota/rate limited: {error_msg}")

        assert status == "success", f"Task failed: {response.get('error')}"
        assert "12" in response.get("result_text", ""), (
            f"Expected '12' in response, got: {response.get('result_text')}"
        )
        print(f"\nOpenAI response: {response.get('result_text')}")
        print(f"Cost: ${response.get('cost_usd', 0):.4f}")

    @pytest.mark.asyncio
    async def test_openai_list_models(self):
        """Test that OpenAI can list available models."""
        from dotenv import load_dotenv

        load_dotenv()

        from src.llm_providers.openai_runtime import OpenAIRuntime

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not configured")

        runtime = OpenAIRuntime(api_key=api_key)

        models = await runtime.list_models()

        assert models, "Should return list of models"
        print(f"\nOpenAI models: {models[:3]}...")  # Print first 3


class TestOllamaExecution:
    """Test Ollama provider execution."""

    @pytest.mark.asyncio
    async def test_ollama_simple_task(self):
        """Test that Ollama can execute a simple task."""
        from dotenv import load_dotenv

        load_dotenv()

        from src.llm_providers.ollama_runtime import OllamaRuntime

        # Use a model that's available in your Ollama instance
        runtime = OllamaRuntime(default_model="deepseek-v3.2:cloud")

        # Check if Ollama is available
        if not await runtime.health_check():
            pytest.skip("Ollama not available")

        response = await runtime.execute(
            prompt="What is 3 + 4? Just answer with the number.",
            max_turns=1,
        )

        # Check for success
        assert response.get("status") != "error", f"Ollama error: {response.get('error_message')}"
        print(f"\nOllama response: {response}")
        print(f"Status: {response.get('status')}")

    @pytest.mark.asyncio
    async def test_ollama_list_models(self):
        """Test that Ollama can list available models."""
        from dotenv import load_dotenv

        load_dotenv()

        from src.llm_providers.ollama_runtime import OllamaRuntime

        runtime = OllamaRuntime()

        if not await runtime.health_check():
            pytest.skip("Ollama not available")

        models = await runtime.list_models()

        assert models, "Should return list of models"
        print(f"\nOllama models: {models}")


class TestProviderList:
    """List all available providers and their status."""

    def test_list_available_runtimes(self):
        """List all configured runtimes and their health status."""
        from agent_runtime import AVAILABLE_RUNTIMES

        print("\n=== Available Runtimes ===")
        for name, runtime in AVAILABLE_RUNTIMES.items():
            print(f"  {name}: {type(runtime).__name__}")

    @pytest.mark.asyncio
    async def test_check_all_runtimes(self):
        """Check health of all runtimes."""
        from agent_runtime import check_all_runtimes

        results = await check_all_runtimes()

        print("\n=== Runtime Health Check ===")
        for runtime, is_healthy in results.items():
            status = "✓" if is_healthy else "✗"
            print(f"  {status} {runtime}: {'healthy' if is_healthy else 'unavailable'}")

        # At least Claude Code should be available
        assert results.get("claude_code") is True, "Claude Code should be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
