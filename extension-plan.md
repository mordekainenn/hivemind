# Hivemind Multi-LLM Extension Plan

## Overview

Extend Hivemind to support multiple LLM providers beyond Claude Code, enabling users to use free and open models (Ollama, local models) as well as cloud APIs (OpenAI, Anthropic, Google Gemini).

## Architecture

### Core Concept: Provider Abstraction Layer

Leverage Hivemind's existing `AgentRuntime` abstraction (`agent_runtime.py`) and extend it with:
1. **LLM Provider Registry** - manages multiple provider backends
2. **Tool Calling Support** - unified interface for function calling across providers
3. **Cost Tracking** - per-provider cost estimation
4. **Frontend UI** - user-configurable provider and model selection per role

### Reference Implementation

Reimplement a simplified version of `local-ai-router` provider system, adapted for Hivemind's agent execution model.

---

## Decisions

| Decision | Value |
|----------|-------|
| Tool calling priority | Native function calling when available; fallback to prompt-based for Ollama |
| Per-role model config | YES - user configurable via frontend |
| Provider fallback | Fail explicitly (no automatic fallback to other providers) |
| Cost tracking | YES - implement per-provider cost estimation |
| Provider system | Simplified reimplementation of local-ai-router |

---

## Phase 1: Architecture Setup

### New Directory Structure

```
src/llm_providers/
├── __init__.py           # Exports: OllamaRuntime, OpenAIRuntime, AnthropicRuntime, GeminiRuntime, LLMProviderRegistry
├── base.py               # Extended provider interface with tool support
├── ollama_runtime.py     # Ollama REST API implementation
├── openai_runtime.py     # OpenAI API implementation
├── anthropic_runtime.py # Anthropic API (direct) implementation
├── gemini_runtime.py    # Google Gemini API implementation
├── registry.py           # LLM provider registry
├── config.py             # Provider configuration helpers
└── cost_tracker.py      # Cost estimation per provider
```

### base.py - Extended Provider Interface

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON Schema

@dataclass  
class ToolUse:
    name: str
    input: dict
    id: str

class LLMProvider(Protocol):
    """Extended provider with tool calling support."""
    
    @property
    def name(self) -> str: ...
    @property
    def supports_native_tools(self) -> bool: ...
    
    async def execute(self, prompt: str, *, model: str, system_prompt: str = "", 
                      max_turns: int = 100, timeout: int = 900, ...) -> RuntimeResponse: ...
    
    async def execute_streaming(self, prompt: str, *, ...) -> AsyncIterator[StreamEvent]: ...
    
    async def execute_with_tools(self, prompt: str, tools: list[ToolDefinition], 
                                  *, model: str, ...) -> tuple[str, list[ToolUse]]: ...
    
    async def list_models(self) -> list[str]: ...
    
    async def health_check(self) -> bool: ...
```

---

## Phase 2: Provider Implementations

### 2.1 Ollama Runtime

**Features:**
- Connect to local Ollama server (default: `http://localhost:11434`)
- Model discovery via `/api/tags`
- Streaming via `/api/generate`
- Tool calling: try native, fallback to prompt-based loop

**Config:**
```python
# config.py additions
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3:latest")
```

### 2.2 OpenAI Runtime

**Features:**
- Use `openai` Python SDK (async)
- Native function calling via `tools` parameter
- Streaming via `AsyncChatCompletions`

**Config:**
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
```

### 2.3 Anthropic API Runtime

**Features:**
- Use `anthropic` Python SDK
- Native tool use support (Claude 3+)
- Streaming via `MessageStream`

**Config:**
```python
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-20250514")
```

### 2.4 Gemini Runtime

**Features:**
- Use `google-generativeai` SDK
- Function calling support
- Streaming

**Config:**
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-pro")
```

---

## Phase 3: Integration with Hivemind

### 3.1 Update agent_runtime.py

Add new runtimes to `AVAILABLE_RUNTIMES`:

```python
AVAILABLE_RUNTIMES: dict[str, AgentRuntime] = {
    "claude_code": ClaudeCodeRuntime(),
    "openclaw": OpenClawRuntime(),
    "ollama": OllamaRuntime(),        # NEW
    "openai": OpenAIRuntime(),       # NEW  
    "anthropic": AnthropicRuntime(),  # NEW
    "gemini": GeminiRuntime(),        # NEW
    "bash": BashRuntime(),
    "http": HTTPRuntime(),
}
```

### 3.2 Update config.py

Add provider-specific config and extend `AgentConfig`:

```python
@dataclass(frozen=True)
class AgentConfig:
    timeout: int = 900
    turns: int = 100
    budget: float = 50.0
    layer: str = "execution"
    emoji: str = "\U0001f527"
    label: str = ""
    legacy: bool = False
    tw_color: str = "blue"
    accent: str = "#638cff"
    # NEW: LLM provider fields
    runtime: str = "claude_code"      # which runtime to use
    llm_model: str = ""                # which model (provider-specific)
    provider_name: str = ""            # display name for UI
```

### 3.3 Environment-based Configuration

```bash
# Default runtime for all agents
AGENT_RUNTIME_DEFAULT=ollama

# Per-role runtime + model override (JSON)
AGENT_RUNTIME_MAP='{"pm": "ollama", "frontend_developer": "openai", "reviewer": "anthropic"}'

# Per-model configuration
OLLAMA_MODEL_MAP='{"pm": "llama3", "researcher": "qwen2.5:3b", "code_reviewer": "mistral"}'
OPENAI_MODEL_MAP='{"pm": "gpt-4o", "researcher": "gpt-4-turbo"}'
```

---

## Phase 4: Tool Calling Implementation

### 4.1 Tool Definition Schema

Standardized HIVEMIND_TOOLS that all providers convert to:

```python
HIVEMIND_TOOLS = {
    "bash": {
        "description": "Execute bash command", 
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
    },
    "read_file": {
        "description": "Read file contents", 
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
    },
    "write_file": {
        "description": "Write content to file", 
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
    },
    "grep": {
        "description": "Search for pattern in files", 
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}}
    },
    "glob": {
        "description": "Find files by pattern", 
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}}
    },
    "web_fetch": {
        "description": "Fetch web content", 
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}
    },
    # ... more tools
}
```

### 4.2 Tool Execution Loop (for providers without native support)

```python
async def execute_with_tools(self, prompt, tools, **kwargs):
    # Initial generation
    response = await self.generate(prompt)
    
    # Parse tool calls from response
    tool_calls = self._parse_tool_calls(response, tools)
    
    while tool_calls:
        # Execute each tool
        tool_results = []
        for tool_call in tool_calls:
            result = await self._execute_tool(tool_call)
            tool_results.append(result)
        
        # Continue conversation with tool results
        context = "\n\n".join(f"[{tc.name}]: {tc.result}" for tc in tool_results)
        response = await self.generate(prompt + "\n\n" + context)
        tool_calls = self._parse_tool_calls(response, tools)
    
    return response
```

---

## Phase 5: Cost Tracking

### cost_tracker.py

```python
class CostTracker:
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "openai": {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        },
        "anthropic": {
            "claude-opus-4": {"input": 15.0, "output": 75.0},
            "claude-sonnet-4": {"input": 3.0, "output": 15.0},
            "claude-haiku-3": {"input": 0.25, "output": 1.25},
        },
        "gemini": {
            "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        },
        "ollama": {"free": {"input": 0, "output": 0}},  # Local models
    }
    
    def estimate_cost(self, provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
        # Return cost in USD
        ...
```

---

## Phase 6: Frontend UI

### 6.1 New Settings Page

Add a "LLM Providers" tab in settings where users can:

1. **Provider Configuration**
   - Add/remove providers
   - Configure API keys
   - Set default models

2. **Role Assignment**
   - Select which provider/model each role uses
   - Visual dropdown with available models per provider

3. **Cost Dashboard**
   - Show cost per session, per role, over time
   - Provider breakdown

### 6.2 UI Components

```
frontend/src/components/Settings/
├── LlmProvidersTab.tsx    # Provider management
├── RoleModelSelector.tsx  # Per-role model assignment
└── CostDashboard.tsx      # Cost visualization
```

### 6.3 API Endpoints

```python
# src/api/providers.py
@router.get("/api/providers")
async def list_providers(): ...

@router.get("/api/providers/{provider}/models")
async def list_models(provider: str): ...

@router.post("/api/providers/{provider}/test")
async def test_provider(provider: str): ...

@router.get("/api/costs")
async def get_costs(project_id: str): ...
```

---

## File Changes Summary

### New Files

| File | Description |
|------|-------------|
| `src/llm_providers/__init__.py` | Module exports |
| `src/llm_providers/base.py` | Extended provider protocol |
| `src/llm_providers/ollama_runtime.py` | Ollama implementation |
| `src/llm_providers/openai_runtime.py` | OpenAI implementation |
| `src/llm_providers/anthropic_runtime.py` | Anthropic direct API |
| `src/llm_providers/gemini_runtime.py` | Gemini implementation |
| `src/llm_providers/registry.py` | Provider registry |
| `src/llm_providers/config.py` | Provider config |
| `src/llm_providers/cost_tracker.py` | Cost estimation |
| `src/api/providers.py` | Provider API endpoints |
| `frontend/src/components/Settings/LlmProvidersTab.tsx` | Provider UI |
| `frontend/src/components/Settings/RoleModelSelector.tsx` | Role assignment UI |
| `frontend/src/components/Settings/CostDashboard.tsx` | Cost UI |

### Modified Files

| File | Change |
|------|--------|
| `agent_runtime.py` | Add new runtimes to AVAILABLE_RUNTIMES |
| `config.py` | Add provider config + extend AgentConfig |
| `.env.example` | Add provider environment variables |
| `requirements.txt` | Add openai, anthropic, google-generativeai |
| `frontend/src/App.tsx` | Add LLM Providers tab to settings |
| `frontend/src/types.ts` | Add provider/types |

---

## Dependencies

Add to `requirements.txt`:
```
openai>=1.0.0
anthropic>=0.25.0
google-generativeai>=0.5.0
```

---

## User Configuration Example

After implementation, users can:

```bash
# Option 1: Use local Ollama models (free)
AGENT_RUNTIME_DEFAULT=ollama
OLLAMA_DEFAULT_MODEL=llama3:latest

# Option 2: Use OpenAI (paid)
OPENAI_API_KEY=sk-...
AGENT_RUNTIME_DEFAULT=openai
OPENAI_DEFAULT_MODEL=gpt-4o

# Option 3: Mix by role
AGENT_RUNTIME_MAP='{"pm": "ollama", "backend_developer": "openai", "reviewer": "anthropic"}'
```

---

## Testing Plan

### Unit Tests
- Mock each runtime's API calls
- Test tool parsing and execution
- Test cost calculation

### Integration Tests
- Test with live Ollama server (if available)
- Test with OpenAI API (mock or test key)

### End-to-End Tests
- Run a simple DAG with different runtimes
- Verify tool execution works across providers

---

## Implementation Order

1. **Phase 1**: Create `src/llm_providers/` with base and config
2. **Phase 2**: Implement Ollama runtime first (easiest, free)
3. **Phase 2**: Implement OpenAI, Anthropic, Gemini runtimes
4. **Phase 3**: Integrate with agent_runtime.py and config.py
5. **Phase 4**: Implement tool calling for Ollama
6. **Phase 5**: Add cost tracking
7. **Phase 6**: Frontend UI for provider configuration

---

## Notes

- Keep backward compatibility with existing Claude Code runtime
- Ollama should be default for free usage
- Frontend should show clear warnings when API keys are missing
- Store provider config in project settings, not global config