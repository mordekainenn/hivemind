import { useEffect, useState } from 'react';
import { useToast } from '../../components/Toast';

interface ProviderInfo {
  name: string;
  enabled: boolean;
  has_api_key: boolean;
  default_model: string;
  base_url: string;
}

interface ProviderStatus {
  available: boolean;
  healthy: boolean;
}

interface CostInfo {
  session_total: number;
  provider_breakdown: Record<string, number>;
}

interface LlmProvidersData {
  providers: Record<string, ProviderInfo>;
  provider_status: Record<string, ProviderStatus>;
  cost: CostInfo;
  brain_layer_runtime?: string;
  brain_layer_model?: string;
  execution_layer_runtime?: string;
  execution_layer_model?: string;
}

const PROVIDER_ICONS: Record<string, string> = {
  claude_code: '🔐',
  openclaw: '🐙',
  ollama: '🦙',
  openai: '🤖',
  anthropic: '🧠',
  gemini: '🌟',
  minimax: '🎯',
  bash: '💻',
  http: '🌐',
};

const PROVIDER_COLORS: Record<string, string> = {
  claude_code: '#d97706',
  openclaw: '#6366f1',
  ollama: '#10b981',
  openai: '#10b981',
  anthropic: '#d97706',
  gemini: '#4285f4',
  minimax: '#ec4899',
};

export default function LlmProvidersTab() {
  const [data, setData] = useState<LlmProvidersData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [loadingModels, setLoadingModels] = useState<Record<string, boolean>>({});
  const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({});
  const toast = useToast();
  const [saving, setSaving] = useState(false);
  const [editingLayer, setEditingLayer] = useState<string | null>(null);
  const [layerConfig, setLayerConfig] = useState({
    brain_layer_runtime: 'claude_code',
    brain_layer_model: '',
    execution_layer_runtime: 'claude_code',
    execution_layer_model: '',
  });

  const availableRuntimes = ['claude_code', 'ollama', 'openai', 'anthropic', 'gemini', 'minimax'];

  const loadData = () => {
    setLoading(true);
    fetch('/api/settings')
      .then(res => res.json())
      .then((d) => {
        setData(d);
        setLayerConfig({
          brain_layer_runtime: d.brain_layer_runtime || 'claude_code',
          brain_layer_model: d.brain_layer_model || '',
          execution_layer_runtime: d.execution_layer_runtime || 'claude_code',
          execution_layer_model: d.execution_layer_model || '',
        });
      })
      .catch(() => setError('Failed to load provider data'))
      .finally(() => setLoading(false));
  };

  const loadModels = (provider: string) => {
    if (availableModels[provider] || loadingModels[provider]) return;
    
    setLoadingModels(prev => ({ ...prev, [provider]: true }));
    fetch(`/api/providers/${provider}/models`)
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setAvailableModels(prev => ({ ...prev, [provider]: data.models }));
        }
      })
      .catch(() => {})
      .finally(() => {
        setLoadingModels(prev => ({ ...prev, [provider]: false }));
      });
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="py-8 text-center">
        <div className="inline-block animate-spin rounded-full h-6 w-6 border-2 border-violet-500 border-t-transparent" />
        <p className="mt-2 text-sm" style={{ color: 'var(--text-muted)' }}>Loading providers...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="py-8 text-center">
        <p className="text-red-500">{error}</p>
        <button onClick={loadData} className="mt-2 text-sm text-violet-500 hover:underline">
          Retry
        </button>
      </div>
    );
  }

  const providers = (data as any)?.llm_providers || {};
  const cost = (data as any)?.llm_providers?._cost || { session_total: 0, provider_breakdown: {} };
  const brainRuntime = layerConfig.brain_layer_runtime;
  const brainModel = layerConfig.brain_layer_model;
  const execRuntime = layerConfig.execution_layer_runtime;
  const execModel = layerConfig.execution_layer_model;
  
  // Filter out the _cost key from providers display
  const providerList = Object.entries(providers).filter(([k]) => !k.startsWith('_'));

  const saveLayerConfig = async () => {
    setSaving(true);
    try {
      const res = await fetch('/api/settings/persist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(layerConfig),
      });
      if (res.ok) {
        toast.success('Configuration saved', 'Restart to apply changes.');
        setEditingLayer(null);
      } else {
        toast.error('Failed to save', 'Configuration could not be saved');
      }
    } catch {
      toast.error('Failed to save', 'Configuration could not be saved');
    }
    setSaving(false);
  };

  return (
    <div className="space-y-6">
      {/* Layer Configuration */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dim)' }}
      >
        <div className="px-5 py-3.5 flex items-center gap-2.5" style={{ borderBottom: '1px solid var(--border-dim)' }}>
          <span className="text-base">🧠</span>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Agent Layer Configuration
          </h2>
        </div>
        <div className="p-5 space-y-4">
          {/* Brain Layer */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                  🧠 Brain Layer
                </span>
                <span className="text-xs px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400">
                  PM, Architect, Orchestrator, Memory
                </span>
              </div>
              <button
                onClick={() => setEditingLayer(editingLayer === 'brain' ? null : 'brain')}
                className="text-xs px-2 py-1 rounded border"
                style={{ borderColor: 'var(--border-subtle)', color: 'var(--text-muted)' }}
              >
                {editingLayer === 'brain' ? 'Cancel' : 'Edit'}
              </button>
            </div>
            {editingLayer === 'brain' ? (
              <div className="flex gap-2">
                <select
                  className="flex-1 px-2 py-1.5 text-sm rounded-lg border"
                  style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-primary)' }}
                  value={layerConfig.brain_layer_runtime}
                  onChange={(e) => setLayerConfig({ ...layerConfig, brain_layer_runtime: e.target.value })}
                >
                  {availableRuntimes.map(r => (
                    <option key={r} value={r}>{PROVIDER_ICONS[r]} {r}</option>
                  ))}
                </select>
                <input
                  type="text"
                  placeholder="Model (optional)"
                  className="flex-1 px-2 py-1.5 text-sm rounded-lg border"
                  style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-primary)' }}
                  value={layerConfig.brain_layer_model}
                  onChange={(e) => setLayerConfig({ ...layerConfig, brain_layer_model: e.target.value })}
                />
                <button
                  onClick={saveLayerConfig}
                  disabled={saving}
                  className="px-3 py-1.5 text-sm rounded-lg bg-violet-600 text-white"
                >
                  {saving ? '...' : 'Save'}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <span style={{ color: 'var(--text-muted)' }}>Runtime:</span>
                <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                  {PROVIDER_ICONS[brainRuntime]} {brainRuntime}
                </span>
                {brainModel && (
                  <>
                    <span style={{ color: 'var(--text-muted)' }}>Model:</span>
                    <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{brainModel}</span>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Execution Layer */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                  ⚡ Execution Layer
                </span>
                <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">
                  Developers, Reviewers
                </span>
              </div>
              <button
                onClick={() => setEditingLayer(editingLayer === 'execution' ? null : 'execution')}
                className="text-xs px-2 py-1 rounded border"
                style={{ borderColor: 'var(--border-subtle)', color: 'var(--text-muted)' }}
              >
                {editingLayer === 'execution' ? 'Cancel' : 'Edit'}
              </button>
            </div>
            {editingLayer === 'execution' ? (
              <div className="flex gap-2">
                <select
                  className="flex-1 px-2 py-1.5 text-sm rounded-lg border"
                  style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-primary)' }}
                  value={layerConfig.execution_layer_runtime}
                  onChange={(e) => setLayerConfig({ ...layerConfig, execution_layer_runtime: e.target.value })}
                >
                  {availableRuntimes.map(r => (
                    <option key={r} value={r}>{PROVIDER_ICONS[r]} {r}</option>
                  ))}
                </select>
                <input
                  type="text"
                  placeholder="Model (optional)"
                  className="flex-1 px-2 py-1.5 text-sm rounded-lg border"
                  style={{ background: 'var(--bg-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-primary)' }}
                  value={layerConfig.execution_layer_model}
                  onChange={(e) => setLayerConfig({ ...layerConfig, execution_layer_model: e.target.value })}
                />
                <button
                  onClick={saveLayerConfig}
                  disabled={saving}
                  className="px-3 py-1.5 text-sm rounded-lg bg-violet-600 text-white"
                >
                  {saving ? '...' : 'Save'}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <span style={{ color: 'var(--text-muted)' }}>Runtime:</span>
                <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                  {PROVIDER_ICONS[execRuntime]} {execRuntime}
                </span>
                {execModel && (
                  <>
                    <span style={{ color: 'var(--text-muted)' }}>Model:</span>
                    <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{execModel}</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Cost Dashboard */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dim)' }}
      >
        <div className="px-5 py-3.5 flex items-center gap-2.5" style={{ borderBottom: '1px solid var(--border-dim)' }}>
          <span className="text-base">💰</span>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Session Costs
          </h2>
        </div>
        <div className="p-5">
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold" style={{ color: 'var(--text-primary)' }}>
              ${cost.session_total.toFixed(4)}
            </span>
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              this session
            </span>
          </div>
          {Object.keys(cost.provider_breakdown || {}).length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.entries(cost.provider_breakdown as Record<string, number>).map(([provider, amount]) => (
                <span
                  key={provider}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs"
                  style={{ background: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}
                >
                  {PROVIDER_ICONS[provider] || '❓'} {provider}: ${(amount || 0).toFixed(4)}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Provider List */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dim)' }}
      >
        <div className="px-5 py-3.5 flex items-center gap-2.5" style={{ borderBottom: '1px solid var(--border-dim)' }}>
          <span className="text-base">🔌</span>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            LLM Providers
          </h2>
        </div>
        <div className="divide-y" style={{ borderColor: 'var(--border-dim)' }}>
          {providerList.map(([key, provider]) => {
            const p = provider as ProviderInfo;
            return (
            <div key={key} className="px-5 py-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center text-xl"
                    style={{ 
                      background: `${PROVIDER_COLORS[key] || '#6366f1'}20`,
                      boxShadow: `0 0 12px ${PROVIDER_COLORS[key] || '#6366f1'}30`
                    }}
                  >
                    {PROVIDER_ICONS[key] || '❓'}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                        {p.name}
                      </span>
                    {p.has_api_key ? (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                          Configured
                        </span>
                      ) : p.name === 'Ollama' ? (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">
                          Local
                        </span>
                      ) : (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                          No API Key
                        </span>
                      )}
                    </div>
                    <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      {p.base_url}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Model selector */}
              <div className="flex items-center gap-2">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Model:</span>
                <button
                  onClick={() => loadModels(key)}
                  className="flex-1 text-left px-3 py-1.5 text-sm rounded-lg border"
                  style={{ 
                    background: 'var(--bg-elevated)', 
                    borderColor: 'var(--border-subtle)',
                    color: 'var(--text-primary)'
                  }}
                >
                  {p.default_model || 'Select model...'}
                  {loadingModels[key] && <span className="ml-2 text-xs text-gray-400">(loading...)</span>}
                </button>
                {availableModels[key] && availableModels[key].length > 0 && (
                  <select
                    className="px-2 py-1.5 text-xs rounded-lg border"
                    style={{ 
                      background: 'var(--bg-elevated)', 
                      borderColor: 'var(--border-subtle)',
                      color: 'var(--text-primary)'
                    }}
                    value={p.default_model}
                    onChange={(e) => {
                      console.log(`Selected ${key} model: ${e.target.value}`);
                    }}
                  >
                    {availableModels[key].map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                )}
              </div>
            </div>
          )})}
          {Object.keys(providers).length === 0 && (
            <div className="px-5 py-8 text-center">
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                No LLM providers configured. Add API keys in your .env file.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Environment Variables Help */}
      <div
        className="rounded-2xl overflow-hidden p-5"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dim)' }}
      >
        <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
          Configuration
        </h3>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
          Add the following to your .env file to enable providers:
        </p>
        <pre className="text-xs p-3 rounded overflow-x-auto" style={{ background: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}>
{`# Default runtime
AGENT_RUNTIME_DEFAULT=ollama

# Ollama (local, free)
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3:latest

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MODEL=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# MiniMax
MINIMAX_API_KEY=...

# Per-role override
AGENT_RUNTIME_MAP='{"pm": "ollama", "reviewer": "anthropic"}'`}</pre>
      </div>
    </div>
  );
}