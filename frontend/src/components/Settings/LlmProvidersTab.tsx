import { useEffect, useState } from 'react';
import { useToast } from '../components/Toast';

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
  const toast = useToast();

  const loadData = () => {
    setLoading(true);
    fetch('/api/settings')
      .then(res => res.json())
      .then(setData)
      .catch(() => setError('Failed to load provider data'))
      .finally(() => setLoading(false));
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

  const providers = data?.providers || {};
  const cost = data?.cost || { session_total: 0, provider_breakdown: {} };

  return (
    <div className="space-y-6">
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
          {Object.keys(cost.provider_breakdown).length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.entries(cost.provider_breakdown).map(([provider, amount]) => (
                <span
                  key={provider}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs"
                  style={{ background: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}
                >
                  {PROVIDER_ICONS[provider] || '❓'} {provider}: ${amount.toFixed(4)}
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
          {Object.entries(providers).map(([key, provider]) => (
            <div key={key} className="px-5 py-4 flex items-center justify-between">
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
                      {provider.name}
                    </span>
                    {provider.has_api_key ? (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                        Configured
                      </span>
                    ) : (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
                        No API Key
                      </span>
                    )}
                  </div>
                  <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                    {provider.default_model || 'default model'}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {provider.base_url}
                </div>
              </div>
            </div>
          ))}
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