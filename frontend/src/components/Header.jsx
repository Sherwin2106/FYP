import { shortModelName } from '../lib/format';

function statusText(health) {
  if (!health) return 'connecting…';
  if (health.status === 'loading') return 'loading model…';
  if (health.status === 'error') return health.error ? 'backend error' : 'backend unavailable';
  if (health.status === 'ready') {
    const cn = health.conceptnet_available ? health.conceptnet_backend : 'offline';
    const lora = health.adapter ? ' + LoRA (VCR)' : '';
    return `${shortModelName(health.model_id)}${lora} · ${health.device} · conceptnet: ${cn}`;
  }
  return 'connecting…';
}

export default function Header({ health }) {
  const status = health?.status || 'loading';
  return (
    <header className="site-header">
      <div>
        <h1 className="wordmark-main">
          Social<em>VCR</em>
        </h1>
        <span className="wordmark-sub">Explainable Visual Commonsense Reasoning — Phase I</span>
      </div>
      <div className={`status-pill status-${status}`} title={health?.error || ''}>
        <span className="status-dot" />
        {statusText(health)}
      </div>
    </header>
  );
}
