function statusText(health) {
  if (!health || health.status === 'loading') return 'Starting up…';
  if (health.status === 'ready') return 'System ready';
  return 'Service unavailable';
}

export default function Header({ health }) {
  const status = health?.status === 'ready' ? 'ready' : !health || health.status === 'loading' ? 'loading' : 'error';
  return (
    <header className="site-header">
      <div>
        <h1 className="site-title">Explainable Visual Commonsense Reasoning</h1>
        <p className="site-subtitle">Social interaction analysis — activity, relationship and intention</p>
      </div>
      <div className={`status status-${status}`} role="status">
        <span className="status-dot" />
        {statusText(health)}
      </div>
    </header>
  );
}
