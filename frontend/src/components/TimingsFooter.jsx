import { timingLabel, formatSeconds } from '../lib/format';
import { ClockIcon } from './Icons';

export default function TimingsFooter({ timings, modelId }) {
  if (!timings) return null;
  const entries = Object.entries(timings).filter(([k]) => k !== 'total');
  return (
    <div className="card timings-footer">
      <ClockIcon style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
      {entries.map(([key, value]) => (
        <span className="chip mono" key={key}>
          {timingLabel(key)}: {formatSeconds(value)}
        </span>
      ))}
      <span className="chip mono" style={{ background: 'var(--highlight)' }}>
        total: {formatSeconds(timings.total)}
      </span>
      {modelId && <span className="mono muted" style={{ fontSize: 11, marginLeft: 'auto' }}>{modelId}</span>}
    </div>
  );
}
