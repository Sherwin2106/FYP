import { pct, confidenceTier } from '../lib/format';

const TIER_CLASS = { high: '', warn: 'warn', low: 'low' };

export default function TaskCard({ title, task }) {
  if (!task) return null;
  const tier = confidenceTier(task);
  const alternatives = (task.alternatives || []).slice(1, 3);

  return (
    <article className="card task-card">
      <p className="eyebrow">{title}</p>
      <h3 className="task-card-label">{task.label}</h3>

      <div className="confidence-track">
        <div
          className={`confidence-fill ${TIER_CLASS[tier]}`}
          style={{ width: `${Math.max(3, Math.round(task.confidence * 100))}%` }}
        />
      </div>
      <p className="confidence-readout">
        <span className="mono">{pct(task.confidence)}</span> confidence
        {task.ambiguous && <span className="ambiguous-flag">· ambiguous</span>}
      </p>

      {alternatives.length > 0 && (
        <ul className="task-card-alts">
          {alternatives.map((alt) => (
            <li key={alt.label_id}>
              <span>{alt.label}</span>
              <span className="mono">{pct(alt.probability)}</span>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
