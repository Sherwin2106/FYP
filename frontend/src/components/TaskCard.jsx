import { pct, confidenceTier } from '../lib/format';

const TIER_CLASS = { high: '', warn: 'warn', low: 'low' };

export default function TaskCard({ title, task }) {
  if (!task) return null;
  const tier = confidenceTier(task);
  const alternatives = (task.alternatives || []).slice(1, 3);

  return (
    <article className="card task-card">
      <p className="label">{title}</p>
      <h3 className="task-card-label">{task.label}</h3>

      <div className="confidence">
        <div className="confidence-head">
          <span>Confidence</span>
          <strong className="num">{pct(task.confidence)}</strong>
        </div>
        <div className="confidence-track">
          <div
            className={`confidence-fill ${TIER_CLASS[tier]}`}
            style={{ width: `${Math.max(3, Math.round(task.confidence * 100))}%` }}
          />
        </div>
        {task.ambiguous && <p className="confidence-note">Close call: the next option scored similarly.</p>}
      </div>

      {alternatives.length > 0 && (
        <div className="alternatives">
          <p className="label" style={{ marginBottom: 6 }}>
            Other possibilities
          </p>
          <ul>
            {alternatives.map((alt) => (
              <li key={alt.label_id}>
                <span>{alt.label}</span>
                <span className="num">{pct(alt.probability)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}
