import { cleanEvidence } from '../lib/format';

export default function CommonsensePanel({ evidence, commonsense }) {
  const items = evidence || [];
  const available = commonsense?.available;

  return (
    <section className="card commonsense-panel">
      <p className="eyebrow">
        Commonsense evidence
        <span className="chip commonsense-tag mono">ConceptNet · {commonsense?.backend || 'offline'}</span>
      </p>
      {available && items.length > 0 ? (
        <ul className="evidence-list">
          {items.map((e, i) => (
            <li key={i}>
              <span className="evidence-mark">▸</span>
              <span>{cleanEvidence(e)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted" style={{ fontSize: 13.5 }}>
          {available
            ? 'No strong commonsense links were found for this scene — the prediction relied mainly on SmolVLM.'
            : 'The ConceptNet backend was offline for this run, so the prediction relied on SmolVLM alone.'}
        </p>
      )}
    </section>
  );
}
