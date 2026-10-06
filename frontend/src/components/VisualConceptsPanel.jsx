const GROUPS = [
  ['people', 'People'],
  ['actions', 'Actions'],
  ['gestures', 'Gestures & expressions'],
  ['objects', 'Objects'],
];

export default function VisualConceptsPanel({ visual }) {
  if (!visual) return null;
  const groups = GROUPS.map(([key, label]) => [label, visual[key] || []]).filter(([, items]) => items.length > 0);

  return (
    <section className="card concepts-panel">
      <p className="eyebrow">What SmolVLM saw</p>
      {visual.caption && <p className="concept-caption">{visual.caption}</p>}
      {visual.setting && (
        <div className="concept-group">
          <p className="concept-group-label">Setting</p>
          <div className="concept-chips">
            <span className="concept-chip">{visual.setting}</span>
          </div>
        </div>
      )}
      {groups.map(([label, items]) => (
        <div className="concept-group" key={label}>
          <p className="concept-group-label">{label}</p>
          <div className="concept-chips">
            {items.map((item, i) => (
              <span className="concept-chip" key={i}>
                {item}
              </span>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
