const ROWS = [
  ['setting', 'Setting'],
  ['people', 'People'],
  ['actions', 'Actions'],
  ['gestures', 'Expressions and gestures'],
  ['objects', 'Objects'],
];

function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

export default function VisualConceptsPanel({ visual }) {
  if (!visual) return null;
  const rows = ROWS.map(([key, label]) => {
    // Prefer the tidied display copies (full versions are used by the reasoning).
    const shown = visual[`display_${key}`];
    const value = Array.isArray(shown) && shown.length ? shown : visual[key];
    const items = Array.isArray(value) ? value : value ? [value] : [];
    return [label, items.map(capitalize).join(', ')];
  }).filter(([, text]) => text);
  // The model sometimes formats its description with markdown; show it as plain prose.
  const caption = (visual.display_caption || visual.caption || '').replace(/\*\*/g, '').replace(/^\s*[-*]\s+/gm, '').trim();

  return (
    <section className="card card-body">
      <h2 className="section-title">Scene description</h2>
      {caption && <p className="scene-caption">{caption}</p>}
      {rows.length > 0 && (
        <table className="concept-table">
          <tbody>
            {rows.map(([label, text]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
