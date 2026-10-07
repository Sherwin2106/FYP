/** Turns internal evidence strings into plain sentences; drops trivial ones (e.g. word variants). */
function readable(text) {
  let t = (text || '').replace(/^ConceptNet:\s*/i, '').replace(/_/g, ' ').trim();
  if (/ means the same as | is derived from /i.test(t)) return null;
  t = t
    .replace(/^'([^']+)' is a defining concept of (.+)$/i, '“$1” is closely associated with $2')
    .replace(/^'([^']+)' matches a keyword of (.+)$/i, '“$1” is an indicator of $2')
    .replace(/^'([^']+)' and (.+?) share related concepts \((.+)\)$/i, '“$1” and $2 are linked through $3')
    .replace(/^'([^']+)' contrasts with (.+)$/i, '“$1” argues against $2')
    .replace(/'([^']+)'/g, '“$1”');
  const same = t.match(/^“(.+)” is (?:an indicator of|closely associated with) (.+)$/i);
  const stem = (w) => w.toLowerCase().replace(/s$/, '');
  if (same && stem(same[1]) === stem(same[2])) return null; // "friends" indicates friends - says nothing
  return t.charAt(0).toUpperCase() + t.slice(1) + (/[.!?]$/.test(t) ? '' : '.');
}

export default function CommonsensePanel({ evidence, commonsense }) {
  const items = [...new Set((evidence || []).map(readable).filter(Boolean))];
  const available = commonsense?.available;

  return (
    <section className="card card-body">
      <h2 className="section-title">Supporting commonsense knowledge</h2>
      {available && items.length > 0 ? (
        <>
          <ul className="evidence-list">
            {items.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
          <p className="source-note">Source: ConceptNet knowledge graph.</p>
        </>
      ) : (
        <p className="muted">No supporting commonsense knowledge was found for this image.</p>
      )}
    </section>
  );
}
