/** Processing time and a report download - the full result JSON, for records or the appendix. */
export default function TimingsFooter({ timings, result }) {
  function download() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const total = timings
    ? (timings.total ?? Object.entries(timings).reduce((sum, [k, v]) => (k === 'total' ? sum : sum + v), 0))
    : null;

  return (
    <div className="card meta-bar">
      <span className="num">{total != null ? `Processing time: ${total.toFixed(1)} s` : ''}</span>
      {result && (
        <button type="button" className="btn btn-secondary btn-sm" onClick={download}>
          Download report
        </button>
      )}
    </div>
  );
}
