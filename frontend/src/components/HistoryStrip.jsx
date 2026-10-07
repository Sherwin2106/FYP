export default function HistoryStrip({ items, activeId, onSelect }) {
  if (!items.length) return null;
  return (
    <div className="card card-body">
      <h2 className="section-title" style={{ fontSize: 15.5 }}>
        Recent analyses
      </h2>
      <div className="history-scroll scrollbar-thin">
        {items.map((item) => (
          <button
            type="button"
            key={item.id}
            className={`history-item${item.id === activeId ? ' active' : ''}`}
            onClick={() => onSelect(item.id)}
            title={item.summary}
          >
            <img src={item.thumbnail} alt={item.summary} />
          </button>
        ))}
      </div>
    </div>
  );
}
