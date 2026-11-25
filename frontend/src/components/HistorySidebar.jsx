export default function HistorySidebar({ docs, activeId, query, onQuery, onSelect, onNew, onDelete }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden>न</span>
        <div><strong>NepDoc Local</strong><div className="tiny">offline document AI</div></div>
      </div>
      <button className="btn btn-primary btn-block" onClick={onNew}>+ New document</button>
      <input className="search" placeholder="Search text in all documents…" value={query} onChange={(e) => onQuery(e.target.value)} />
      <div className="history">
        {docs.length === 0 && <p className="tiny muted pad">{query ? "No matches." : "No documents yet."}</p>}
        {docs.map((d) => (
          <div key={d.id} className={`hist ${d.id === activeId ? "on" : ""}`} onClick={() => onSelect(d.id)}>
            <div className="hist-thumb">{d.thumbnail ? <img src={d.thumbnail} alt="" /> : <span className={`dot dot-${d.status === "failed" ? "bad" : "warn"}`} />}</div>
            <div className="hist-body">
              <div className="hist-name">{d.filename}</div>
              <div className="tiny muted">
                {d.status === "done" ? `${d.components} components${d.by_type?.stamp ? ` · ${d.by_type.stamp} stamp` : ""}` : d.status}
              </div>
            </div>
            <button className="icon" title="Delete" onClick={(e) => { e.stopPropagation(); onDelete(d.id); }}>×</button>
          </div>
        ))}
      </div>
    </aside>
  );
}
