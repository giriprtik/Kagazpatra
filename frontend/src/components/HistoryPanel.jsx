import { meta } from "../types.js";

const fmt = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

function Change({ entry }) {
  const { action, before, after } = entry;
  if (action === "ai_review") return <span>AI review by {entry.model}: {entry.suggestions} suggestion{entry.suggestions === 1 ? "" : "s"}</span>;
  if (action === "review") return <span>{entry.verified ? "Marked as reviewed" : "Review flag removed"}</span>;
  if (action === "add") return <span>Added {meta(after.type).label.toLowerCase()}{after.text ? `: “${after.text.slice(0, 60)}”` : ""}</span>;
  if (action === "delete") return <span>Deleted {meta(before.type).label.toLowerCase()}{before.text ? `: “${before.text.slice(0, 60)}”` : ""}</span>;
  const parts = [];
  if (before.type !== after.type) parts.push(`type ${meta(before.type).label} → ${meta(after.type).label}`);
  if (JSON.stringify(before.bbox) !== JSON.stringify(after.bbox)) parts.push("box moved/resized");
  if (before.text !== after.text) parts.push(`text “${(before.text || "").slice(0, 40)}” → “${(after.text || "").slice(0, 40)}”`);
  return <span>{parts.join(" · ") || "updated"}</span>;
}

export default function HistoryPanel({ corrections, onSelect }) {
  if (!corrections?.length) return <p className="muted">No corrections yet. Use “✎ Correct mistakes” above the page.</p>;
  return (
    <div className="history-list">
      {[...corrections].reverse().map((e, i) => (
        <div key={i} className="hist-entry" onClick={() => e.component_id && onSelect(e.component_id)}>
          <span className={`hist-action a-${e.action}`}>{e.action === "ai_review" ? "AI" : e.action}{e.via ? " · AI" : ""}</span>
          <div>
            <div className="small"><Change entry={e} /></div>
            <div className="tiny muted">{e.component_id ? `${e.component_id} · ` : ""}{fmt(e.at)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
