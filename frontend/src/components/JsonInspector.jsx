import { useMemo, useState } from "react";

function highlight(json) {
  const esc = json.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g, (m) => {
    const cls = m.startsWith('"') ? (m.trimEnd().endsWith(":") ? "j-key" : "j-str")
      : m === "true" || m === "false" ? "j-bool" : m === "null" ? "j-null" : "j-num";
    return `<span class="${cls}">${m}</span>`;
  });
}

// Strip UI-only URL fields so the copy matches what's stored in the database.
function forStorage(record) {
  const clone = JSON.parse(JSON.stringify(record));
  clone.pages?.forEach((p) => delete p.image_url);
  clone.components?.forEach((c) => { if (c.asset) { delete c.asset.png_url; delete c.asset.transparent_url; } });
  return clone;
}

export default function JsonInspector({ record, downloadUrl }) {
  const [scope, setScope] = useState("full");
  const [copied, setCopied] = useState(false);
  const data = useMemo(() => {
    const r = forStorage(record);
    return scope === "components" ? r.components : scope === "fields" ? r.fields : r;
  }, [record, scope]);
  const text = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return (
    <div className="stack">
      <div className="row-between">
        <div className="seg">
          {["full", "components", "fields"].map((s) => (
            <button key={s} className={scope === s ? "on" : ""} onClick={() => setScope(s)}>{s}</button>
          ))}
        </div>
        <div className="row">
          <button className="btn btn-sm" onClick={async () => { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1200); }}>
            {copied ? "Copied" : "Copy"}
          </button>
          <a className="btn btn-sm btn-primary" href={downloadUrl} download>Download .json</a>
        </div>
      </div>
      <p className="small muted">{(text.length / 1024).toFixed(1)} KB · schema v{record.schema_version} · stored in the local database (SQLite / Postgres JSONB)</p>
      <pre className="json" dangerouslySetInnerHTML={{ __html: highlight(text) }} />
    </div>
  );
}
