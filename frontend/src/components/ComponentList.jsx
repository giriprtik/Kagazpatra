import { useEffect, useRef, useState } from "react";
import { EDIT_TYPES, isDeva, meta } from "../types.js";

function Confidence({ value }) {
  if (value == null) return null;
  const pct = Math.round(value * 100);
  return <span className={`conf ${pct >= 80 ? "hi" : pct >= 55 ? "mid" : "lo"}`}>{pct}%</span>;
}

function Editor({ c, busy, onUpdate, onDelete }) {
  const [text, setText] = useState(c.text || "");
  const [status, setStatus] = useState(null);   // {kind: "saving"|"saved"|"error", msg}
  useEffect(() => setText(c.text || ""), [c.id, c.text]);
  useEffect(() => setStatus(null), [c.id]);
  useEffect(() => {
    if (status?.kind !== "saved") return undefined;
    const t = setTimeout(() => setStatus(null), 2500);
    return () => clearTimeout(t);
  }, [status]);
  const textual = ["text", "title", "handwriting", "table", "caption", "header_footer", "footnote"].includes(c.type);
  const dirty = text !== (c.text || "");

  async function save(changes, label) {
    setStatus({ kind: "saving", msg: label });
    const r = await onUpdate(c.id, changes);
    setStatus(r?.ok ? { kind: "saved", msg: "Saved" } : { kind: "error", msg: r?.error || "Couldn't save" });
  }

  return (
    <div className="editor" onClick={(e) => e.stopPropagation()}>
      <label className="ed-row">
        <span>Type</span>
        <select value={c.type} disabled={busy} onChange={(e) => save({ type: e.target.value }, "Changing type…")}>
          {EDIT_TYPES.map((t) => <option key={t} value={t}>{meta(t).label}</option>)}
        </select>
      </label>
      {(textual || c.text) && (
        <label className="ed-col">
          <span>Text <span className="tiny muted">(⌘/Ctrl + Enter to save)</span></span>
          <textarea className={isDeva(text) ? "ne" : ""} rows={Math.min(8, Math.max(2, text.split("\n").length + 1))}
            value={text} disabled={busy} onChange={(e) => { setText(e.target.value); setStatus(null); }}
            onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && dirty && !busy) save({ text }, "Saving…"); }}
            placeholder="Type the correct text…" />
        </label>
      )}
      <div className="row ed-actions">
        {(textual || c.text) && <button className="btn btn-sm btn-primary" disabled={busy || !dirty}
          title={dirty ? "Save the text" : "No changes to save"} onClick={() => save({ text }, "Saving…")}>
          {status?.kind === "saving" ? "Saving…" : "Save text"}</button>}
        {textual && <button className="btn btn-sm" disabled={busy} onClick={() => save({ reocr: true }, "Re-reading…")}
          title="Run OCR again on this box (after moving or resizing it)">↻ Re-read</button>}
        <span className="grow" />
        <button className="btn btn-sm btn-danger-ghost" disabled={busy}
          onClick={() => window.confirm(`Delete ${meta(c.type).label.toLowerCase()} ${c.id}?`) && onDelete(c.id)}>Delete</button>
      </div>
      {status && <div className={`ed-status s-${status.kind}`} role="status">
        {status.kind === "saved" ? "✓ " : status.kind === "error" ? "✗ " : ""}{status.msg}</div>}
    </div>
  );
}

export default function ComponentList({ components, selected, onSelect, editing, busy, onUpdate, onDelete }) {
  const refs = useRef({});
  useEffect(() => { selected && refs.current[selected.id]?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [selected]);

  return (
    <div className="comp-list">
      {editing && <div className="hint small">Pick a component here or on the page. Changes save immediately and are logged in History.</div>}
      {components.map((c) => {
        const m = meta(c.type);
        const open = selected?.id === c.id;
        return (
          <div key={c.id} ref={(el) => (refs.current[c.id] = el)} className={`comp ${open ? "open" : ""}`}
            style={{ "--c": m.color }} onClick={() => onSelect(c.id)}>
            <div className="comp-head">
              <span className="order mono">{c.reading_order}</span>
              <span className="type-chip">{m.label}</span>
              <span className="small muted mono">{c.id}</span>
              {c.attributes?.corrected && <span className="edited-tag" title="Corrected by a person">✎ edited</span>}
              <span className="grow" />
              {!c.attributes?.corrected && <span className="small muted" title="Detection confidence">det <Confidence value={c.confidence} /></span>}
              {c.text_confidence != null && <span className="small muted" title="OCR confidence">{c.text_source === "manual" ? "typed" : <>ocr <Confidence value={c.text_confidence} /></>}</span>}
            </div>
            {!(open && editing) && c.text && <div className={`comp-text ${isDeva(c.text) ? "ne" : ""}`}>{open ? c.text : c.text.split("\n").slice(0, 2).join(" ⏎ ")}</div>}
            {!c.text && c.asset && !(open && editing) && <img className="comp-thumb" src={c.asset.transparent_url || c.asset.png_url} alt={m.label} />}
            {open && editing && <Editor c={c} busy={busy} onUpdate={onUpdate} onDelete={onDelete} />}
            {open && !editing && (
              <div className="comp-detail small">
                <dl>
                  <dt>Source</dt><dd>{c.source}{c.attributes?.model_label ? ` (${c.attributes.model_label})` : ""}</dd>
                  <dt>Box (px)</dt><dd className="mono">{c.bbox.x0}, {c.bbox.y0} → {c.bbox.x1}, {c.bbox.y1} ({c.bbox.width}×{c.bbox.height})</dd>
                  <dt>Box (0–1)</dt><dd className="mono">{c.bbox_norm.x0.toFixed(3)}, {c.bbox_norm.y0.toFixed(3)} → {c.bbox_norm.x1.toFixed(3)}, {c.bbox_norm.y1.toFixed(3)}</dd>
                  {c.script && <><dt>Script</dt><dd>{{ ne: "Devanagari (Nepali)", en: "Latin (English)", mixed: "Mixed" }[c.script]}</dd></>}
                  {c.attributes?.ink_color && <><dt>Ink</dt><dd>{c.attributes.ink_color}</dd></>}
                  {c.attributes?.overlapped_by_stamp && <><dt>Note</dt><dd>Stamp ink removed before OCR</dd></>}
                </dl>
                {c.text && <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(c.text); }}>Copy text</button>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
