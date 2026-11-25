import { useState } from "react";
import { meta } from "../types.js";

const FLAG_TEXT = {
  digits_changed: "Digits changed",
  big_rewrite: "Large rewrite",
  type_change: "Type change",
  layer_change: "Moves to another layer",
  deletes_box: "Deletes a box",
};

function Diff({ parts }) {
  return (
    <div className="ai-diff ne">
      {parts.map(([op, text], i) => <span key={i} className={`d-${op}`}>{text}</span>)}
    </div>
  );
}

function Suggestion({ s, component, busy, onAccept, onReject, onSelect }) {
  const [draft, setDraft] = useState(s.value || "");
  const [editingText, setEditingText] = useState(false);
  const pending = s.status === "pending";
  const type = component?.type || s.before_type;
  return (
    <div className={`ai-sug st-${s.status}`} style={{ "--c": meta(type).color }}>
      <div className="ai-sug-head" onClick={() => component && onSelect(s.component_id)}>
        <span className="type-chip">{meta(type).label}</span>
        <span className="mono tiny muted">{s.component_id}</span>
        <span className="ai-action">
          {s.action === "set_text" && "Fix text"}
          {s.action === "set_type" && <>Change type to <b>{meta(s.value).label}</b></>}
          {s.action === "delete" && "Delete box"}
        </span>
        {!pending && <span className={`ai-status s-${s.status}`}>{s.status}</span>}
      </div>
      {s.reason && <p className="small muted ai-reason">{s.reason}</p>}
      {s.via && <p className="tiny ai-via">After your fix of {s.via.replace("recheck:", "")}</p>}
      {s.action === "set_text" && !editingText && <Diff parts={s.diff || []} />}
      {s.action === "set_text" && editingText && (
        <textarea className="ne" rows={Math.min(8, Math.max(2, draft.split("\n").length + 1))}
          value={draft} onChange={(e) => setDraft(e.target.value)} />
      )}
      {s.action === "delete" && s.before && <div className="ai-diff ne"><span className="d-del">{s.before}</span></div>}
      {s.applied_value && <p className="tiny muted">Accepted with your edit: <span className="ne">{s.applied_value}</span></p>}
      {s.kept_original && <p className="tiny muted">You kept the original text.</p>}
      {s.flags.length > 0 && (
        <div className="ai-flags">{s.flags.map((f) => <span key={f} className={`ai-flag f-${f}`}>⚠ {FLAG_TEXT[f] || f}</span>)}</div>
      )}
      {s.status === "stale" && <p className="tiny muted">Not applied: the component changed after the review ran.</p>}
      {pending && (
        <div className="ai-actions">
          <button className="btn btn-sm btn-primary" disabled={busy}
            onClick={() => onAccept(s.id, editingText && draft !== s.value ? draft : undefined)}>
            {editingText && draft !== s.value ? "Accept my edit" : "Accept"}
          </button>
          {s.action === "set_text" && (
            <button className="btn btn-sm" disabled={busy} onClick={() => setEditingText((e) => !e)}>
              {editingText ? "Show diff" : "Edit first"}
            </button>
          )}
          <button className="btn btn-sm" disabled={busy} onClick={() => onReject(s.id)}>Reject</button>
        </div>
      )}
    </div>
  );
}

export default function AiReviewPanel({ api, docId, record, busy, llm, onEdit, onSelect, autoCheck, onAutoCheck, aiCheck }) {
  const status = llm;
  const [running, setRunning] = useState(false);
  const [filter, setFilter] = useState("pending");
  const review = record.ai_review;

  async function run() {
    setRunning(true);
    try { await onEdit(() => api.runAiReview(docId)); setFilter("pending"); }
    finally { setRunning(false); }
  }

  const ready = status?.enabled && status?.reachable && status?.model_installed;
  const suggestions = review?.suggestions || [];
  const pending = suggestions.filter((s) => s.status === "pending");
  const safe = pending.filter((s) => s.flags.length === 0);
  const shown = filter === "pending" ? pending : suggestions.filter((s) => s.status !== "superseded");
  const comps = Object.fromEntries(record.components.map((c) => [c.id, c]));

  return (
    <div className="ai-panel">
      <div className="ai-intro">
        <div className="small">
          <b>{status?.model || "Local LLM"}</b>{" "}
          {status == null ? <span className="muted">checking…</span>
            : !status.enabled ? <span className="muted">disabled (NEPDOC_LLM_ENABLED=0)</span>
            : status.error ? <span className="bad">{status.error}</span>
            : !status.reachable ? <span className="bad">Ollama not running. Start it with <code>ollama serve</code></span>
            : !status.model_installed ? <span className="bad">model not installed. Run <code>ollama pull {status.model}</code></span>
            : <span className="ok">ready · {status.local ? "runs on this Mac" : "REMOTE server"}</span>}
        </div>
        <p className="tiny muted">
          Sends only the OCR text (no images) to the model. It suggests changes; nothing is applied until you accept.
          Text you corrected by hand is locked.
        </p>
        <button className="btn btn-primary btn-sm" disabled={!ready || running || busy} onClick={run}>
          {running ? "Reviewing…" : review ? "↻ Run again" : "✨ Run AI review"}
        </button>
        <label className="check small">
          <input type="checkbox" checked={autoCheck} onChange={(e) => onAutoCheck(e.target.checked)} />
          <span>Re-check automatically after I fix text or change a type
            <span className="tiny muted"> · looks for the same mistake in other boxes and checks the neighbours</span></span>
        </label>
        {(aiCheck?.state === "running" || aiCheck?.state === "waiting") &&
          <p className="tiny muted"><span className="mini-spin" /> Re-checking around {aiCheck.cid}…</p>}
        {running && <p className="tiny muted">Reading the text with {status?.model}. On a 16 GB Mac this can take a minute or two.</p>}
      </div>

      {review && (
        <>
          <div className="ai-summary">
            <span className="small">
              {suggestions.length} suggestion{suggestions.length === 1 ? "" : "s"} · {pending.length} pending
              <span className="tiny muted">
                {review.sent.components > 0 && ` · full review ${(review.duration_ms / 1000).toFixed(1)} s, ${review.sent.components} components sent`}
                {review.rechecks?.length > 0 && ` · ${review.rechecks.length} re-check${review.rechecks.length === 1 ? "" : "s"} after your edits`}
                {review.rejected ? ` · ${review.rejected} invalid ignored` : ""}</span>
            </span>
            <div className="seg">
              <button className={filter === "pending" ? "on" : ""} onClick={() => setFilter("pending")}>Pending</button>
              <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>All</button>
            </div>
          </div>
          {safe.length > 0 && (
            <button className="btn btn-sm" disabled={busy} onClick={() => onEdit(() => api.acceptSafe(docId))}
              title="Accepts suggestions without warnings. Digit changes, rewrites, type changes and deletions always need a click each.">
              ✓ Accept all safe ({safe.length})
            </button>
          )}
          {shown.length === 0 && (
            <p className="muted small">{suggestions.length ? "All suggestions handled." : "The model found nothing to fix."}</p>
          )}
          <div className="ai-list">
            {shown.map((s) => (
              <Suggestion key={`${review.id}-${s.id}`} s={s} component={comps[s.component_id]} busy={busy}
                onSelect={onSelect}
                onAccept={(sid, value) => onEdit(() => api.acceptSuggestion(docId, sid, value))}
                onReject={(sid) => onEdit(() => api.rejectSuggestion(docId, sid))} />
            ))}
          </div>
          <button className="btn btn-sm ai-discard" disabled={busy}
            onClick={() => window.confirm("Discard this review and its remaining suggestions?") && onEdit(() => api.discardAiReview(docId))}>
            Discard review
          </button>
        </>
      )}
    </div>
  );
}
