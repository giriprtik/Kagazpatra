import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import AiReviewPanel from "./components/AiReviewPanel.jsx";
import ComponentList from "./components/ComponentList.jsx";
import DocumentCanvas from "./components/DocumentCanvas.jsx";
import EngineBar from "./components/EngineBar.jsx";
import FieldsPanel from "./components/FieldsPanel.jsx";
import HistoryPanel from "./components/HistoryPanel.jsx";
import HistorySidebar from "./components/HistorySidebar.jsx";
import JsonInspector from "./components/JsonInspector.jsx";
import LayerGallery from "./components/LayerGallery.jsx";
import UploadZone from "./components/UploadZone.jsx";

function AiCheckPill({ check, onOpen }) {
  if (!check) return null;
  if (check.state === "clean") return <span className="ai-pill clean">✓ AI re-checked: nothing else to fix</span>;
  if (check.state === "waiting" || check.state === "running")
    return <span className="ai-pill running" title="The local LLM is looking for the same mistake elsewhere">
      <span className="mini-spin" /> AI re-checking around {check.cid}…</span>;
  if (check.state === "found")
    return <button className="ai-pill found" onClick={onOpen}>✨ AI found {check.found} more suggestion{check.found === 1 ? "" : "s"}: review</button>;
  return <span className="ai-pill err" title={check.message}>AI re-check failed</span>;
}

const TABS = [
  { key: "components", label: "Components" },
  { key: "layers", label: "Visual layers" },
  { key: "ai", label: "AI review" },
  { key: "fields", label: "Fields" },
  { key: "json", label: "JSON" },
  { key: "history", label: "History" },
];

export default function App() {
  const [engines, setEngines] = useState(null);
  const [docs, setDocs] = useState([]);
  const [query, setQuery] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("components");
  const [selected, setSelected] = useState(null);
  const [hidden, setHidden] = useState(() => new Set());
  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [llm, setLlm] = useState(null);
  const [aiCheck, setAiCheck] = useState(null);
  const [autoCheck, setAutoCheck] = useState(() => {
    try { return localStorage.getItem("nepdoc.autoCheck") !== "0"; } catch { return true; }
  });
  const toggleAutoCheck = (on) => {
    setAutoCheck(on);
    try { localStorage.setItem("nepdoc.autoCheck", on ? "1" : "0"); } catch { /* storage unavailable */ }
  };
  const poll = useRef(null);

  const refreshList = useCallback(async () => {
    try { setDocs(await api.list(query)); } catch (e) { setError(e.message); }
  }, [query]);

  useEffect(() => { refreshList(); }, [refreshList]);
  useEffect(() => {
    const load = () => api.llmStatus().then(setLlm).catch(() => setLlm(null));
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    const load = () => api.engines().then(setEngines).catch((e) => setError(e.message));
    load();
    const t = setInterval(() => engines?.state !== "ready" && load(), 3000);
    return () => clearInterval(t);
  }, [engines?.state]);

  // Load + poll the active document until processing finishes.
  useEffect(() => {
    clearTimeout(poll.current);
    if (!activeId) { setDoc(null); return; }
    let cancelled = false;
    const tick = async () => {
      try {
        const d = await api.get(activeId);
        if (cancelled) return;
        setDoc(d);
        if (d.status === "queued" || d.status === "processing") poll.current = setTimeout(tick, 900);
        else refreshList();
      } catch (e) { if (!cancelled) setError(e.message); }
    };
    setSelected(null); setPage(1); setEditing(false); tick();
    return () => { cancelled = true; clearTimeout(poll.current); };
  }, [activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  const record = doc?.record;
  const pageInfo = record?.pages?.find((p) => p.page === page);
  const pageComponents = useMemo(
    () => (record?.components || []).filter((c) => c.page === page), [record, page]);
  const selectedComponent = record?.components.find((c) => c.id === selected) || null;

  const select = (id) => {
    setSelected(id);
    const c = record?.components.find((x) => x.id === id);
    if (c && c.page !== page) setPage(c.page);
  };

  // Every correction returns the updated document; the server re-crops, re-orders and re-extracts fields.
  // Returns {ok, error, doc} so callers (the editor) can show inline feedback.
  async function edit(fn, selectAfter) {
    setBusy(true);
    try {
      const updated = await fn();
      setDoc(updated);
      if (selectAfter !== undefined) setSelected(selectAfter(updated));
      refreshList();
      return { ok: true, doc: updated };
    } catch (e) {
      setError(e.message);
      return { ok: false, error: e.message };
    } finally { setBusy(false); }
  }

  // ---- AI re-check after manual edits: debounced, one at a time, coalesced per component ----
  const llmReady = !!(llm?.enabled && llm?.reachable && llm?.model_installed);
  const recheck = useRef({ queue: [], timer: null, running: false, found: 0 });
  const activeRef = useRef(activeId);
  useEffect(() => {
    activeRef.current = activeId;
    clearTimeout(recheck.current.timer);
    recheck.current.queue = [];
    recheck.current.found = 0;
    setAiCheck(null);
  }, [activeId]);

  useEffect(() => {
    if (aiCheck?.state !== "clean") return undefined;
    const t = setTimeout(() => setAiCheck(null), 4000);
    return () => clearTimeout(t);
  }, [aiCheck]);

  function scheduleRecheck(cid) {
    if (!autoCheck || !llmReady || !cid) return;
    const r = recheck.current;
    r.queue = [...r.queue.filter((x) => x !== cid), cid];
    if (!r.running) setAiCheck({ state: "waiting", cid });
    clearTimeout(r.timer);
    r.timer = setTimeout(drainRechecks, 1500);   // wait for a pause in editing
  }

  async function drainRechecks() {
    const r = recheck.current;
    if (r.running) return;
    r.running = true;
    const docId = activeRef.current;
    let failed = null;
    while (r.queue.length && activeRef.current === docId) {
      const cid = r.queue.shift();
      setAiCheck({ state: "running", cid });
      try {
        const res = await api.recheck(docId, cid);
        r.found += res.recheck?.suggestions || 0;
        if (activeRef.current === docId) setDoc(await api.get(docId));   // latest, including edits made meanwhile
      } catch (e) {
        if (e.status === 429) { r.queue.unshift(cid); await new Promise((ok) => setTimeout(ok, 3000)); }
        else if (e.status !== 404) failed = e.message;
      }
    }
    r.running = false;
    if (activeRef.current !== docId) return;
    if (failed) setAiCheck({ state: "error", message: failed });
    else setAiCheck(r.found ? { state: "found", found: r.found } : { state: "clean" });
  }

  async function updateComponent(cid, changes) {
    const res = await edit(() => api.patchComponent(activeId, cid, changes));
    if (res.ok && (changes.text !== undefined || changes.type || changes.reocr)) scheduleRecheck(cid);
    return res;
  }
  const deleteComponent = (cid) => edit(() => api.deleteComponent(activeId, cid), () => null);
  async function addComponent(bbox, type) {
    const res = await edit(() => api.addComponent(activeId, { page, bbox, type, ocr: true }), (d) => d.new_component_id);
    const added = res.ok && res.doc.record.components.find((c) => c.id === res.doc.new_component_id);
    if (added?.text) scheduleRecheck(added.id);
    return res;
  }
  const toggleReview = () => edit(() => api.review(activeId, !record?.review?.verified));

  async function onUploaded(id) {
    await refreshList();
    setActiveId(id);
    setTab("components");
  }

  async function remove(id) {
    if (!window.confirm("Delete this document, its crops and its JSON? This can't be undone.")) return;
    await api.remove(id);
    if (id === activeId) setActiveId(null);
    refreshList();
  }

  return (
    <div className="app">
      <HistorySidebar docs={docs} activeId={activeId} query={query} onQuery={setQuery}
        onSelect={setActiveId} onNew={() => setActiveId(null)} onDelete={remove} />

      <main className="workspace">
        <EngineBar engines={engines} />
        {error && <div className="alert alert-error" onClick={() => setError(null)}>{error} <span className="muted">(click to dismiss)</span></div>}

        {!activeId && <UploadZone ready={engines?.state === "ready"} limits={engines?.limits} onUploaded={onUploaded} onError={setError} />}

        {activeId && doc && doc.status !== "done" && (
          <div className="processing card">
            {doc.status === "failed" ? (
              <>
                <h2>Processing failed</h2>
                <p className="mono small">{doc.error}</p>
              </>
            ) : (
              <>
                <div className="spinner" />
                <h2>Processing {doc.filename}</h2>
                <p className="muted">{doc.stage || "Waiting in queue…"}</p>
                <p className="small muted">Everything runs on this Mac — nothing is uploaded anywhere.</p>
              </>
            )}
          </div>
        )}

        {record && (
          <div className="doc-grid">
            <section className="canvas-pane">
              <div className="pane-head">
                <div>
                  <h2>{doc.filename}</h2>
                  <p className="small muted">
                    {record.page_count} page{record.page_count > 1 ? "s" : ""} · {record.stats.components} components ·{" "}
                    {(record.stats.timings.total_ms / 1000).toFixed(1)} s
                    {record.corrections?.length ? ` · ${record.corrections.filter((c) => c.action !== "review" && c.action !== "ai_review").length} corrections` : ""}
                  </p>
                </div>
                <AiCheckPill check={aiCheck} onOpen={() => { setTab("ai"); setAiCheck(null); recheck.current.found = 0; }} />
                <button className={`btn btn-sm ${record.review?.verified ? "btn-ok" : ""}`} onClick={toggleReview} disabled={busy}
                  title="Mark that a person has checked this document">
                  {record.review?.verified ? "✓ Reviewed" : "Mark reviewed"}
                </button>
                {record.page_count > 1 && (
                  <div className="seg">
                    {record.pages.map((p) => (
                      <button key={p.page} className={p.page === page ? "on" : ""} onClick={() => setPage(p.page)}>Page {p.page}</button>
                    ))}
                  </div>
                )}
              </div>
              {pageInfo && (
                <DocumentCanvas page={pageInfo} components={pageComponents} hidden={hidden}
                  onToggleType={(t) => setHidden((h) => { const n = new Set(h); n.has(t) ? n.delete(t) : n.add(t); return n; })}
                  selected={selected} onSelect={select} editing={editing} busy={busy}
                  onToggleEdit={() => { setEditing((e) => !e); setTab("components"); }}
                  onChangeBox={(cid, bbox) => updateComponent(cid, { bbox })} onAdd={addComponent} />
              )}
            </section>

            <section className="inspector">
              <div className="tabs" role="tablist">
                {TABS.map((t) => (
                  <button key={t.key} role="tab" aria-selected={tab === t.key} className={tab === t.key ? "on" : ""}
                    onClick={() => setTab(t.key)}>
                    {t.label}
                    {t.key === "fields" && record.fields.length > 0 && <span className="count">{record.fields.length}</span>}
                    {t.key === "ai" && record.ai_review?.suggestions.some((s) => s.status === "pending") &&
                      <span className="count">{record.ai_review.suggestions.filter((s) => s.status === "pending").length}</span>}
                    {t.key === "history" && record.corrections?.length > 0 && <span className="count">{record.corrections.length}</span>}
                  </button>
                ))}
              </div>
              <div className="tab-body">
                {tab === "components" && <ComponentList components={record.components} selected={selectedComponent}
                  onSelect={select} editing={editing} busy={busy} onUpdate={updateComponent} onDelete={deleteComponent} />}
                {tab === "layers" && <LayerGallery components={record.components} onSelect={(id) => { select(id); setTab("components"); }} />}
                {tab === "ai" && <AiReviewPanel api={api} docId={doc.id} record={record} busy={busy} llm={llm}
                  onEdit={edit} onSelect={select} autoCheck={autoCheck} onAutoCheck={toggleAutoCheck} aiCheck={aiCheck} />}
                {tab === "fields" && <FieldsPanel fields={record.fields} onSelect={select} />}
                {tab === "json" && <JsonInspector record={record} downloadUrl={api.recordUrl(doc.id)} />}
                {tab === "history" && <HistoryPanel corrections={record.corrections} onSelect={(id) => { select(id); setTab("components"); }} />}
              </div>
              {record.warnings?.length > 0 && (
                <div className="alert alert-warn small">{record.warnings.map((w, i) => <div key={i}>{w}</div>)}</div>
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
