export default function EngineBar({ engines }) {
  if (!engines) return <div className="engine-bar muted small">Connecting to the local backend…</div>;
  if (engines.state === "loading") return <div className="engine-bar small"><span className="dot dot-warn" /> Loading models…</div>;
  if (engines.state === "error") return <div className="engine-bar small"><span className="dot dot-bad" /> Models failed to load: {engines.error}</div>;
  const e = engines.engines;
  return (
    <div className="engine-bar small">
      <span className="dot dot-ok" /> <strong>Offline</strong>
      <span className="sep">·</span> Layout <span className="mono">{e.layout}</span>
      <span className="sep">·</span> OCR <span className="mono">{e.ocr || "none"}</span>
      {engines.warnings?.length > 0 && <span className="warn-chip" title={engines.warnings.join("\n")}>⚠ {engines.warnings.length} fallback{engines.warnings.length > 1 ? "s" : ""}</span>}
    </div>
  );
}
