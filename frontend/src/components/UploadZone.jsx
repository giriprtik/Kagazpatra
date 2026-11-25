import { useRef, useState } from "react";
import { api } from "../api.js";

const ACCEPT = "image/jpeg,image/png,image/webp,image/tiff,image/heic,image/heif,.heic,.heif,application/pdf";
const OPTIONS = [
  { key: "transparent", label: "Transparent PNGs for emblems, stamps & signatures", def: true },
  { key: "suppress_stamp_ink", label: "Remove stamp ink before reading text underneath", def: true },
  { key: "ocr_tables", label: "Read tables", def: true },
  { key: "ocr_stamps", label: "Also try to read text inside stamps", def: false },
];

export default function UploadZone({ ready, limits, onUploaded, onError }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [opts, setOpts] = useState(Object.fromEntries(OPTIONS.map((o) => [o.key, o.def])));

  async function send(file) {
    if (!file) return;
    const max = (limits?.max_upload_mb || 25) * 1024 * 1024;
    if (file.size > max) { onError(`File is larger than ${limits?.max_upload_mb || 25} MB`); return; }
    setBusy(true);
    try {
      const res = await api.upload(file, opts);
      onUploaded(res.id);
    } catch (e) { onError(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="upload-wrap">
      <div className={`dropzone ${drag ? "drag" : ""} ${!ready ? "disabled" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); ready && send(e.dataTransfer.files[0]); }}
        onClick={() => ready && input.current?.click()} role="button" tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && ready && input.current?.click()}>
        <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => send(e.target.files[0])} />
        <div className="drop-icon" aria-hidden>
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M12 17v-6M9 14l3-3 3 3" />
          </svg>
        </div>
        <h2>{busy ? "Uploading…" : ready ? "Drop a Nepali document here" : "Waiting for models…"}</h2>
        <p className="muted">Letters, lalpurja, citizenship, certificates · JPEG, PNG, HEIC, WEBP, TIFF or PDF
          (first {limits?.max_pdf_pages || 5} pages) · up to {limits?.max_upload_mb || 25} MB</p>
        <p className="small privacy">🔒 Processed entirely on this machine. Nothing leaves your computer.</p>
      </div>
      <div className="options card">
        {OPTIONS.map((o) => (
          <label key={o.key} className="check">
            <input type="checkbox" checked={opts[o.key]} onChange={(e) => setOpts({ ...opts, [o.key]: e.target.checked })} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
