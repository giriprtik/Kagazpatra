import { useMemo, useRef, useState } from "react";
import { EDIT_TYPES, meta } from "../types.js";

const MIN = 8;

/** Bounding-box overlay. In edit mode: drag to move, corner handles to resize,
 *  and the "Draw box" tool adds a missing component. */
export default function DocumentCanvas({ page, components, hidden, onToggleType, selected, onSelect,
  editing, onToggleEdit, busy, onChangeBox, onAdd }) {
  const [hover, setHover] = useState(null);
  const [zoom, setZoom] = useState(100);
  const [drawing, setDrawing] = useState(false);
  const [newType, setNewType] = useState("text");
  const [drag, setDrag] = useState(null); // {id, mode, start:{x,y}, orig, box}
  const svg = useRef(null);
  const counts = useMemo(() => components.reduce((m, c) => ({ ...m, [c.type]: (m[c.type] || 0) + 1 }), {}), [components]);
  const visible = components.filter((c) => !hidden.has(c.type));
  const focus = hover || selected;
  const fs = Math.max(14, page.width / 70);

  const toSvg = (e) => {
    const pt = svg.current.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.current.getScreenCTM().inverse());
    return { x: Math.max(0, Math.min(page.width, p.x)), y: Math.max(0, Math.min(page.height, p.y)) };
  };

  function begin(e, id, mode, orig) {
    if (!editing || busy) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const start = toSvg(e);
    setDrag({ id, mode, start, orig, box: { ...orig } });
    if (id) onSelect(id);
  }

  function move(e) {
    if (!drag) return;
    const p = toSvg(e), dx = p.x - drag.start.x, dy = p.y - drag.start.y, o = drag.orig;
    let b;
    if (drag.mode === "draw") b = { x0: Math.min(o.x0, p.x), y0: Math.min(o.y0, p.y), x1: Math.max(o.x0, p.x), y1: Math.max(o.y0, p.y) };
    else if (drag.mode === "move") {
      const w = o.x1 - o.x0, h = o.y1 - o.y0;
      const x0 = Math.max(0, Math.min(page.width - w, o.x0 + dx)), y0 = Math.max(0, Math.min(page.height - h, o.y0 + dy));
      b = { x0, y0, x1: x0 + w, y1: y0 + h };
    } else {
      b = { ...o };
      if (drag.mode.includes("w")) b.x0 = Math.min(o.x0 + dx, o.x1 - MIN);
      if (drag.mode.includes("e")) b.x1 = Math.max(o.x1 + dx, o.x0 + MIN);
      if (drag.mode.includes("n")) b.y0 = Math.min(o.y0 + dy, o.y1 - MIN);
      if (drag.mode.includes("s")) b.y1 = Math.max(o.y1 + dy, o.y0 + MIN);
    }
    setDrag({ ...drag, box: b });
  }

  function end() {
    if (!drag) return;
    const { id, mode, orig, box } = drag;
    setDrag(null);
    const round = (b) => ({ x0: Math.round(b.x0), y0: Math.round(b.y0), x1: Math.round(b.x1), y1: Math.round(b.y1) });
    if (mode === "draw") {
      if (box.x1 - box.x0 >= MIN && box.y1 - box.y0 >= MIN) { onAdd(round(box), newType); setDrawing(false); }
      return;
    }
    const moved = ["x0", "y0", "x1", "y1"].some((k) => Math.abs(box[k] - orig[k]) > 2);
    if (moved) onChangeBox(id, round(box));
  }

  function startDraw(e) {
    if (!editing || !drawing || busy) return;
    const p = toSvg(e);
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDrag({ id: null, mode: "draw", start: p, orig: { x0: p.x, y0: p.y, x1: p.x, y1: p.y }, box: { x0: p.x, y0: p.y, x1: p.x, y1: p.y } });
  }

  const boxOf = (c) => (drag && drag.id === c.id ? drag.box : c.bbox);
  const handle = fs * 0.55;

  return (
    <div className="canvas">
      <div className="layer-chips">
        {Object.entries(counts).map(([type, n]) => (
          <button key={type} className={`chip ${hidden.has(type) ? "off" : ""}`} style={{ "--c": meta(type).color }}
            onClick={() => onToggleType(type)} title={hidden.has(type) ? "Show" : "Hide"}>
            <span className="swatch" /> {meta(type).label} <span className="count">{n}</span>
          </button>
        ))}
        <span className="zoom">
          <button onClick={() => setZoom((z) => Math.max(50, z - 25))} aria-label="Zoom out">−</button>
          <span className="small mono">{zoom}%</span>
          <button onClick={() => setZoom((z) => Math.min(300, z + 25))} aria-label="Zoom in">+</button>
        </span>
      </div>

      <div className={`edit-bar ${editing ? "on" : ""}`}>
        <button className={`btn btn-sm ${editing ? "btn-primary" : ""}`} onClick={() => { setDrawing(false); onToggleEdit(); }}>
          {editing ? "✓ Done editing" : "✎ Correct mistakes"}
        </button>
        {editing && (
          <>
            <button className={`btn btn-sm ${drawing ? "btn-dark" : ""}`} onClick={() => setDrawing((d) => !d)} disabled={busy}>
              {drawing ? "Drag on the page…" : "＋ Draw missing box"}
            </button>
            <select className="select-sm" value={newType} onChange={(e) => setNewType(e.target.value)} title="Type for the new box">
              {EDIT_TYPES.map((t) => <option key={t} value={t}>{meta(t).label}</option>)}
            </select>
            <span className="small muted">{busy ? "Saving…" : "Select a box, then drag it or its corners. Edit text and type on the right."}</span>
          </>
        )}
      </div>

      <div className="stage">
        <div className="page" style={{ width: `${zoom}%` }}>
          <img src={page.image_url} alt={`Page ${page.page}`} draggable={false} />
          <svg ref={svg} viewBox={`0 0 ${page.width} ${page.height}`} preserveAspectRatio="none"
            className={`overlay ${editing && drawing ? "drawing" : ""}`}
            onPointerMove={move} onPointerUp={end} onPointerLeave={end}>
            <rect x="0" y="0" width={page.width} height={page.height} fill="transparent" onPointerDown={startDraw}
              onClick={() => !drawing && editing && onSelect(null)} />
            {visible.map((c) => {
              const b = boxOf(c), m = meta(c.type), w = b.x1 - b.x0, h = b.y1 - b.y0;
              const active = focus === c.id, isSel = selected === c.id;
              return (
                <g key={c.id} className="box" opacity={focus && !active ? 0.35 : 1}
                  onMouseEnter={() => setHover(c.id)} onMouseLeave={() => setHover(null)}
                  onClick={(e) => { e.stopPropagation(); onSelect(c.id); }}>
                  <rect x={b.x0} y={b.y0} width={w} height={h} fill={m.color} fillOpacity={active ? 0.18 : 0.06}
                    stroke={m.color} strokeWidth={active ? fs / 4 : fs / 8} rx={fs / 5}
                    strokeDasharray={c.attributes?.corrected ? `${fs / 2} ${fs / 4}` : undefined}
                    style={{ cursor: editing && isSel ? "move" : "pointer" }}
                    onPointerDown={(e) => isSel && begin(e, c.id, "move", c.bbox)} />
                  <text x={b.x0 + 4} y={Math.max(fs, b.y0 - fs * 0.3)} fontSize={fs} fill={m.color} className="box-label">
                    {c.reading_order}·{m.label}{c.attributes?.corrected ? " ✎" : ""}
                  </text>
                  {editing && isSel && [["nw", b.x0, b.y0], ["ne", b.x1, b.y0], ["sw", b.x0, b.y1], ["se", b.x1, b.y1]].map(([mode, x, y]) => (
                    <rect key={mode} x={x - handle / 2} y={y - handle / 2} width={handle} height={handle} rx={handle / 4}
                      className="handle" style={{ cursor: `${mode}-resize` }} stroke={m.color} strokeWidth={fs / 8}
                      onPointerDown={(e) => begin(e, c.id, mode, c.bbox)} />
                  ))}
                </g>
              );
            })}
            {drag?.mode === "draw" && (
              <rect x={drag.box.x0} y={drag.box.y0} width={drag.box.x1 - drag.box.x0} height={drag.box.y1 - drag.box.y0}
                fill={meta(newType).color} fillOpacity="0.15" stroke={meta(newType).color} strokeWidth={fs / 6}
                strokeDasharray={`${fs / 2} ${fs / 4}`} />
            )}
          </svg>
        </div>
      </div>
    </div>
  );
}
