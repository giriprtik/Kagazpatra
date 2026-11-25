import { useState } from "react";
import { meta } from "../types.js";

export default function LayerGallery({ components, onSelect }) {
  const visuals = components.filter((c) => c.asset);
  const [transparent, setTransparent] = useState(true);
  if (!visuals.length) return <p className="muted">No visual elements (emblems, stamps, signatures, tables) found.</p>;
  return (
    <div className="stack">
      <label className="check small">
        <input type="checkbox" checked={transparent} onChange={(e) => setTransparent(e.target.checked)} />
        <span>Show transparent versions where available</span>
      </label>
      <div className="gallery">
        {visuals.map((c) => {
          const src = (transparent && c.asset.transparent_url) || c.asset.png_url;
          const m = meta(c.type);
          return (
            <figure key={c.id} className="tile" style={{ "--c": m.color }}>
              <div className={`tile-img ${transparent && c.asset.transparent_url ? "checker" : ""}`} onClick={() => onSelect(c.id)}>
                <img src={src} alt={`${m.label} ${c.id}`} />
              </div>
              <figcaption>
                <span className="type-chip">{m.label}</span>
                <span className="small muted">{Math.round(c.confidence * 100)}%{c.attributes?.ink_color ? ` · ${c.attributes.ink_color} ink` : ""}</span>
                <a className="small" href={src} download={`${c.id}-${c.type}.png`} onClick={(e) => e.stopPropagation()}>Download</a>
              </figcaption>
            </figure>
          );
        })}
      </div>
      <p className="small muted">Files are stored locally under <span className="mono">data/assets/&lt;document&gt;/</span>.</p>
    </div>
  );
}
