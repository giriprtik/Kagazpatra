import { isDeva } from "../types.js";

export default function FieldsPanel({ fields, onSelect }) {
  if (!fields.length) {
    return <p className="muted">No labelled fields found (Chalani No., Darta No., Patra Sankhya, Miti, Fiscal year, Citizenship No.).</p>;
  }
  return (
    <div className="fields">
      {fields.map((f, i) => {
        const date = f.key === "date" && typeof f.normalized === "object" ? f.normalized : null;
        return (
          <button key={i} className="field" onClick={() => onSelect(f.component_id)} title="Show on the document">
            <span className="field-label">{f.label}</span>
            <span className={`field-value ${isDeva(f.value) ? "ne" : ""}`}>{f.value}</span>
            <span className="small muted">
              {date ? <>{date.bs} BS{date.ad ? ` · ${date.ad} AD` : ""}</> : typeof f.normalized === "string" && f.normalized !== f.value ? f.normalized : ""}
              {" "}· {f.component_id}
            </span>
          </button>
        );
      })}
    </div>
  );
}
