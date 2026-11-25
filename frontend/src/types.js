// Visual language for layer types (used by overlay, chips, gallery).
export const TYPE_META = {
  emblem: { label: "Emblem", color: "#8b5cf6" },
  logo: { label: "Logo", color: "#a855f7" },
  figure: { label: "Figure", color: "#6366f1" },
  stamp: { label: "Stamp", color: "#e11d48" },
  signature: { label: "Signature", color: "#2563eb" },
  handwriting: { label: "Handwriting", color: "#0891b2" },
  title: { label: "Title", color: "#d97706" },
  text: { label: "Text", color: "#16a34a" },
  table: { label: "Table", color: "#0d9488" },
  caption: { label: "Caption", color: "#65a30d" },
  footnote: { label: "Footnote", color: "#78716c" },
  header_footer: { label: "Header/footer", color: "#94a3b8" },
  formula: { label: "Formula", color: "#c026d3" },
};

export const meta = (type) => TYPE_META[type] || { label: type, color: "#64748b" };

export const isDeva = (s) => !!s && /[ऀ-ॿ]/.test(s);

// Types offered when correcting or drawing a component (order = how often they're needed).
export const EDIT_TYPES = ["text", "title", "handwriting", "table", "stamp", "signature", "emblem", "logo",
  "figure", "caption", "header_footer", "footnote", "formula"];
