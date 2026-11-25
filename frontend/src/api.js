// All calls go to /api on the same origin (Vite proxy in dev, FastAPI in production).
async function request(path, options = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

const post = (path, body = {}) => request(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  engines: () => request("/api/engines"),
  list: (q = "") => request(`/api/documents${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  get: (id) => request(`/api/documents/${id}`),
  remove: (id) => request(`/api/documents/${id}`, { method: "DELETE" }),
  upload: (file, options) => {
    const form = new FormData();
    form.append("file", file);
    Object.entries(options).forEach(([k, v]) => form.append(k, String(v)));
    return request("/api/documents", { method: "POST", body: form });
  },
  recordUrl: (id) => `/api/documents/${id}/record.json`,
  patchComponent: (id, cid, changes) => request(`/api/documents/${id}/components/${cid}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes) }),
  addComponent: (id, body) => request(`/api/documents/${id}/components`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  deleteComponent: (id, cid) => request(`/api/documents/${id}/components/${cid}`, { method: "DELETE" }),
  review: (id, verified) => request(`/api/documents/${id}/review`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ verified }) }),
  llmStatus: () => request("/api/llm"),
  runAiReview: (id) => post(`/api/documents/${id}/ai-review`),
  recheck: (id, cid) => post(`/api/documents/${id}/ai-review/recheck`, { component_id: cid }),
  acceptSafe: (id) => post(`/api/documents/${id}/ai-review/accept-safe`),
  acceptSuggestion: (id, sid, value) => post(`/api/documents/${id}/ai-review/${sid}/accept`,
    value === undefined ? {} : { value }),
  rejectSuggestion: (id, sid) => post(`/api/documents/${id}/ai-review/${sid}/reject`),
  discardAiReview: (id) => request(`/api/documents/${id}/ai-review`, { method: "DELETE" }),
};
