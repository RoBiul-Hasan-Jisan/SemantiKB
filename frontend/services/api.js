/**
 * Thin fetch wrapper around the backend REST API.
 * Centralizing all HTTP calls here means components never construct URLs
 * or handle fetch/error boilerplate directly.
 */
const BASE_URL = window.PKA_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch (_) {}
    throw new Error(`API error (${res.status}): ${detail}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  health: () => request("/api/health"),

  listDocuments: () => request("/api/documents"),

  uploadDocument: (file, existingDocumentId) => {
    const form = new FormData();
    form.append("file", file);
    const qs = existingDocumentId ? `?existing_document_id=${encodeURIComponent(existingDocumentId)}` : "";
    return request(`/api/documents/upload${qs}`, { method: "POST", body: form });
  },

  listVersions: (documentId) => request(`/api/documents/${documentId}/versions`),

  getSummary: (documentId, version) =>
    request(`/api/documents/${documentId}/summary${version ? `?version=${version}` : ""}`),

  getChunks: (documentId, strategy = "semantic", version) => {
    const params = new URLSearchParams({ strategy });
    if (version) params.set("version", version);
    return request(`/api/documents/${documentId}/chunks?${params.toString()}`);
  },

  diffVersions: (documentId, versionA, versionB) =>
    request("/api/documents/diff", {
      method: "POST",
      body: JSON.stringify({ document_id: documentId, version_a: versionA, version_b: versionB }),
    }),

  chat: (payload) => request("/api/chat", { method: "POST", body: JSON.stringify(payload) }),

  retrieveOnly: (payload) => request("/api/retrieve", { method: "POST", body: JSON.stringify(payload) }),

  getChunkingConfig: () => request("/api/config/chunking"),

  evaluate: (queries, k = 5) =>
    request(`/api/evaluate?k=${k}`, { method: "POST", body: JSON.stringify(queries) }),

  evaluationHistory: () => request("/api/evaluate/history"),
};
