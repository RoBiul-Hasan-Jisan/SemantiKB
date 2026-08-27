import { api } from "../services/api.js";

export class DocumentInsightsPanel {
  constructor(container) {
    this.container = container;
    this.documentId = null;
    this.strategy = "semantic";
    this.render();
    this.loadChunkingConfig();
  }

  async show(documentId) {
    this.documentId = documentId;
    await this.refresh();
  }

  async refresh() {
    if (!this.documentId) return;
    try {
      const [summary, chunks] = await Promise.all([
        api.getSummary(this.documentId).catch(() => null),
        api.getChunks(this.documentId, this.strategy),
      ]);
      this.renderContent(summary, chunks);
    } catch (e) {
      console.error(e);
    }
  }

  render() {
    this.container.innerHTML = `
      <div class="insights-panel">
        <h3>Document Summary</h3>
        <div id="summary-box" class="empty">Select a document to see its summary.</div>

        <h3>Semantic Chunking Config</h3>
        <div id="chunk-config" class="config-box"></div>

        <h3>Chunks</h3>
        <div class="strategy-selector">
          <label><input type="radio" name="strategy" value="fixed" /> Fixed</label>
          <label><input type="radio" name="strategy" value="recursive" /> Recursive</label>
          <label><input type="radio" name="strategy" value="semantic" checked /> Semantic</label>
        </div>
        <div id="chunk-list" class="chunk-list"></div>
      </div>
    `;
    this.container.querySelectorAll('input[name="strategy"]').forEach((el) => {
      el.addEventListener("change", (e) => {
        this.strategy = e.target.value;
        this.refresh();
      });
    });
  }

  async loadChunkingConfig() {
    try {
      const cfg = await api.getChunkingConfig();
      this.container.querySelector("#chunk-config").innerHTML = `
        <div class="config-grid">
          <div><span>Similarity threshold</span><strong>${cfg.similarity_threshold}</strong></div>
          <div><span>Min chunk size</span><strong>${cfg.min_size_tokens} tok</strong></div>
          <div><span>Max chunk size</span><strong>${cfg.max_size_tokens} tok</strong></div>
          <div><span>Overlap</span><strong>${cfg.overlap_tokens} tok</strong></div>
          <div><span>Prefer paragraph boundaries</span><strong>${cfg.prefer_paragraph_boundaries}</strong></div>
        </div>
        <p class="hint">Tune these via environment variables (see .env.example) — lower the
        threshold for fewer/larger chunks, raise it for more/smaller chunks.</p>
      `;
    } catch (e) {
      console.error(e);
    }
  }

  renderContent(summary, chunks) {
    const summaryEl = this.container.querySelector("#summary-box");
    if (summary) {
      summaryEl.classList.remove("empty");
      summaryEl.innerHTML = `
        <p>${this.escape(summary.summary)}</p>
        <details>
          <summary>Section summaries (${Object.keys(summary.section_summaries).length})</summary>
          <ul>${Object.entries(summary.section_summaries)
            .map(([title, s]) => `<li><strong>${this.escape(title)}:</strong> ${this.escape(s)}</li>`)
            .join("")}</ul>
        </details>
      `;
    } else {
      summaryEl.textContent = "Summary not available yet (still processing, or Ollama unreachable).";
    }

    const chunkListEl = this.container.querySelector("#chunk-list");
    chunkListEl.innerHTML = `<div class="chunk-count">${chunks.length} chunks (${this.strategy})</div>` +
      chunks
        .map(
          (c) => `<div class="chunk-item">
            <div class="chunk-meta">#${c.chunk_index} · p.${c.page_start}-${c.page_end} · ${c.token_count} tok${
            c.boundary_similarity != null ? ` · boundary sim ${c.boundary_similarity.toFixed(2)}` : ""
          }</div>
            <div class="chunk-text">${this.escape(c.text.slice(0, 220))}${c.text.length > 220 ? "…" : ""}</div>
          </div>`
        )
        .join("");
  }

  escape(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
}
