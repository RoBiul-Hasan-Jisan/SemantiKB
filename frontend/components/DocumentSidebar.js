import { api } from "../services/api.js";

export class DocumentSidebar {
  constructor(container, { onSelectDocument }) {
    this.container = container;
    this.onSelectDocument = onSelectDocument;
    this.documents = [];
    this.selectedId = null;
    this.render();
    this.refresh();
  }

  async refresh() {
    try {
      this.documents = await api.listDocuments();
    } catch (e) {
      console.error(e);
    }
    this.renderList();
  }

  render() {
    this.container.innerHTML = `
      <div class="sidebar">
        <h2>Documents</h2>
        <label class="upload-btn">
          + Upload PDF / TXT
          <input type="file" accept=".pdf,.txt" id="file-input" hidden />
        </label>
        <div id="upload-status" class="upload-status"></div>
        <div id="doc-list" class="doc-list"></div>
      </div>
    `;
    this.container.querySelector("#file-input").addEventListener("change", (e) => this.handleUpload(e));
  }

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const statusEl = this.container.querySelector("#upload-status");
    statusEl.textContent = `Ingesting ${file.name}... (parsing, chunking, embedding, summarizing)`;
    try {
      const result = await api.uploadDocument(file);
      statusEl.textContent = `✓ ${result.filename} indexed (${result.status})`;
      await this.refresh();
    } catch (err) {
      statusEl.textContent = `✗ ${err.message}`;
    }
    e.target.value = "";
  }

  renderList() {
    const listEl = this.container.querySelector("#doc-list");
    if (!this.documents.length) {
      listEl.innerHTML = `<p class="empty">No documents yet. Upload a PDF or TXT to get started.</p>`;
      return;
    }
    listEl.innerHTML = this.documents
      .map(
        (d) => `
        <div class="doc-item ${d.document_id === this.selectedId ? "selected" : ""}" data-id="${d.document_id}">
          <div class="doc-name">${d.filename}</div>
          <div class="doc-meta">v${d.latest_version} · ${d.status}</div>
        </div>`
      )
      .join("");
    listEl.querySelectorAll(".doc-item").forEach((el) => {
      el.addEventListener("click", () => {
        this.selectedId = el.dataset.id;
        this.renderList();
        this.onSelectDocument(this.selectedId);
      });
    });
  }
}
