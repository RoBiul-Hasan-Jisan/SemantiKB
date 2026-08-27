import { api } from "../services/api.js";

export class VersionHistory {
  constructor(container) {
    this.container = container;
    this.documentId = null;
    this.versions = [];
    this.render();
  }

  async show(documentId) {
    this.documentId = documentId;
    this.versions = await api.listVersions(documentId);
    this.render();
  }

  render() {
    if (!this.documentId) {
      this.container.innerHTML = `<p class="empty">Select a document to see its version history.</p>`;
      return;
    }
    this.container.innerHTML = `
      <h3>Version History</h3>
      <table class="version-table">
        <thead><tr><th>Version</th><th>Valid From</th><th>Valid To</th><th>Pages</th></tr></thead>
        <tbody>
          ${this.versions
            .map(
              (v) => `<tr>
                <td>v${v.version}${v.is_latest ? " (latest)" : ""}</td>
                <td>${new Date(v.valid_from).toLocaleDateString()}</td>
                <td>${v.valid_to ? new Date(v.valid_to).toLocaleDateString() : "—"}</td>
                <td>${v.page_count}</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>
      ${this.versions.length >= 2 ? this.renderDiffForm() : ""}
      <div id="diff-result"></div>
    `;
    if (this.versions.length >= 2) {
      this.container.querySelector("#diff-form").addEventListener("submit", (e) => this.handleDiff(e));
    }
  }

  renderDiffForm() {
    const options = this.versions.map((v) => `<option value="${v.version}">v${v.version}</option>`).join("");
    return `
      <form id="diff-form" class="diff-form">
        <label>Compare <select id="version-a">${options}</select></label>
        <label>with <select id="version-b">${options}</select></label>
        <button type="submit">Show changes</button>
      </form>
    `;
  }

  async handleDiff(e) {
    e.preventDefault();
    const a = Number(this.container.querySelector("#version-a").value);
    const b = Number(this.container.querySelector("#version-b").value);
    const result = await api.diffVersions(this.documentId, a, b);
    const resEl = this.container.querySelector("#diff-result");
    resEl.innerHTML = `
      <div class="diff-result">
        <div>Similarity: ${(result.similarity_ratio * 100).toFixed(1)}%</div>
        <div class="diff-col added"><strong>Added (${result.added_lines.length})</strong>
          <ul>${result.added_lines.slice(0, 20).map((l) => `<li>+ ${this.escape(l)}</li>`).join("")}</ul>
        </div>
        <div class="diff-col removed"><strong>Removed (${result.removed_lines.length})</strong>
          <ul>${result.removed_lines.slice(0, 20).map((l) => `<li>- ${this.escape(l)}</li>`).join("")}</ul>
        </div>
      </div>
    `;
  }

  escape(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
}
