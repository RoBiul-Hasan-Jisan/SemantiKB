import { DocumentSidebar } from "../components/DocumentSidebar.js";
import { ChatPanel } from "../components/ChatPanel.js";
import { VersionHistory } from "../components/VersionHistory.js";
import { DocumentInsightsPanel } from "../components/DocumentInsightsPanel.js";
import { EvaluationPanel } from "../components/EvaluationPanel.js";

export class App {
  constructor(root) {
    this.root = root;
    this.selectedDocumentId = null;
    this.activeTab = "chat";
    this.render();
  }

  render() {
    this.root.innerHTML = `
      <div class="app-shell">
        <div id="sidebar-slot"></div>
        <div class="main-area">
          <nav class="tabs">
            <button data-tab="chat" class="tab active">Chat</button>
            <button data-tab="insights" class="tab">Summary & Chunks</button>
            <button data-tab="versions" class="tab">Version History</button>
            <button data-tab="eval" class="tab">Evaluation</button>
          </nav>
          <div id="tab-chat" class="tab-panel active"></div>
          <div id="tab-insights" class="tab-panel"></div>
          <div id="tab-versions" class="tab-panel"></div>
          <div id="tab-eval" class="tab-panel"></div>
        </div>
      </div>
    `;

    this.sidebar = new DocumentSidebar(this.root.querySelector("#sidebar-slot"), {
      onSelectDocument: (id) => this.handleSelectDocument(id),
    });
    this.chatPanel = new ChatPanel(this.root.querySelector("#tab-chat"), () => ({
      documentId: this.selectedDocumentId,
      strategy: "semantic",
    }));
    this.insightsPanel = new DocumentInsightsPanel(this.root.querySelector("#tab-insights"));
    this.versionHistory = new VersionHistory(this.root.querySelector("#tab-versions"));
    this.evalPanel = new EvaluationPanel(this.root.querySelector("#tab-eval"), () => this.selectedDocumentId);

    this.root.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => this.switchTab(btn.dataset.tab));
    });
  }

  switchTab(tab) {
    this.activeTab = tab;
    this.root.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    this.root.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    this.root.querySelector(`#tab-${tab}`).classList.add("active");
  }

  handleSelectDocument(documentId) {
    this.selectedDocumentId = documentId;
    this.insightsPanel.show(documentId);
    this.versionHistory.show(documentId);
  }
}
