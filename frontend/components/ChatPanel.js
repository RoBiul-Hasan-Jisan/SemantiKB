import { api } from "../services/api.js";

export class ChatPanel {
  constructor(container, getContext) {
    this.container = container;
    this.getContext = getContext; // () => { documentId, strategy, timeFilter }
    this.messages = [];
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="chat-panel">
        <div id="chat-messages" class="chat-messages"></div>
        <form id="chat-form" class="chat-form">
          <input id="chat-input" type="text" placeholder="Ask a question about your documents..." autocomplete="off" />
          <input id="time-filter" type="text" placeholder="time filter (optional, e.g. 2025 or latest)" class="time-filter-input" />
          <button type="submit">Ask</button>
        </form>
      </div>
    `;
    this.container.querySelector("#chat-form").addEventListener("submit", (e) => this.handleSubmit(e));
  }

  async handleSubmit(e) {
    e.preventDefault();
    const input = this.container.querySelector("#chat-input");
    const timeInput = this.container.querySelector("#time-filter");
    const query = input.value.trim();
    if (!query) return;
    input.value = "";

    this.addMessage("user", query);
    const thinkingId = this.addMessage("assistant", "Retrieving relevant context...", true);

    const ctx = this.getContext();
    try {
      const payload = {
        query,
        document_ids: ctx.documentId ? [ctx.documentId] : null,
        strategy: ctx.strategy || "semantic",
        time_filter: timeInput.value.trim() || null,
      };
      const response = await api.chat(payload);
      this.updateMessage(thinkingId, response.answer, response.citations, response.retrieved_chunks);
    } catch (err) {
      this.updateMessage(thinkingId, `Error: ${err.message}`, [], []);
    }
  }

  addMessage(role, text, pending = false) {
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    this.messages.push({ id, role, text });
    const el = document.createElement("div");
    el.className = `chat-msg ${role} ${pending ? "pending" : ""}`;
    el.id = id;
    el.innerHTML = `<div class="bubble">${this.escape(text)}</div>`;
    this.container.querySelector("#chat-messages").appendChild(el);
    this.scrollToBottom();
    return id;
  }

  updateMessage(id, text, citations = [], retrievedChunks = []) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("pending");
    const citeHtml = citations.length
      ? `<div class="citations">
           <strong>Sources:</strong>
           <ul>${citations
             .map((c) => `<li>${this.escape(c.document_name)} · v${c.version} · p.${c.page} · <code>${c.chunk_id}</code></li>`)
             .join("")}</ul>
         </div>`
      : "";
    const sourcesHtml = retrievedChunks.length
      ? `<details class="retrieved-sources">
           <summary>Retrieved context (${retrievedChunks.length})</summary>
           ${retrievedChunks
             .map(
               (c) => `<div class="source-block">
                  <div class="source-meta">${this.escape(c.filename)} · v${c.version} · p.${c.page_start}-${c.page_end}${
                 c.section ? " · " + this.escape(c.section) : ""
               }</div>
                  <div class="source-text">${this.escape(c.text.slice(0, 400))}${c.text.length > 400 ? "…" : ""}</div>
                </div>`
             )
             .join("")}
         </details>`
      : "";
    el.innerHTML = `<div class="bubble">${this.escape(text)}${citeHtml}</div>${sourcesHtml}`;
    this.scrollToBottom();
  }

  scrollToBottom() {
    const box = this.container.querySelector("#chat-messages");
    box.scrollTop = box.scrollHeight;
  }

  escape(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
}
