import { api } from "../services/api.js";

const METRIC_LABELS = {
  precision_at_k: "Precision@K",
  recall_at_k: "Recall@K",
  mrr: "MRR",
  ndcg_at_k: "NDCG@K",
  answer_relevance: "Answer Relevance",
  faithfulness: "Faithfulness",
  num_chunks: "# Chunks",
  storage_bytes: "Storage (bytes)",
  avg_retrieval_latency_ms: "Latency (ms)",
};

export class EvaluationPanel {
  constructor(container, getSelectedDocumentId) {
    this.container = container;
    this.getSelectedDocumentId = getSelectedDocumentId;
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="eval-panel">
        <h3>Chunking Strategy Evaluation</h3>
        <p class="hint">Provide labeled queries (query + relevant page numbers) to compare
        Fixed vs Recursive vs Semantic chunking on identical documents.</p>
        <textarea id="eval-queries" rows="6" placeholder='[{"query": "What is the refund policy?", "document_id": "doc_xxx", "relevant_pages": [2,3]}]'></textarea>
        <button id="run-eval">Run Evaluation</button>
        <div id="eval-results"></div>
      </div>
    `;
    this.container.querySelector("#run-eval").addEventListener("click", () => this.runEval());
  }

  async runEval() {
    const resultsEl = this.container.querySelector("#eval-results");
    let queries;
    try {
      queries = JSON.parse(this.container.querySelector("#eval-queries").value);
    } catch (e) {
      resultsEl.innerHTML = `<p class="error">Invalid JSON: ${e.message}</p>`;
      return;
    }
    resultsEl.innerHTML = `<p>Running evaluation across fixed / recursive / semantic strategies...</p>`;
    try {
      const results = await api.evaluate(queries);
      this.renderResults(results);
    } catch (e) {
      resultsEl.innerHTML = `<p class="error">${e.message}</p>`;
    }
  }

  renderResults(results) {
    const metricKeys = Object.keys(METRIC_LABELS);
    const resultsEl = this.container.querySelector("#eval-results");
    resultsEl.innerHTML = `
      <table class="eval-table">
        <thead>
          <tr><th>Metric</th>${results.map((r) => `<th>${r.strategy}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${metricKeys
            .map(
              (key) => `<tr>
                <td>${METRIC_LABELS[key]}</td>
                ${results
                  .map((r) => `<td>${typeof r[key] === "number" ? r[key].toFixed(key.includes("bytes") || key.includes("chunks") ? 0 : 3) : r[key]}</td>`)
                  .join("")}
              </tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;
  }
}
