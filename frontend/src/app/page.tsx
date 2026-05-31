"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  API_BASE,
  AskResponse,
  DocumentRecord,
  apiJson
} from "@/lib/api";

export default function Home() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [status, setStatus] = useState("Checking backend...");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [uploading, setUploading] = useState(false);

  const selectedIds = useMemo(() => Array.from(selected), [selected]);

  async function refreshDocuments() {
    const docs = await apiJson<DocumentRecord[]>("/api/documents");
    setDocuments(docs);
    setSelected((current) => {
      if (current.size > 0) return current;
      return new Set(docs.map((doc) => doc.id));
    });
  }

  useEffect(() => {
    async function load() {
      try {
        const health = await apiJson<{
          status: string;
          gemini_configured: boolean;
          documents: number;
        }>("/api/health");
        setStatus(
          health.gemini_configured
            ? "Backend ready"
            : "Backend running, Gemini key missing"
        );
        await refreshDocuments();
      } catch (err) {
        setStatus("Backend unavailable");
        setError(err instanceof Error ? err.message : "Could not reach backend");
      }
    }
    load();
  }, []);

  async function ingestSamples() {
    setError("");
    setIngesting(true);
    try {
      await apiJson("/api/documents/ingest-samples?download_missing=true", {
        method: "POST"
      });
      await refreshDocuments();
      setStatus("NASA sample documents loaded");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not ingest samples");
    } finally {
      setIngesting(false);
    }
  }

  async function uploadFiles(files: FileList | null) {
    if (!files?.length) return;
    setError("");
    setUploading(true);
    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    try {
      const response = await fetch(`${API_BASE}/api/documents/upload`, {
        method: "POST",
        body: form
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        throw new Error(errorBody?.detail || "Upload failed");
      }
      await refreshDocuments();
      setStatus("Upload ingested");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setError("");
    setLoading(true);
    setAnswer(null);
    try {
      const result = await apiJson<AskResponse>("/api/ask", {
        method: "POST",
        body: JSON.stringify({
          question,
          document_ids: selectedIds
        })
      });
      setAnswer(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Question failed");
    } finally {
      setLoading(false);
    }
  }

  function toggleDocument(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <main className="shell">
      <section className="header">
        <div>
          <p className="eyebrow">Atman Artwork LLP task</p>
          <h1>Grounded document Q&A</h1>
          <p className="summary">
            Ask questions against uploaded or sample NASA Moon to Mars documents.
            Answers cite retrieved passages and abstain when the sources do not
            support a response.
          </p>
        </div>
        <div className="status">{status}</div>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="grid">
        <aside className="panel documents">
          <div className="panelHeader">
            <h2>Documents</h2>
            <span>{documents.length} loaded</span>
          </div>

          <button
            className="primary"
            onClick={ingestSamples}
            disabled={ingesting}
          >
            {ingesting ? "Loading NASA docs..." : "Load NASA sample docs"}
          </button>

          <label className="upload">
            <input
              type="file"
              accept=".pdf,.txt,.md,.markdown"
              multiple
              onChange={(event) => uploadFiles(event.target.files)}
              disabled={uploading}
            />
            {uploading ? "Uploading..." : "Upload PDF / text / markdown"}
          </label>

          <div className="docList">
            {documents.length === 0 && (
              <p className="muted">Load sample docs or upload your own files.</p>
            )}
            {documents.map((doc) => (
              <label className="docItem" key={doc.id}>
                <input
                  type="checkbox"
                  checked={selected.has(doc.id)}
                  onChange={() => toggleDocument(doc.id)}
                />
                <span>
                  <strong>{doc.title}</strong>
                  <small>
                    {doc.file_type.toUpperCase()} · {doc.chunk_count} chunks
                  </small>
                </span>
              </label>
            ))}
          </div>
        </aside>

        <section className="panel qa">
          <form onSubmit={askQuestion} className="askForm">
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What are the main Moon to Mars objectives?"
              rows={4}
            />
            <button
              className="primary"
              disabled={loading || selectedIds.length === 0}
            >
              {loading ? "Answering..." : "Ask from selected sources"}
            </button>
          </form>

          {answer && (
            <div className="answerBlock">
              <div className={answer.supported ? "badge ok" : "badge warn"}>
                {answer.supported ? "Grounded answer" : "Not found in sources"}
              </div>
              <p className="answer">{answer.answer}</p>

              <h2>Citations</h2>
              {answer.citations.length === 0 && (
                <p className="muted">No supporting citation was found.</p>
              )}
              <div className="citations">
                {answer.citations.map((citation) => (
                  <article className="citation" key={citation.source_id}>
                    <div>
                      <strong>
                        {citation.source_id} · {citation.document_title}
                      </strong>
                      <small>
                        Page {citation.page ?? "n/a"} · score{" "}
                        {citation.score.toFixed(3)}
                      </small>
                    </div>
                    <p>{citation.snippet}</p>
                    {citation.source_url && (
                      <a
                        href={citation.source_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open source
                      </a>
                    )}
                  </article>
                ))}
              </div>

              <details className="retrieval">
                <summary>Retrieved passages</summary>
                {answer.retrieved_chunks.map((chunk) => (
                  <article key={chunk.source_id}>
                    <strong>
                      {chunk.source_id} · {chunk.document_title}
                    </strong>
                    <small>
                      Page {chunk.page ?? "n/a"} · score {chunk.score.toFixed(3)}
                    </small>
                    <p>{chunk.text}</p>
                  </article>
                ))}
              </details>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
