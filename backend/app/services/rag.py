from datetime import datetime, timezone
from pathlib import Path
import re

from app.config import Settings
from app.schemas import AskResponse, Citation, DocumentRecord, RetrievedChunk
from app.services.chunker import chunk_pages
from app.services.documents import (
    detect_file_type,
    document_id_for,
    load_document,
    title_from_path,
)
from app.services.embeddings import Embedder
from app.services.gemini import GeminiError
from app.services.llm import LLMClient, LLMError
from app.services.registry import DocumentRegistry
from app.services.vector_store import VectorStore


UNKNOWN_ANSWER = "I couldn't find this in the provided sources."


class RagService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedder: Embedder,
        vector_store: VectorStore,
        registry: DocumentRegistry,
    ):
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.registry = registry

    async def ingest_path(
        self,
        path: Path,
        title: str | None = None,
        source_url: str | None = None,
        replace: bool = False,
    ) -> DocumentRecord:
        document_id = document_id_for(path, source_url=source_url)
        existing = self.registry.get(document_id)
        if (
            existing
            and not replace
            and self.vector_store.collection_exists()
            and self._record_matches_embedding(existing)
        ):
            return existing

        pages = load_document(path)
        chunks = chunk_pages(
            pages,
            chunk_chars=self.settings.chunk_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError(f"No extractable text found in {path.name}.")

        vectors = await self.embedder.embed_texts(
            [chunk.text for chunk in chunks],
            task_type="RETRIEVAL_DOCUMENT",
        )
        if not vectors:
            raise GeminiError("Embedding response was empty.")

        self.vector_store.ensure_collection(len(vectors[0]))
        self.vector_store.delete_document(document_id)

        record = DocumentRecord(
            id=document_id,
            title=title or title_from_path(path),
            filename=path.name,
            source_url=source_url,
            file_type=detect_file_type(path),
            created_at=datetime.now(tz=timezone.utc),
            chunk_count=len(chunks),
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self._embedding_model(),
            embedding_dimensions=len(vectors[0]),
        )

        payloads = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_id = f"{document_id}:{chunk.chunk_index}"
            payloads.append(
                {
                    "chunk_id": chunk_id,
                    "vector": vector,
                    "payload": {
                        "chunk_id": chunk_id,
                        "document_id": record.id,
                        "document_title": record.title,
                        "filename": record.filename,
                        "source_url": record.source_url,
                        "file_type": record.file_type,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                }
            )

        self.vector_store.upsert_chunks(payloads)
        self.registry.upsert(record)
        return record

    def _embedding_dimensions(self) -> int | None:
        if self.settings.embedding_provider.lower().strip() == "local":
            return self.settings.local_embedding_dimensions
        return None

    def _embedding_model(self) -> str | None:
        provider = self.settings.embedding_provider.lower().strip()
        if provider == "fastembed":
            return self.settings.fastembed_model
        if provider == "gemini":
            return self.settings.gemini_embedding_model
        if provider == "local":
            return f"local-hash-v2-{self.settings.local_embedding_dimensions}"
        return None

    def _record_matches_embedding(self, record: DocumentRecord) -> bool:
        provider = self.settings.embedding_provider.lower().strip()
        if record.embedding_provider != provider:
            return False
        if record.embedding_model != self._embedding_model():
            return False
        if provider == "local":
            return record.embedding_dimensions == self.settings.local_embedding_dimensions
        return True

    async def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> AskResponse:
        selected_ids = document_ids or []
        query_vector = await self.embedder.embed_text(
            question,
            task_type="RETRIEVAL_QUERY",
        )
        hits = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k or self.settings.top_k,
            document_ids=selected_ids,
        )
        retrieved = self._to_retrieved_chunks(hits)

        if not retrieved or retrieved[0].score < self.settings.min_relevance_score:
            return AskResponse(
                answer=UNKNOWN_ANSWER,
                supported=False,
                citations=[],
                retrieved_chunks=retrieved,
            )

        try:
            model_result = await self.llm.generate_json(
                self._build_prompt(question, retrieved)
            )
        except LLMError as exc:
            if retrieved[0].score >= self.settings.extractive_fallback_score:
                return self._extractive_fallback(question, retrieved, str(exc))
            return AskResponse(
                answer=UNKNOWN_ANSWER,
                supported=False,
                citations=[],
                retrieved_chunks=retrieved,
                raw_model_output={"llm_error": str(exc)},
            )
        supported = bool(model_result.get("supported"))
        answer = str(model_result.get("answer") or "").strip()
        citation_ids = [
            str(item)
            for item in model_result.get("citation_ids", [])
            if isinstance(item, str)
        ]

        if not supported or not answer:
            return AskResponse(
                answer=UNKNOWN_ANSWER,
                supported=False,
                citations=[],
                retrieved_chunks=retrieved,
                raw_model_output=model_result,
            )

        citations = self._citations_from_ids(citation_ids, retrieved)
        if not citations:
            return AskResponse(
                answer=UNKNOWN_ANSWER,
                supported=False,
                citations=[],
                retrieved_chunks=retrieved,
                raw_model_output=model_result,
            )

        return AskResponse(
            answer=answer,
            supported=True,
            citations=citations,
            retrieved_chunks=retrieved,
            raw_model_output=model_result,
        )

    def _to_retrieved_chunks(self, hits) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunks.append(
                RetrievedChunk(
                    source_id=f"S{index}",
                    chunk_id=str(payload.get("chunk_id")),
                    document_id=str(payload.get("document_id")),
                    document_title=str(payload.get("document_title")),
                    filename=str(payload.get("filename")),
                    source_url=payload.get("source_url"),
                    page=payload.get("page"),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                )
            )
        return chunks

    def _build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        used_chars = 0
        for chunk in chunks:
            block = (
                f"[{chunk.source_id}]\n"
                f"Document: {chunk.document_title}\n"
                f"File: {chunk.filename}\n"
                f"Page: {chunk.page if chunk.page is not None else 'n/a'}\n"
                f"Passage:\n{chunk.text}\n"
            )
            if used_chars + len(block) > self.settings.max_context_chars:
                break
            blocks.append(block)
            used_chars += len(block)

        context = "\n---\n".join(blocks)
        return f"""
You are a careful retrieval-augmented QA assistant.
Answer the user's question only from the provided sources.

Rules:
- If the sources do not clearly support the answer, return unsupported.
- If the question asks for secrets, credentials, current events, hiring details, or facts not present in the sources, return unsupported.
- Do not use outside knowledge.
- Cite only source IDs whose passage directly supports the answer.
- If you cannot cite a source ID for the answer, return unsupported.
- Return JSON only with this exact shape:
  {{"supported": true, "answer": "short grounded answer", "citation_ids": ["S1"]}}
  or
  {{"supported": false, "answer": "{UNKNOWN_ANSWER}", "citation_ids": []}}

Question:
{question}

Sources:
{context}
""".strip()

    @staticmethod
    def _citations_from_ids(
        source_ids: list[str],
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        chunk_by_id = {chunk.source_id: chunk for chunk in chunks}
        citations: list[Citation] = []
        for source_id in source_ids:
            chunk = chunk_by_id.get(source_id)
            if not chunk:
                continue
            snippet = chunk.text[:450].strip()
            if len(chunk.text) > 450:
                snippet += "..."
            citations.append(
                Citation(
                    source_id=chunk.source_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    filename=chunk.filename,
                    source_url=chunk.source_url,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    snippet=snippet,
                    score=chunk.score,
                )
            )
        return citations

    def _extractive_fallback(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        reason: str,
    ) -> AskResponse:
        best = chunks[0]
        answer = self._best_supported_excerpt(question, best.text)
        citation = self._citations_from_ids([best.source_id], chunks)
        return AskResponse(
            answer=f"From the retrieved source: {answer}",
            supported=True,
            citations=citation,
            retrieved_chunks=chunks,
            raw_model_output={"fallback": "extractive", "reason": reason},
        )

    @staticmethod
    def _best_supported_excerpt(question: str, text: str) -> str:
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) > 3
        }
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        if not sentences:
            return text[:500].strip()

        def score(sentence: str) -> int:
            terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            return len(query_terms & terms)

        ranked = sorted(sentences, key=score, reverse=True)
        excerpt = " ".join(ranked[:2]).strip()
        if len(excerpt) > 650:
            excerpt = excerpt[:650].rsplit(" ", 1)[0].strip() + "..."
        return excerpt
