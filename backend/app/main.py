from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import AskRequest, AskResponse, IngestSamplesResponse, UploadResponse
from app.services.documents import save_upload
from app.services.embeddings import create_embedder
from app.services.gemini import GeminiClient, GeminiError
from app.services.llm import LLMError, create_llm
from app.services.rag import RagService
from app.services.registry import DocumentRegistry
from app.services.samples import download_sample_sources, sample_files
from app.services.vector_store import VectorStore


settings = get_settings()
gemini = GeminiClient(settings)
embedder = create_embedder(settings, gemini)
llm = create_llm(settings, gemini)
registry = DocumentRegistry(settings.registry_path)
vector_store = VectorStore(settings.qdrant_path, settings.collection_name)
rag = RagService(settings, llm, embedder, vector_store, registry)

app = FastAPI(title=settings.app_name)

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
configured_origins = [
    origin.strip()
    for origin in settings.frontend_origin.split(",")
    if origin.strip()
]
if "*" in configured_origins:
    origins = ["*"]
else:
    for origin in configured_origins:
        if origin not in origins:
            origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=settings.frontend_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.sample_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    if (
        settings.auto_ingest_samples
        and (settings.embedding_provider != "gemini" or gemini.is_configured())
    ):
        await ingest_samples_internal(download_missing=settings.allow_sample_download)


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "status": "ok",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_configured": llm.is_configured(),
        "gemini_configured": gemini.is_configured(),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.fastembed_model
        if settings.embedding_provider == "fastembed"
        else settings.embedding_provider,
        "documents": len(registry.list()),
    }


@app.get("/api/documents")
def list_documents():
    return registry.list()


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    try:
        documents = []
        for upload in files:
            path = await save_upload(upload, settings.uploads_dir)
            documents.append(await rag.ingest_path(path, replace=True))
        return UploadResponse(documents=documents)
    except (ValueError, GeminiError, LLMError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/documents/ingest-samples", response_model=IngestSamplesResponse)
async def ingest_samples(download_missing: bool = True):
    try:
        documents, downloaded, skipped = await ingest_samples_internal(download_missing)
        return IngestSamplesResponse(
            documents=documents,
            downloaded=downloaded,
            skipped=skipped,
        )
    except (ValueError, GeminiError, LLMError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        return await rag.answer(
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )
    except (GeminiError, LLMError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def ingest_samples_internal(download_missing: bool):
    downloaded: list[str] = []
    skipped: list[str] = []
    if download_missing:
        if not settings.allow_sample_download:
            raise ValueError("Sample downloading is disabled by ALLOW_SAMPLE_DOWNLOAD=false.")
        downloaded, skipped = await run_in_threadpool(download_sample_sources, settings)

    documents = []
    files = sample_files(settings)
    if not files:
        raise ValueError(
            "No sample documents found. Run scripts/download_sample_docs.py or enable downloading."
        )
    for path, source in files:
        documents.append(
            await rag.ingest_path(
                path,
                title=source.get("title"),
                source_url=source.get("url"),
                replace=False,
            )
        )
    return documents, downloaded, skipped
