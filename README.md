# Atman RAG

A full-stack RAG application for the Atman Artwork LLP hiring task. Users can load public sample documents or upload PDF/text/markdown files, select the documents to query, ask questions, and receive grounded answers with citations. If the retrieved sources do not support the answer, the app returns: `I couldn't find this in the provided sources.`

## Stack

- Backend: FastAPI
- Frontend: Next.js
- LLM: OpenRouter free model, default `openai/gpt-oss-20b:free`
- Embeddings: low-memory local hash embeddings by default, with optional FastEmbed/Hugging Face embeddings for machines with more RAM
- Vector database: Qdrant local mode via `qdrant-client`
- Sample corpus: public NASA Moon to Mars PDF documents

OpenRouter keeps chat generation free for low-volume demos. The deployed default uses local hash embeddings so indexing documents does not use LLM quota and fits Render's free 512 MiB memory limit. Qdrant local mode keeps the vector database free and simple.

## Get The Free API Key

Only the chat model needs an API key. Embeddings do not need a hosted API key.

1. Go to OpenRouter: https://openrouter.ai/settings/keys
2. Sign in.
3. Create an API key.
4. Put the key in the root `.env` file:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-oss-20b:free
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_APP_NAME=Atman RAG

EMBEDDING_PROVIDER=local
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
LOCAL_EMBEDDING_DIMENSIONS=512
```

If this specific free model is temporarily rate-limited, browse https://openrouter.ai/collections/free-models and replace `OPENROUTER_MODEL` with another current model slug that ends in `:free`.

FastEmbed is supported for local experiments by setting `EMBEDDING_PROVIDER=fastembed`, but it can exceed Render free-tier memory while loading the ONNX model. Keep `EMBEDDING_PROVIDER=local` for the deployed demo unless you move to a larger instance.

Do not commit `.env`. It is already ignored.

## Run Locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

Optional sample download:

```powershell
python scripts\download_sample_docs.py
```

You can also click `Load NASA sample docs` in the UI. That downloads and ingests the same public NASA PDFs.

## How It Works

1. Documents are parsed from PDF, text, or markdown.
2. Text is split into overlapping chunks with document/page metadata.
3. Chunks are embedded locally with a deterministic low-memory embedder.
4. Vectors and metadata are stored in Qdrant local mode.
5. A user question is embedded and searched against selected documents.
6. The top retrieved chunks are sent to OpenRouter with a strict JSON prompt.
7. The answer is returned only if retrieval and the model both support it.
8. Citations are shown from the retrieved chunks, including document, page, score, and snippet.

## Honesty Strategy

The app has three abstention layers:

- Retrieval threshold: if the top similarity score is below `MIN_RELEVANCE_SCORE`, the backend does not call the LLM and returns the "not found" answer.
- Prompt contract: the model must return JSON with `supported: false` when the provided passages do not clearly answer the question.
- Citation guard: if the model says an answer is supported but does not return a valid source ID, the backend abstains instead of showing an uncited answer.
- Rate-limit fallback: if a free LLM provider is temporarily rate-limited and retrieval is high confidence, the backend returns a short extractive answer from the top passage with a citation instead of failing the request.

This is intentionally conservative because the assignment values honesty over confident hallucination.

## Evaluation

Use [docs/evaluation.md](docs/evaluation.md) before submission. It includes answerable questions, abstention questions, and a citation spot-check table.

With the backend running and sample documents already ingested, you can also run:

```powershell
python scripts\run_eval.py --api-base http://localhost:8000
```

For the deployed app, run the same smoke check against Render:

```powershell
python scripts\run_eval.py --api-base https://your-render-api.onrender.com
```

The script checks whether answerable questions return supported answers with citations, and whether abstention questions return unsupported answers. Still do a manual citation spot check because semantic correctness cannot be fully proven by this script.

## Sample Questions

Answerable:

- What are NASA's Moon to Mars objectives?
- Why does NASA describe Moon to Mars as an evolutionary architecture?
- What role does Gateway play in NASA's lunar architecture?
- What are the main exploration challenges when comparing the Moon and Mars?

Should abstain:

- What is the CEO of Atman Artwork's favorite programming language?
- What salary did NASA assign to this hiring task?
- Which private database vendor does the document require?

## Deploy

Backend on Render:

1. Push this repo to GitHub.
2. In Render, create a new Web Service from the GitHub repo.
3. Use `render.yaml` if Render detects it. If configuring manually, set:
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `LLM_PROVIDER=openrouter`
   - `OPENROUTER_API_KEY=your_key`
   - `OPENROUTER_MODEL=openai/gpt-oss-20b:free`
   - `OPENROUTER_APP_NAME=Atman RAG`
   - `OPENROUTER_SITE_URL=https://your-vercel-domain.vercel.app`
   - `EMBEDDING_PROVIDER=local`
   - `FASTEMBED_MODEL=BAAI/bge-small-en-v1.5`
   - `LOCAL_EMBEDDING_DIMENSIONS=512`
   - `AUTO_INGEST_SAMPLES=true`
   - `ALLOW_SAMPLE_DOWNLOAD=true`
   - `FRONTEND_ORIGIN=*` for the first test, then replace it with the Vercel URL.
   - `FRONTEND_ORIGIN_REGEX=https://.*\.vercel\.app` if you want Vercel preview deployments to work too.
5. Deploy and open `https://your-render-api.onrender.com/api/health`.
6. The first deploy can take longer because the app downloads the sample PDFs and indexes them.

Frontend on Vercel:

1. Import the same GitHub repo in Vercel.
2. Set the project root directory to `frontend`.
3. Keep the framework preset as Next.js.
4. Add environment variable:
   - `NEXT_PUBLIC_API_BASE_URL=https://your-render-api.onrender.com`
5. Deploy.
6. Open the Vercel URL, load NASA docs, ask one answerable question and one abstention question.
7. Go back to Render and set `FRONTEND_ORIGIN=https://your-vercel-domain.vercel.app`, then redeploy the backend. If you are testing from Vercel preview URLs, also keep `FRONTEND_ORIGIN_REGEX=https://.*\.vercel\.app`.

If the frontend still calls `https://your-render-api.onrender.com`, the Vercel environment variable was not available at build time. Update `NEXT_PUBLIC_API_BASE_URL`, make sure it is enabled for the deployment environment you are using, and redeploy the frontend.

## Where It Breaks At Scale

- Qdrant local mode is fine for demos, but production should use Qdrant Cloud, pgvector, or another managed vector DB.
- Render free services may sleep, so the first request can be slow.
- Startup sample ingestion can be slow because PDFs must be downloaded, parsed, embedded, and indexed.
- Free OpenRouter models can have lower rate limits, higher latency, and changing availability.
- Local hash embeddings are intentionally lightweight for the free deployment. FastEmbed or a hosted embedding API would improve semantic retrieval on a larger budget.
- The retrieval threshold is static; a production app should tune it with evaluation questions.
- There is no user auth or per-user document isolation yet.
- Uploaded files are stored on the backend filesystem, which may be ephemeral on free hosting.

## With Another Week

- Add streaming answers.
- Add automated RAG evaluation with expected answer/abstention cases.
- Add document deletion and re-indexing controls.
- Add reranking for higher citation precision.
- Move vector storage to Qdrant Cloud or Postgres + pgvector.
- Add auth and per-user document collections.
