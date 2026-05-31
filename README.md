# Atman RAG

A full-stack RAG application for the Atman Artwork LLP hiring task. Users can load public sample documents or upload PDF/text/markdown files, select the documents to query, ask questions, and receive grounded answers with citations. If the retrieved sources do not support the answer, the app returns: `I couldn't find this in the provided sources.`

## Stack

- Backend: FastAPI
- Frontend: Next.js
- LLM: Gemini API, default `gemini-2.5-flash-lite`
- Embeddings: Gemini API, default `gemini-embedding-001`
- Vector database: Qdrant local mode via `qdrant-client`
- Sample corpus: public NASA Moon to Mars PDF documents

Qdrant local mode keeps the demo simple and free: no separate vector DB account is required. The code can move to Qdrant Cloud later by changing the vector store connection.

## Get The Free API Key

1. Go to Google AI Studio: https://aistudio.google.com/app/apikey
2. Sign in with a Google account.
3. Create an API key.
4. Keep billing disabled if you want to stay on the free tier.
5. Put the key in the root `.env` file:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

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
3. Chunks are embedded using Gemini embeddings.
4. Vectors and metadata are stored in Qdrant local mode.
5. A user question is embedded and searched against selected documents.
6. The top retrieved chunks are sent to Gemini with a strict JSON prompt.
7. The answer is returned only if retrieval and the model both support it.
8. Citations are shown from the retrieved chunks, including document, page, score, and snippet.

## Honesty Strategy

The app has three abstention layers:

- Retrieval threshold: if the top similarity score is below `MIN_RELEVANCE_SCORE`, the backend does not call the LLM and returns the "not found" answer.
- Prompt contract: Gemini must return JSON with `supported: false` when the provided passages do not clearly answer the question.
- Citation guard: if Gemini says an answer is supported but does not return a valid source ID, the backend abstains instead of showing an uncited answer.

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
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL=gemini-2.5-flash-lite`
   - `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`
   - `AUTO_INGEST_SAMPLES=true`
   - `ALLOW_SAMPLE_DOWNLOAD=true`
   - `FRONTEND_ORIGIN=*` for the first test, then replace it with the Vercel URL.
   - `FRONTEND_ORIGIN_REGEX=https://.*\.vercel\.app` if you want Vercel preview deployments to work too.
5. Deploy and open `https://your-render-api.onrender.com/api/health`.
6. After the first successful deploy, open the backend URL once. The free service may take a little while to wake up.

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
