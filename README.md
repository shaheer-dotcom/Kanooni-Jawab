# Kanooni Jawab

Multi-jurisdiction legal research chatbot. Ask questions about case law, legislation, or legal scenarios; answers are retrieved from your corpus, grounded in a selected jurisdiction, and returned with citations.

## How the chatbot works

```text
User question + jurisdiction
        │
        ▼
   FastAPI (/api/chat/stream)
        │
        ▼
   LegalRetriever pipeline
        │
   ┌────┴────────────────────────────────────────┐
   │ 1. Structured SQL? (MetadataDB / Postgres)  │── yes → formatted metadata answer
   │ 2. Intent + query expansion (Gemini #1)     │
   │ 3. Generic & non-grounded?                  │── yes → direct LLM answer (no search)
   │ 4. Hybrid search in Qdrant (dense + sparse) │
   │ 5. RRF fusion → MMR re-ranking              │
   │ 6. Self-RAG retry if evidence is weak       │
   │ 7. Grounded answer (Gemini #2, streamed)    │
   └─────────────────────────────────────────────┘
        │
        ▼
   SSE events → chat UI (thinking / tokens / sources)
```

### Request flow

1. **UI** (`static/`) waits until `/health` reports the retriever is ready, then sends the question and jurisdiction to `POST /api/chat/stream`.
2. **API** (`api.py`) normalizes the jurisdiction into a `JurisdictionContext` and streams Server-Sent Events (SSE) from the retriever.
3. **Retriever** (`retriever.py`) runs the pipeline below and yields events: `thinking`, `token`, `done`, or `error`.

### Pipeline stages

| Stage | What happens |
| --- | --- |
| **Jurisdiction binding** | Every answer is scoped to one jurisdiction (Canada, Hong Kong, Pakistan, UK, US, Australia, India). Filters and prompts use that jurisdiction’s config from `jurisdiction_registry.py`. |
| **Structured metadata path** | Queries that look like exact lookups (court, year, act number, etc.) hit PostgreSQL via `structured_query.py` instead of vector search. |
| **Intent classification** | Gemini classifies the query (`case_law`, `legislation`, `hybrid`, `topic`, `scenario_analysis`, `generic`) and expands it into a small set of sub-queries (RAG-Fusion). |
| **Direct LLM path** | Generic, non-document-grounded questions can skip corpus search and answer from general legal knowledge (no invented citations). |
| **Hybrid retrieval** | Sub-queries are embedded with **BGE-M3** (dense + learned sparse) and searched in **Qdrant**. Cases and legislation share a collection and are filtered by payload fields such as `doc_type`. |
| **Ranking** | Ranked lists are merged with **Reciprocal Rank Fusion (RRF)**, then diversified with **MMR**. |
| **Self-RAG** | If retrieved evidence looks insufficient, the pipeline can re-query before answering. |
| **Grounded generation** | Gemini streams an answer using retrieved chunks and intent-specific prompt rules (e.g. IRAC for scenarios, section citations for legislation). |
| **Citations & PDFs** | The UI shows sources; document metadata and PDF links come from `/api/documents/{doc_id}` (local storage or S3). |

### Streaming UX

The frontend consumes SSE:

- `thinking` — status text (“Analyzing your query…”, “searching Canada corpus…”)
- `token` — answer text chunks as they are generated
- `done` — final answer, intent, route, latency, and references
- `error` — failure message

## Stack

| Layer | Technology |
| --- | --- |
| Web UI | Static HTML/CSS/JS |
| API | FastAPI + Uvicorn |
| Embeddings | BAAI/bge-m3 (dense + sparse) |
| Vector DB | Qdrant |
| Metadata DB | PostgreSQL (SQLAlchemy / SQLModel) |
| LLM | Google Gemini |
| Optional files | Local PDF dir or AWS S3 |

## Quick start

### Prerequisites

- Python 3.10+
- Running **PostgreSQL** (metadata)
- Running **Qdrant** with your jurisdiction collections
- **Gemini** API key

### Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys and URLs
```

Required env vars include `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `DATABASE_URL`. See `.env.example` for the full list.

### Run

```bash
uvicorn api:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000). The first boot loads the embedding model and connects to Qdrant/Postgres; the UI stays disabled until `/health` reports `ready: true`.

### CLI (optional)

You can also exercise the retriever from the command line via `retriever.py` (see its `main()` entrypoint).

## Project layout

```text
api.py                   # FastAPI server, chat SSE, document & eval routes
retriever.py             # End-to-end retrieval + answer pipeline
structured_query.py      # PostgreSQL metadata / structured lookups
jurisdiction_registry.py # Per-jurisdiction citation patterns & vocab
middleware.py            # JurisdictionContext / access control helpers
bge_m3_embeddings.py     # Dense + sparse embedding helpers
config.py                # Settings from environment
document_lookup.py       # Document metadata lookup
models.py                # Data models
static/                  # Chat UI
evaluation/              # Golden-set eval harness + Gemini judge
tests/                   # Unit tests
```

## Main API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Retriever readiness |
| `GET` | `/api/jurisdictions` | Supported jurisdictions |
| `POST` | `/api/chat/stream` | Chat (SSE) — body: `{ "query", "jurisdiction" }` |
| `GET` | `/api/documents/{doc_id}` | Document metadata |
| `GET` | `/api/documents/{doc_id}/file` | PDF (local or S3 redirect) |
| `POST` | `/api/admin/eval/run` | Start evaluation (`X-Admin-Token`) |
| `GET` | `/api/admin/eval/status` | Eval progress |
| `GET` | `/api/admin/eval/results` | Latest eval results |

## Evaluation

Golden questions live under `evaluation/golden/`. Admin endpoints run the harness in the background and score answers (faithfulness, relevancy, precision) when a judge model is configured. Set `ADMIN_API_TOKEN` in `.env` before calling admin routes.

## Notes

- Secrets stay in `.env` (gitignored). Commit only `.env.example`.
- Answers are intended to be **jurisdiction-scoped** and **corpus-grounded** when retrieval runs; the generic path deliberately avoids fabricating case or statute citations.
- Ingestion of new PDFs into Qdrant/Postgres is separate from this chat service; this repo focuses on retrieval and the web UI.
