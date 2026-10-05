"""
config.py — Kanooni Jawab Universal Configuration
==========================================
All runtime settings loaded from environment variables.
SQLModel table definitions live in ingestion/models.py.
"""
import ssl_fix  # noqa: F401 — before any HTTP/SSL clients
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]


class Config:
    # ── API Keys ──────────────────────────────────────────────────────────────
    GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY          = os.getenv("QDRANT_API_KEY")
    JWT_SECRET              = os.getenv("JWT_SECRET", "change-me-in-production")

    # ── Databases ─────────────────────────────────────────────────────────────
    DATABASE_URL            = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/legal_db")
    QDRANT_URL              = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_TIMEOUT          = int(os.getenv("QDRANT_TIMEOUT", "300"))

    # One Qdrant collection per jurisdiction. Cases and legislation share the
    # collection and are differentiated by the "doc_type" payload field.
    QDRANT_CANADA_COLLECTION = os.getenv("QDRANT_CANADA_COLLECTION", "canada")

    # ── Embedding (BAAI/bge-m3: dense 1024-dim + learned sparse) ──────────────
    EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    BGE_M3_USE_FP16      = os.getenv("BGE_M3_USE_FP16", "false").lower() in ("1", "true", "yes")
    BGE_M3_MAX_LENGTH    = int(os.getenv("BGE_M3_MAX_LENGTH", "8192"))
    DENSE_VECTOR_NAME    = os.getenv("DENSE_VECTOR_NAME", "text-dense")
    SPARSE_VECTOR_NAME   = os.getenv("SPARSE_VECTOR_NAME", "text-sparse")

    # ── Gemini ────────────────────────────────────────────────────────────────
    GEMINI_MODEL             = os.getenv("GEMINI_MODEL",             "gemini-2.5-flash")
    GEMINI_TEMPERATURE       = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
    INGEST_PROMPT_CHARS      = int(os.getenv("INGEST_PROMPT_CHARS",  "8000"))

    # ── Retriever ─────────────────────────────────────────────────────────────
    ALLOW_GENERAL_LEGAL_ANSWERS    = os.getenv("ALLOW_GENERAL_LEGAL_ANSWERS", "true").lower() in ("1","true","yes")
    # When true, generic non-document-grounded queries skip Qdrant/RAPTOR and are
    # answered directly by Gemini (definitions, broad concepts, etc.).
    SKIP_RETRIEVAL_FOR_GENERIC     = os.getenv("SKIP_RETRIEVAL_FOR_GENERIC", "true").lower() in ("1","true","yes")
    RETRIEVAL_TOP_K                = int(os.getenv("RETRIEVAL_TOP_K",                "5"))
    RETRIEVAL_TOP_K_DOCUMENT       = int(os.getenv("RETRIEVAL_TOP_K_DOCUMENT",       "15"))
    RETRIEVAL_TOP_K_AMBIGUOUS      = int(os.getenv("RETRIEVAL_TOP_K_AMBIGUOUS",      "8"))
    RETRIEVAL_TOP_K_HYBRID_CASE    = int(os.getenv("RETRIEVAL_TOP_K_HYBRID_CASE",    "10"))
    RETRIEVAL_TOP_K_HYBRID_LEGIS   = int(os.getenv("RETRIEVAL_TOP_K_HYBRID_LEGIS",   "10"))
    RETRIEVAL_TOP_K_SCENARIO_CASE  = int(os.getenv("RETRIEVAL_TOP_K_SCENARIO_CASE",  "12"))
    RETRIEVAL_TOP_K_SCENARIO_LEGIS = int(os.getenv("RETRIEVAL_TOP_K_SCENARIO_LEGIS", "12"))
    ANSWER_TOP_K                   = int(os.getenv("ANSWER_TOP_K",                   "10"))
    MMR_LAMBDA                     = float(os.getenv("MMR_LAMBDA",                   "0.7"))
    RRF_K                          = int(os.getenv("RRF_K",                          "60"))

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Root folder layout:
    #   MULTI_INPUT_ROOT/
    #     hong_kong/cases/        hong_kong/legislation/
    #     pakistan/cases/         pakistan/legislation/
    #     united_kingdom/cases/   united_kingdom/legislation/
    #     canada/cases/           canada/legislation/
    #     united_states/cases/    united_states/legislation/
    #     australia/cases/        australia/legislation/
    #     india/cases/            india/legislation/
    MULTI_INPUT_ROOT      = os.getenv("MULTI_INPUT_ROOT",      "./input_documents")
    INGEST_BATCH_SIZE     = int(os.getenv("INGEST_BATCH_SIZE",    "50"))
    QDRANT_UPSERT_BATCH   = int(os.getenv("QDRANT_UPSERT_BATCH",  "100"))
    INGEST_MAX_RETRIES    = int(os.getenv("INGEST_MAX_RETRIES",    "3"))

    # ── S3 (optional) ─────────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION            = os.getenv("AWS_REGION",       "ap-east-1")
    S3_BUCKET             = os.getenv("S3_BUCKET")
    S3_FOLDER_PATH        = os.getenv("S3_FOLDER_PATH",   "public/canada/cases")
    CA_LEGIS_S3_FOLDER_PATH = os.getenv("CA_LEGIS_S3_FOLDER_PATH", "public/canada/legislation")
    S3_PRESIGN_EXPIRES_SECONDS = int(os.getenv("S3_PRESIGN_EXPIRES_SECONDS", "3600"))

    # ── Local PDF storage (ingested documents served to citation viewer) ───
    PDF_STORAGE_DIR       = os.getenv("PDF_STORAGE_DIR", str(_REPO_ROOT / "storage" / "pdfs"))

    # ── Evaluation harness ───────────────────────────────────────────────────
    # Judge model used to score faithfulness / relevancy / context quality.
    # Defaults to the same model as generation; override with a stronger model
    # (e.g. gemini-2.5-pro) for higher-confidence judging if your quota allows.
    GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", GEMINI_MODEL)

    # Directory containing one golden-set JSON file per jurisdiction key,
    # e.g. evaluation/golden/pakistan.json, evaluation/golden/hong_kong.json
    EVAL_GOLDEN_DIR = os.getenv(
        "EVAL_GOLDEN_DIR",
        str(Path(__file__).resolve().parent / "evaluation" / "golden"),
    )

    # How many golden items to evaluate concurrently (Gemini calls are network
    # I/O bound). Keep at 1 if you are on a low Gemini rate-limit tier.
    EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))

    # Pass/fail thresholds used to colour-code the dashboard and CLI summary.
    EVAL_PASS_THRESHOLD_FAITHFULNESS = float(os.getenv("EVAL_PASS_THRESHOLD_FAITHFULNESS", "0.8"))
    EVAL_PASS_THRESHOLD_RELEVANCY    = float(os.getenv("EVAL_PASS_THRESHOLD_RELEVANCY",    "0.7"))
    EVAL_PASS_THRESHOLD_PRECISION    = float(os.getenv("EVAL_PASS_THRESHOLD_PRECISION",    "0.6"))

    # Directory where evaluation JSON reports are written
    EVAL_OUTPUT_DIR = os.getenv(
        "EVAL_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "evaluation" / "reports"),
    )

    # Shared secret required in the "X-Admin-Token" header to trigger or read
    # evaluation runs via the API. Must be set to a real secret in production.
    ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "")

    _JURISDICTION_COLLECTION_KEYS = (
        "hong_kong",
        "pakistan",
        "united_kingdom",
        "canada",
        "united_states",
        "australia",
        "india",
    )

    @classmethod
    def qdrant_collection_for(cls, jurisdiction_key: str) -> str:
        """Return the single per-jurisdiction Qdrant collection name."""
        key = (jurisdiction_key or "canada").strip().lower().replace("-", "_").replace(" ", "_")
        env_name = f"QDRANT_{key.upper()}_COLLECTION"
        override = os.getenv(env_name)
        if override:
            return override
        named = getattr(cls, env_name, None)
        if named:
            return named
        return key

    @classmethod
    def all_qdrant_collections(cls) -> list[str]:
        names: list[str] = []
        for key in cls._JURISDICTION_COLLECTION_KEYS:
            name = cls.qdrant_collection_for(key)
            if name not in names:
                names.append(name)
        return names