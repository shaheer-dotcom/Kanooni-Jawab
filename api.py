"""
api.py — FastAPI web server for Kanooni Jawab chat UI
==============================================
Run:  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config
from document_lookup import get_document_metadata
from evaluation.report import matrix_summary
from evaluation.runner import get_latest_run, is_run_in_progress, run_evaluation_background
from middleware import dev_context, normalize_jurisdiction
from retriever import LegalRetriever

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_retriever: Optional[LegalRetriever] = None
_retriever_lock = threading.Lock()
_init_error: Optional[str] = None


def _get_retriever() -> LegalRetriever:
    global _retriever, _init_error
    if _retriever is not None:
        return _retriever
    if _init_error:
        raise HTTPException(status_code=503, detail=f"Retriever failed to load: {_init_error}")
    raise HTTPException(status_code=503, detail="Retriever is still initializing. Please wait.")


def _init_retriever_background() -> None:
    global _retriever, _init_error
    try:
        log.info("Background: initialising LegalRetriever...")
        with _retriever_lock:
            _retriever = LegalRetriever()
        log.info("Background: LegalRetriever ready.")
    except Exception as exc:
        log.exception("Retriever initialisation failed: %s", exc)
        _init_error = str(exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    thread = threading.Thread(target=_init_retriever_background, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Kanooni Jawab", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    jurisdiction: str = Field(default="Canada")


class EvalRunRequest(BaseModel):
    jurisdictions: list[str] = Field(default_factory=list)
    skip_judge: bool = False


def _require_admin(x_admin_token: str = Header(default="", alias="X-Admin-Token")) -> None:
    if not Config.ADMIN_API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_TOKEN is not configured on the server",
        )
    if x_admin_token != Config.ADMIN_API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


SUPPORTED_JURISDICTIONS = [
    "Canada",
    "Hong Kong",
    "Pakistan",
    "United Kingdom",
    "United States",
    "Australia",
    "India",
]


@app.get("/health")
def health():
    return {
        "ready": _retriever is not None,
        "error": _init_error,
        "jurisdictions": SUPPORTED_JURISDICTIONS,
    }


@app.get("/api/jurisdictions")
def jurisdictions():
    return {"jurisdictions": SUPPORTED_JURISDICTIONS}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    meta = get_document_metadata(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")
    return meta


def _object_key_from_file_url(file_url: str) -> str:
    if file_url.startswith("s3://"):
        without_scheme = file_url[5:]
        _bucket, _, key = without_scheme.partition("/")
        return unquote(key)
    path = unquote(urlparse(file_url).path).lstrip("/")
    if not path:
        raise ValueError(f"Could not parse S3 key from URL: {file_url}")
    return path


def _presigned_s3_url(file_url: str) -> Optional[str]:
    if not Config.AWS_ACCESS_KEY_ID or not Config.AWS_SECRET_ACCESS_KEY or not Config.S3_BUCKET:
        return None
    try:
        import boto3
    except ImportError:
        return None
    try:
        object_key = _object_key_from_file_url(file_url)
        client = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION,
        )
        return client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": Config.S3_BUCKET, "Key": object_key},
            ExpiresIn=Config.S3_PRESIGN_EXPIRES_SECONDS,
        )
    except Exception as exc:
        log.warning("S3 presign failed for %s: %s", file_url, exc)
        return None


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    meta = get_document_metadata(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")

    file_url = str(meta.get("file_url") or "")
    if file_url.startswith(("http://", "https://", "s3://")):
        presigned = _presigned_s3_url(file_url)
        if presigned:
            return RedirectResponse(url=presigned, status_code=302)
        if file_url.startswith(("http://", "https://")):
            return RedirectResponse(url=file_url, status_code=302)

    file_name = meta.get("file_name") or f"{doc_id}.pdf"
    candidates = [
        Path(Config.PDF_STORAGE_DIR) / f"{doc_id}.pdf",
        Path(Config.PDF_STORAGE_DIR) / file_name,
    ]
    pdf_path = next((path for path in candidates if path.is_file()), None)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=file_name,
    )


@app.post("/api/chat/stream")
def chat_stream(body: ChatRequest):
    retriever = _get_retriever()

    try:
        canonical = normalize_jurisdiction(body.jurisdiction)
        ctx = dev_context(canonical)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_generator():
        for event in retriever.answer_with_events(query=body.query, jurisdiction_ctx=ctx):
            event_type = event.get("type", "message")
            payload = json.dumps(event.get("data", {}), default=str)
            yield f"event: {event_type}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/admin/eval/run")
def admin_eval_run(body: EvalRunRequest, _: None = Depends(_require_admin)):
    retriever = _get_retriever()
    if is_run_in_progress():
        raise HTTPException(status_code=409, detail="Evaluation already in progress")
    keys = body.jurisdictions or None
    run_id = run_evaluation_background(
        retriever,
        jurisdiction_keys=keys,
        skip_judge=body.skip_judge,
    )
    return {"status": "started", "run_id": run_id}


@app.get("/api/admin/eval/status")
def admin_eval_status(_: None = Depends(_require_admin)):
    latest = get_latest_run()
    return {
        "in_progress": is_run_in_progress(),
        "latest": matrix_summary(latest) if latest else None,
    }


@app.get("/api/admin/eval/results")
def admin_eval_results(_: None = Depends(_require_admin)):
    latest = get_latest_run()
    if not latest:
        raise HTTPException(status_code=404, detail="No evaluation results yet")
    return latest


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
