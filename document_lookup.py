"""Lookup ingested document metadata from Postgres."""

from __future__ import annotations

from typing import Any, Optional

from sqlmodel import Session

from models import CASE_MODEL_REGISTRY, LEGIS_MODEL_REGISTRY, engine


def _lookup_keys(doc_id: str) -> list[Any]:
    keys: list[Any] = [doc_id]
    try:
        keys.append(int(doc_id))
    except (TypeError, ValueError):
        pass
    return keys


def _get_row(session: Session, model: Any, doc_id: str) -> Any | None:
    for key in _lookup_keys(doc_id):
        try:
            row = session.get(model, key)
        except Exception:
            continue
        if row is not None:
            return row
    return None


def get_document_metadata(doc_id: str) -> Optional[dict[str, Any]]:
    with Session(engine) as session:
        for model in CASE_MODEL_REGISTRY.values():
            case = _get_row(session, model, doc_id)
            if case:
                return {
                    "doc_id": str(getattr(case, "id", doc_id)),
                    "file_name": case.file_name,
                    "doc_name": case.case_name,
                    "doc_type": "Case",
                    "file_url": case.file_url or f"/api/documents/{doc_id}/file",
                    "jurisdiction": case.jurisdiction,
                }
        for model in LEGIS_MODEL_REGISTRY.values():
            legis = _get_row(session, model, doc_id)
            if legis:
                return {
                    "doc_id": str(getattr(legis, "id", doc_id)),
                    "file_name": legis.file_name,
                    "doc_name": legis.legislation_name,
                    "doc_type": "Legislation",
                    "file_url": legis.file_url or f"/api/documents/{doc_id}/file",
                    "jurisdiction": legis.jurisdiction,
                }
    return None
