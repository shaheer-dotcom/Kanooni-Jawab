"""
structured_query.py — Relational Metadata Layer for Kanooni Jawab (PostgreSQL only)
===========================================================================
Provides exact-match SQL lookups over document metadata,
complementing Qdrant vector search for structured queries like:
  - "All cases decided in Lahore High Court in 2022"
  - "Cases citing Act No. 5 of 1997"
  - "Legislation effective after January 2020 in Pakistan"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from sqlalchemy import create_engine, text

from config import Config

log = logging.getLogger(__name__)


@dataclass
class CaseRecord:
    doc_id: str
    case_name: str
    citation: str
    court: str
    year: int
    jurisdiction: str
    parties: str
    judge: str
    doc_type: str = "Case"
    qdrant_id: Optional[str] = None
    file_name: Optional[str] = None


@dataclass
class LegislationRecord:
    doc_id: str
    title: str
    act_number: str
    jurisdiction: str
    effective_date: Optional[date]
    doc_type: str = "Legislation"
    qdrant_id: Optional[str] = None
    file_name: Optional[str] = None


class MetadataDB:
    """
    PostgreSQL-backed metadata store for structured legal document queries.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or Config.DATABASE_URL
        if not self.database_url.startswith("postgresql"):
            raise ValueError(
                "MetadataDB requires PostgreSQL DATABASE_URL. "
                "Please set Config.DATABASE_URL to a postgres URL."
            )
        self._engine = create_engine(self.database_url, future=True, pool_pre_ping=True)
        self._create_schema()
        log.info("MetadataDB initialised on PostgreSQL")

    def _create_schema(self) -> None:
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS cases (
                doc_id       TEXT PRIMARY KEY,
                case_name    TEXT NOT NULL,
                citation     TEXT,
                court        TEXT,
                year         INTEGER,
                jurisdiction TEXT NOT NULL,
                parties      TEXT,
                judge        TEXT,
                qdrant_id    TEXT,
                file_name    TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_cases_jurisdiction ON cases(jurisdiction)",
            "CREATE INDEX IF NOT EXISTS idx_cases_court ON cases(court)",
            "CREATE INDEX IF NOT EXISTS idx_cases_year ON cases(year)",
            "CREATE INDEX IF NOT EXISTS idx_cases_citation ON cases(citation)",
            """
            CREATE TABLE IF NOT EXISTS legislation (
                doc_id         TEXT PRIMARY KEY,
                title          TEXT NOT NULL,
                act_number     TEXT,
                jurisdiction   TEXT NOT NULL,
                effective_date DATE,
                qdrant_id      TEXT,
                file_name      TEXT,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_leg_jurisdiction ON legislation(jurisdiction)",
            "CREATE INDEX IF NOT EXISTS idx_leg_act_number ON legislation(act_number)",
            "CREATE INDEX IF NOT EXISTS idx_leg_eff_date ON legislation(effective_date)",
            """
            CREATE TABLE IF NOT EXISTS citations (
                citing_doc_id TEXT NOT NULL,
                cited_doc_id  TEXT NOT NULL,
                PRIMARY KEY (citing_doc_id, cited_doc_id),
                FOREIGN KEY (citing_doc_id) REFERENCES cases(doc_id),
                FOREIGN KEY (cited_doc_id)  REFERENCES cases(doc_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_cit_cited ON citations(cited_doc_id)",
            """
            CREATE TABLE IF NOT EXISTS leg_refs (
                case_doc_id TEXT NOT NULL,
                leg_doc_id  TEXT NOT NULL,
                PRIMARY KEY (case_doc_id, leg_doc_id),
                FOREIGN KEY (case_doc_id) REFERENCES cases(doc_id),
                FOREIGN KEY (leg_doc_id)  REFERENCES legislation(doc_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_legref_leg ON leg_refs(leg_doc_id)",
        ]
        with self._engine.begin() as conn:
            for stmt in ddl:
                conn.execute(text(stmt))

    def upsert_case(self, rec: CaseRecord) -> None:
        sql = text(
            """
            INSERT INTO cases
                (doc_id, case_name, citation, court, year,
                 jurisdiction, parties, judge, qdrant_id, file_name)
            VALUES
                (:doc_id, :case_name, :citation, :court, :year,
                 :jurisdiction, :parties, :judge, :qdrant_id, :file_name)
            ON CONFLICT (doc_id) DO UPDATE SET
                case_name = EXCLUDED.case_name,
                citation = EXCLUDED.citation,
                court = EXCLUDED.court,
                year = EXCLUDED.year,
                jurisdiction = EXCLUDED.jurisdiction,
                parties = EXCLUDED.parties,
                judge = EXCLUDED.judge,
                qdrant_id = EXCLUDED.qdrant_id,
                file_name = EXCLUDED.file_name
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "doc_id": rec.doc_id,
                    "case_name": rec.case_name,
                    "citation": rec.citation,
                    "court": rec.court,
                    "year": rec.year,
                    "jurisdiction": rec.jurisdiction,
                    "parties": rec.parties,
                    "judge": rec.judge,
                    "qdrant_id": rec.qdrant_id,
                    "file_name": rec.file_name,
                },
            )

    def upsert_legislation(self, rec: LegislationRecord) -> None:
        sql = text(
            """
            INSERT INTO legislation
                (doc_id, title, act_number, jurisdiction, effective_date,
                 qdrant_id, file_name)
            VALUES
                (:doc_id, :title, :act_number, :jurisdiction, :effective_date,
                 :qdrant_id, :file_name)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                act_number = EXCLUDED.act_number,
                jurisdiction = EXCLUDED.jurisdiction,
                effective_date = EXCLUDED.effective_date,
                qdrant_id = EXCLUDED.qdrant_id,
                file_name = EXCLUDED.file_name
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "doc_id": rec.doc_id,
                    "title": rec.title,
                    "act_number": rec.act_number,
                    "jurisdiction": rec.jurisdiction,
                    "effective_date": rec.effective_date,
                    "qdrant_id": rec.qdrant_id,
                    "file_name": rec.file_name,
                },
            )

    def add_citation(self, citing_id: str, cited_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("INSERT INTO citations (citing_doc_id, cited_doc_id) VALUES (:a,:b) ON CONFLICT DO NOTHING"),
                {"a": citing_id, "b": cited_id},
            )

    def add_leg_ref(self, case_id: str, leg_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("INSERT INTO leg_refs (case_doc_id, leg_doc_id) VALUES (:a,:b) ON CONFLICT DO NOTHING"),
                {"a": case_id, "b": leg_id},
            )

    def find_cases(
        self,
        jurisdiction: str,
        *,
        court: Optional[str] = None,
        year: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        judge: Optional[str] = None,
        party: Optional[str] = None,
        citation_contains: Optional[str] = None,
        limit: int = 50,
    ) -> list[CaseRecord]:
        clauses = ["jurisdiction = :jurisdiction"]
        params: dict[str, Any] = {"jurisdiction": jurisdiction, "limit": limit}

        if court:
            clauses.append("court ILIKE :court")
            params["court"] = f"%{court}%"
        if year:
            clauses.append("year = :year")
            params["year"] = year
        if year_from:
            clauses.append("year >= :year_from")
            params["year_from"] = year_from
        if year_to:
            clauses.append("year <= :year_to")
            params["year_to"] = year_to
        if judge:
            clauses.append("judge ILIKE :judge")
            params["judge"] = f"%{judge}%"
        if party:
            clauses.append("parties ILIKE :party")
            params["party"] = f"%{party}%"
        if citation_contains:
            clauses.append("citation ILIKE :citation_contains")
            params["citation_contains"] = f"%{citation_contains}%"

        sql = text(
            f"SELECT * FROM cases WHERE {' AND '.join(clauses)} ORDER BY year DESC NULLS LAST LIMIT :limit"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._row_to_case(r) for r in rows]

    def find_cases_citing_legislation(
        self, leg_doc_id: str, jurisdiction: str, limit: int = 50
    ) -> list[CaseRecord]:
        sql = text(
            """
            SELECT c.* FROM cases c
            JOIN leg_refs lr ON lr.case_doc_id = c.doc_id
            WHERE lr.leg_doc_id = :leg_doc_id AND c.jurisdiction = :jurisdiction
            ORDER BY c.year DESC NULLS LAST LIMIT :limit
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                sql,
                {"leg_doc_id": leg_doc_id, "jurisdiction": jurisdiction, "limit": limit},
            ).mappings().all()
        return [self._row_to_case(r) for r in rows]

    def find_cases_citing_case(
        self, cited_doc_id: str, jurisdiction: str, limit: int = 50
    ) -> list[CaseRecord]:
        sql = text(
            """
            SELECT c.* FROM cases c
            JOIN citations ci ON ci.citing_doc_id = c.doc_id
            WHERE ci.cited_doc_id = :cited_doc_id AND c.jurisdiction = :jurisdiction
            ORDER BY c.year DESC NULLS LAST LIMIT :limit
            """
        )
        with self._engine.connect() as conn:
            rows = conn.execute(
                sql,
                {"cited_doc_id": cited_doc_id, "jurisdiction": jurisdiction, "limit": limit},
            ).mappings().all()
        return [self._row_to_case(r) for r in rows]

    def get_citation_graph(self, doc_id: str, depth: int = 2) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        frontier = {doc_id}
        with self._engine.connect() as conn:
            for _ in range(depth):
                if not frontier:
                    break
                rows = conn.execute(
                    text(
                        """
                        SELECT citing_doc_id, cited_doc_id
                        FROM citations
                        WHERE citing_doc_id = ANY(:frontier)
                        """
                    ),
                    {"frontier": list(frontier)},
                ).mappings().all()
                next_frontier: set[str] = set()
                for r in rows:
                    citing = str(r["citing_doc_id"])
                    cited = str(r["cited_doc_id"])
                    graph.setdefault(citing, []).append(cited)
                    next_frontier.add(cited)
                frontier = next_frontier - set(graph.keys())
        return graph

    def find_legislation(
        self,
        jurisdiction: str,
        *,
        act_number: Optional[str] = None,
        title_contains: Optional[str] = None,
        effective_after: Optional[date] = None,
        effective_before: Optional[date] = None,
        limit: int = 50,
    ) -> list[LegislationRecord]:
        clauses = ["jurisdiction = :jurisdiction"]
        params: dict[str, Any] = {"jurisdiction": jurisdiction, "limit": limit}

        if act_number:
            clauses.append("act_number ILIKE :act_number")
            params["act_number"] = f"%{act_number}%"
        if title_contains:
            clauses.append("title ILIKE :title_contains")
            params["title_contains"] = f"%{title_contains}%"
        if effective_after:
            clauses.append("effective_date >= :effective_after")
            params["effective_after"] = effective_after
        if effective_before:
            clauses.append("effective_date <= :effective_before")
            params["effective_before"] = effective_before

        sql = text(
            f"SELECT * FROM legislation WHERE {' AND '.join(clauses)} "
            "ORDER BY effective_date DESC NULLS LAST LIMIT :limit"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._row_to_leg(r) for r in rows]

    def get_legislation_by_act_number(
        self, act_number: str, jurisdiction: str
    ) -> Optional[LegislationRecord]:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT * FROM legislation
                    WHERE act_number = :act_number AND jurisdiction = :jurisdiction
                    LIMIT 1
                    """
                ),
                {"act_number": act_number, "jurisdiction": jurisdiction},
            ).mappings().first()
        return self._row_to_leg(row) if row else None

    _STRUCTURED_PATTERNS = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\ball cases\b",
            r"\bby (judge|justice|court)\b",
            r"\bdecided in\b",
            r"\bbetween (\d{4}) and (\d{4})\b",
            r"\bsince \d{4}\b",
            r"\bbefore \d{4}\b",
            r"\bafter \d{4}\b",
            r"\bcases citing\b",
            r"\blegislation effective\b",
            r"\bhow many cases\b",
            r"\blist (?:all )?cases\b",
        ]
    ]

    @classmethod
    def is_structured_query(cls, query: str) -> bool:
        return any(p.search(query or "") for p in cls._STRUCTURED_PATTERNS)

    def structured_query(
        self, query: str, jurisdiction: str, limit: int = 20
    ) -> dict[str, Any]:
        q = query.lower()
        result_type = "unknown"
        results: list[Any] = []

        years = [int(m.group(0)) for m in re.finditer(r"\b(?:19|20)\d{2}\b", query)]
        year_from = min(years) if len(years) >= 2 else None
        year_to = max(years) if len(years) >= 2 else None
        single_year = years[0] if len(years) == 1 else None

        judge_match = re.search(r"\bjustice\s+([A-Za-z\s]+?)(?:\s+in|\s+of|$)", query, re.I)
        judge = judge_match.group(1).strip() if judge_match else None

        court_match = re.search(
            r"(supreme court|high court|court of appeal|district court|sessions court|federal court)",
            query, re.I,
        )
        court = court_match.group(1) if court_match else None

        if "citing" in q:
            cit_match = re.search(r"citing\s+([A-Z0-9\s\.\-/]+)", query, re.I)
            if cit_match:
                citation_fragment = cit_match.group(1).strip()
                results = self.find_cases(
                    jurisdiction,
                    citation_contains=citation_fragment,
                    limit=limit,
                )
                result_type = "case_citations"
        elif "legislation" in q or "act" in q or "ordinance" in q:
            eff_after = date(year_from, 1, 1) if year_from else None
            eff_before = date(year_to, 12, 31) if year_to else None
            results = self.find_legislation(
                jurisdiction,
                effective_after=eff_after,
                effective_before=eff_before,
                limit=limit,
            )
            result_type = "legislation"
        else:
            results = self.find_cases(
                jurisdiction,
                court=court,
                year=single_year,
                year_from=year_from,
                year_to=year_to,
                judge=judge,
                limit=limit,
            )
            result_type = "cases"

        answered = len(results) > 0
        return {
            "results": results,
            "answered": answered,
            "result_type": result_type,
            "count": len(results),
        }

    @staticmethod
    def _row_to_case(row: Any) -> CaseRecord:
        return CaseRecord(
            doc_id=row["doc_id"],
            case_name=row["case_name"],
            citation=row.get("citation") or "",
            court=row.get("court") or "",
            year=row.get("year") or 0,
            jurisdiction=row["jurisdiction"],
            parties=row.get("parties") or "",
            judge=row.get("judge") or "",
            qdrant_id=row.get("qdrant_id"),
            file_name=row.get("file_name"),
        )

    @staticmethod
    def _row_to_leg(row: Any) -> LegislationRecord:
        eff: Optional[date] = row.get("effective_date")
        if isinstance(eff, str):
            try:
                eff = date.fromisoformat(eff)
            except ValueError:
                eff = None
        return LegislationRecord(
            doc_id=row["doc_id"],
            title=row["title"],
            act_number=row.get("act_number") or "",
            jurisdiction=row["jurisdiction"],
            effective_date=eff,
            qdrant_id=row.get("qdrant_id"),
            file_name=row.get("file_name"),
        )

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
