"""
retriever.py — Kanooni Jawab Enhanced Multi-Jurisdiction Pipeline (v2)
================================================================
Full production pipeline with ALL improvements integrated:

  1.  Jurisdiction enforcement   — from JurisdictionContext (server-enforced)
  2.  Structured query detection — MetadataDB for exact-match SQL lookups
  3.  Gemini Call #1             — intent + RAG-Fusion (jurisdiction-aware)
  4.  Hybrid Search              — Dense + Sparse per sub-query
  5.  RRF Fusion                 — merge all ranked lists → Top-K
  6.  MMR Re-ranking             — diversity selection
  7.  Self-RAG retry             — re-queries on insufficient evidence
  8.  Per-intent prompts         — different prompt templates per intent type
  9.  Gemini Call #2             — grounded answer generation (streaming)
"""

from __future__ import annotations

import ssl_fix  # noqa: F401 — must run before google.genai / aiohttp

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from google import genai
from qdrant_client import QdrantClient, models as qmodels

from bge_m3_embeddings import embed_query_sparse, get_embeddings
from config import Config
from jurisdiction_registry import (
    build_case_routing_regex,
    build_legislation_routing_regex,
    get_jurisdiction_config,
    get_jurisdiction_key,
    get_legislation_qdrant_conditions,
    get_vocab_hint_block,
    normalize_jurisdiction,
)
from middleware import JurisdictionContext, JurisdictionAccessError
from structured_query import MetadataDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Intent data class
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    intent_type: str = "generic"
    is_document_grounded: bool = False
    needs_cases: bool = True
    needs_legislation: bool = True
    route_label: str = "ambiguous"
    sub_queries: list[str] = field(default_factory=list)
    rationale: str = ""
    is_structured: bool = False  # NEW: triggers SQL path


# ---------------------------------------------------------------------------
# RRF helpers (unchanged)
# ---------------------------------------------------------------------------

_RRF_K = 60


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    score_map: dict[str, dict[str, Any]],
    top_n: int = 100,
) -> list[str]:
    rrf_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _rrf_score(rank)
    sorted_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)
    return sorted_ids[:top_n]


# ---------------------------------------------------------------------------
# Scenario-analysis regex (jurisdiction-agnostic)
# ---------------------------------------------------------------------------

_SCENARIO_ANALYSIS_REGEX = re.compile(
    r"|".join([
        r"\bdiscuss\b", r"\banaly[sz]e\b", r"\bdetermine\b",
        r"\bapplicable\s+standard\b", r"\bstandard\s+of\s+proof\b",
        r"\bburden\s+of\s+proof\b", r"\brole\s+of\s+evidence\b",
        r"\bcustody\b", r"\bcare\s+arrangement\b", r"\bcosts\b",
        r"\binterim\s+arrangement\b", r"\ballegations?\b",
        r"\bshould\s+determine\b",
    ]),
    re.IGNORECASE,
)

_DOCUMENT_GROUNDED_REGEX = re.compile(
    r"|".join([
        r"\bapplicant\b", r"\brespondent\b", r"\bpetitioner\b",
        r"\bdefendant\b", r"\bplaintiff\b", r"\bappellant\b",
        r"\bclaimant\b", r"\bcase name\b", r"\bthis case\b",
        r"\bthis judgment\b", r"\bthis document\b", r"\bthis ordinance\b",
        r"\baccording to the\b", r"\bkey factors\b", r"\bfactors for\b",
        r"\bwho is the\b", r"\bwhat is the (case|judgment)\b",
    ]),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Per-intent prompt templates
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_ADDITIONS: dict[str, str] = {
    "legislation": """
LEGISLATION-SPECIFIC RULES:
- Always cite the specific section/article number alongside the provision's content.
- Note if a section has been amended, repealed, or has transitional provisions.
- If multiple sections apply, list them in logical order (general → specific).
- Distinguish between primary legislation and subsidiary legislation/regulations.
""",
    "case_law": """
CASE LAW-SPECIFIC RULES:
- Identify the ratio decidendi (binding principle) vs obiter dicta.
- Note the court level and whether the decision is binding or persuasive.
- If multiple cases are cited, distinguish leading authority from supporting cases.
- State the outcome clearly: allowed / dismissed / remitted / varied.
""",
    "hybrid": """
HYBRID QUERY RULES:
- First address the statutory framework (legislation), then how courts have
  interpreted it (case law).
- Cross-reference: when citing a case, note which statutory provisions it interpreted.
- If legislation and case law appear to conflict, flag the tension explicitly.
""",
    "topic": """
TOPIC QUERY RULES:
- Structure your answer as: (1) Legal definition, (2) Statutory basis,
  (3) Key judicial interpretations, (4) Practical application.
- Use sources to ground each part; do not invent citations.
""",
    "scenario_analysis": """
SCENARIO ANALYSIS RULES:
- Apply the IRAC framework: Issue → Rule → Application → Conclusion.
- Identify all relevant legal issues in the scenario first.
- For each issue, cite the applicable statute sections and/or case principles.
- Be explicit about gaps in evidence and how they affect the analysis.
- Provide a clear conclusion per issue, noting any uncertainty.
""",
    "generic": """
GENERAL QUERY RULES:
- Provide a clear, well-structured explanation using careful general legal knowledge.
- Do not invent specific case citations, statute numbers, dates, or party names.
- This answer is not grounded in a retrieved document corpus; keep it educational and high-level.
""",
}


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class LegalRetriever:
    """
    Full multi-jurisdiction Kanooni Jawab pipeline with all production improvements.
    """

    def __init__(self) -> None:
        # Load embedding models before QdrantClient on Windows — initializing
        # Qdrant first can trigger a native pyarrow access violation when
        # sentence-transformers loads afterward.
        logging.info("Loading BGE-M3 embedding model: %s", Config.EMBEDDING_MODEL)
        self.embeddings = get_embeddings()
        logging.info("BGE-M3 dense + learned sparse encoder ready.")

        logging.info("Connecting to Qdrant: %s", Config.QDRANT_URL)
        self.q_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=getattr(Config, "QDRANT_TIMEOUT", 300),
        )
        logging.info("Qdrant client ready.")

        self.gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        logging.info("Connecting to PostgreSQL metadata store...")
        self.metadata_db = MetadataDB()

        logging.info("Ensuring Qdrant payload indexes...")
        self._ensure_filter_indexes()
        logging.info("LegalRetriever initialisation complete.")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_doc_type(doc_type: str) -> str:
        kind = (doc_type or "").strip().lower()
        if kind == "case":
            return "Case"
        if kind == "legislation":
            return "Legislation"
        raise ValueError("doc_type must be 'Case' or 'Legislation'.")

    def _collection_for(self, jurisdiction_key: str, doc_type: str = "") -> str:
        # doc_type is retained for call-site compatibility; Canada-style ingest
        # stores cases and legislation in one collection per jurisdiction.
        _ = doc_type
        return Config.qdrant_collection_for(jurisdiction_key)

    def _collection_for_doc_type(self, doc_type: str, jurisdiction_key: str = "canada") -> str:
        self._canonical_doc_type(doc_type)
        return self._collection_for(jurisdiction_key, doc_type)

    @staticmethod
    def _is_document_grounded_query(query: str) -> bool:
        q = query or ""
        if _DOCUMENT_GROUNDED_REGEX.search(q):
            return True
        return len(q) >= 250 and bool(_SCENARIO_ANALYSIS_REGEX.search(q))

    # ------------------------------------------------------------------
    # Jurisdiction-aware routing
    # ------------------------------------------------------------------

    def _get_routing_regexes(
        self, jurisdiction_key: str
    ) -> tuple[Optional[re.Pattern], Optional[re.Pattern]]:
        """Build case + legislation routing regexes for the active jurisdiction."""
        case_re = build_case_routing_regex(jurisdiction_key)
        leg_re = build_legislation_routing_regex(jurisdiction_key)
        return case_re, leg_re

    # ------------------------------------------------------------------
    # Step 1 — Intent Classification + Query Expansion (jurisdiction-aware)
    # ------------------------------------------------------------------

    def _build_intent_prompt(self, query: str, jurisdiction_ctx: JurisdictionContext) -> str:
        vocab_block = get_vocab_hint_block(jurisdiction_ctx.registry_key)
        cfg = jurisdiction_ctx.config

        court_hierarchy_str = ""
        if cfg and cfg.court_hierarchy:
            court_hierarchy_str = (
                f"Court hierarchy for {jurisdiction_ctx.canonical_name}:\n"
                + "\n".join(f"  • {c}" for c in cfg.court_hierarchy)
            )

        return f"""
You are Kanooni Jawab-Router, a legal query analyst for the {jurisdiction_ctx.canonical_name} jurisdiction.

{vocab_block}

{court_hierarchy_str}

Given a user query you must:

A) CLASSIFY INTENT into exactly one of:
   - "generic"           — general legal concept or definition
   - "specific_document" — asks about a named case, judgment, or statute
   - "topic"             — a legal topic/area of law
   - "case_law"          — asks specifically about court judgments/decisions
   - "legislation"       — asks about statutes, ordinances, sections, act numbers
   - "hybrid"            — requires both case law and legislation
   - "scenario_analysis" — asks to apply law to a set of facts

B) DETERMINE routing:
   - needs_cases:       true/false
   - needs_legislation: true/false
   - For intent_type "generic" that is NOT document-grounded: set BOTH needs_cases
     and needs_legislation to false and route_label to "generic" (no corpus search).

C) DETERMINE document_grounded: true when the query references specific parties,
   case names, act/cap/section numbers, or "this case/judgment/ordinance".

D) DETERMINE is_structured: true when the query asks for a list/count of cases/laws
   by court, year, judge, or act number — i.e. a metadata search, not semantic.

E) GENERATE RAG-FUSION sub-queries:
   Produce 2 alternative phrasings that preserve legal meaning but vary vocabulary
   using {jurisdiction_ctx.canonical_name}-specific legal terminology.
   Include the ORIGINAL query as sub_queries[0].

Return ONLY ONE valid JSON object:
{{
  "intent_type": "<one of the seven types above>",
  "is_document_grounded": <true|false>,
  "is_structured": <true|false>,
  "needs_cases": <true|false>,
  "needs_legislation": <true|false>,
  "route_label": "<case|legislation|hybrid|ambiguous|generic|structured>",
  "sub_queries": ["<original>", "<variant1>", "<variant2>"],
  "rationale": "<one sentence explaining the classification>"
}}

USER QUERY (Jurisdiction: {jurisdiction_ctx.canonical_name}):
{query}
"""

    def _parse_intent_response(self, raw: str, query: str) -> Optional[QueryIntent]:
        json_str = self._extract_first_json_object(raw)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        sub_queries: list[str] = data.get("sub_queries") or []
        if not sub_queries or sub_queries[0] != query:
            sub_queries = [query] + [q for q in sub_queries if q != query]
        sub_queries = sub_queries[:5]

        intent_type = str(data.get("intent_type", "generic"))
        is_document_grounded = bool(data.get("is_document_grounded", False))
        is_structured = bool(data.get("is_structured", False))
        needs_cases = bool(data.get("needs_cases", True))
        needs_legislation = bool(data.get("needs_legislation", True))
        route_label = str(data.get("route_label", "ambiguous"))

        intent = QueryIntent(
            intent_type=intent_type,
            is_document_grounded=is_document_grounded,
            is_structured=is_structured,
            needs_cases=needs_cases,
            needs_legislation=needs_legislation,
            route_label=route_label,
            sub_queries=sub_queries,
            rationale=str(data.get("rationale", "")),
        )
        return self._normalize_intent_routing(intent)

    def classify_and_expand(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        _max_retries: int = 2,
    ) -> QueryIntent:
        """Gemini Call #1 — jurisdiction-aware intent + sub-query expansion."""
        prompt = self._build_intent_prompt(query, jurisdiction_ctx)

        for attempt in range(1, _max_retries + 1):
            try:
                resp = self.gemini_client.models.generate_content(
                    model=Config.GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                    },
                )
                raw = resp.text or ""
                intent = self._parse_intent_response(raw, query)
                if intent is None:
                    raise ValueError("No valid JSON in intent response")

                logging.info(
                    "Intent (attempt %d): type=%s grounded=%s structured=%s route=%s sub_queries=%d",
                    attempt,
                    intent.intent_type,
                    intent.is_document_grounded,
                    intent.is_structured,
                    intent.route_label,
                    len(intent.sub_queries),
                )
                return intent

            except Exception as exc:
                logging.warning("Intent attempt %d/%d failed: %s", attempt, _max_retries, exc)

        logging.warning("All intent attempts failed; using heuristic fallback.")
        return self._heuristic_intent(query, jurisdiction_ctx)

    # ------------------------------------------------------------------
    # Heuristic fallback (now jurisdiction-aware)
    # ------------------------------------------------------------------

    def _heuristic_intent(
        self, query: str, jurisdiction_ctx: JurisdictionContext
    ) -> QueryIntent:
        case_re, leg_re = self._get_routing_regexes(jurisdiction_ctx.registry_key)

        leg_hit = bool(leg_re and leg_re.search(query))
        case_hit = bool(case_re and case_re.search(query))
        scenario_hit = bool(_SCENARIO_ANALYSIS_REGEX.search(query))
        structured_hit = MetadataDB.is_structured_query(query)

        if structured_hit:
            route, nc, nl, intent_type = "structured", True, True, "generic"
        elif leg_hit and case_hit:
            route, nc, nl, intent_type = "hybrid", True, True, "hybrid"
        elif leg_hit:
            route, nc, nl, intent_type = "legislation", False, True, "legislation"
        elif case_hit:
            route, nc, nl, intent_type = "case", True, False, "case_law"
        elif scenario_hit:
            route, nc, nl, intent_type = "hybrid", True, True, "scenario_analysis"
        else:
            route, nc, nl, intent_type = "generic", False, False, "generic"

        sub_queries = self._generate_fallback_sub_queries(query, jurisdiction_ctx)

        intent = QueryIntent(
            intent_type=intent_type,
            is_document_grounded=self._is_document_grounded_query(query),
            is_structured=structured_hit,
            needs_cases=nc,
            needs_legislation=nl,
            route_label=route,
            sub_queries=sub_queries,
            rationale="heuristic fallback",
        )
        return self._normalize_intent_routing(intent)

    @staticmethod
    def _generate_fallback_sub_queries(
        query: str, jurisdiction_ctx: JurisdictionContext
    ) -> list[str]:
        """Produce keyword-variation sub-queries using jurisdiction-specific vocab."""
        sub_queries = [query]
        cfg = jurisdiction_ctx.config
        jname = jurisdiction_ctx.canonical_name

        generic_synonyms = [
            ("legislation", "statute"), ("statute", "legislation"),
            ("ordinance", "act"), ("act", "ordinance"),
            ("assets", "property"), ("property", "assets"),
            ("case", "judgment"), ("judgment", "ruling"),
            ("court", "tribunal"),
        ]

        for old, new in generic_synonyms:
            if old.lower() in query.lower():
                variant = re.sub(re.escape(old), new, query, count=1, flags=re.IGNORECASE)
                if variant != query and variant not in sub_queries:
                    sub_queries.append(variant)
                if len(sub_queries) >= 4:
                    break

        if len(sub_queries) < 3 and cfg and cfg.legal_vocab_hints:
            hint = cfg.legal_vocab_hints[0]
            sub_queries.append(f"{query} — {jname} {hint}")

        if len(sub_queries) < 4:
            sub_queries.append(f"legal provisions related to: {query}")

        return sub_queries[:5]

    @staticmethod
    def _normalize_intent_routing(intent: QueryIntent) -> QueryIntent:
        """Ensure generic non-grounded queries do not trigger corpus search."""
        # Cap sub-queries to 3 to limit embedding calls
        if len(intent.sub_queries) > 3:
            intent.sub_queries = intent.sub_queries[:3]
        if (
            intent.intent_type == "generic"
            and not intent.is_document_grounded
            and not intent.is_structured
            and intent.route_label != "manual"
        ):
            intent.needs_cases = False
            intent.needs_legislation = False
            intent.route_label = "generic"
        return intent

    @staticmethod
    def should_skip_retrieval(intent: QueryIntent) -> bool:
        """True when the query should be answered by Gemini without Qdrant/RAPTOR."""
        if not Config.SKIP_RETRIEVAL_FOR_GENERIC:
            return False
        if not Config.ALLOW_GENERAL_LEGAL_ANSWERS:
            return False
        if intent.route_label == "manual":
            return False
        if intent.is_structured or intent.is_document_grounded:
            return False
        if intent.intent_type != "generic":
            return False
        return not intent.needs_cases and not intent.needs_legislation

    def _build_direct_answer_prompt(
        self,
        query: str,
        intent: QueryIntent,
        jurisdiction_ctx: JurisdictionContext,
    ) -> str:
        intent_addition = _INTENT_SYSTEM_ADDITIONS.get("generic", "")
        return f"""
You are Kanooni Jawab, a legal research assistant specialised in {jurisdiction_ctx.canonical_name} law.

Active jurisdiction: {jurisdiction_ctx.canonical_name}
Query intent: {intent.intent_type}

{intent_addition.strip()}

DIRECT-ANSWER MODE (no retrieved corpus):
- Answer using careful general legal knowledge only.
- Do NOT invent specific case citations, statute numbers, dates, or party names.
- Do NOT claim the answer is grounded in retrieved documents.
- Set "sources_used": [] and "insufficient_evidence": false when you can explain the concept.
- If the question requires jurisdiction-specific statute text or a named case you cannot
  answer from general knowledge, say so briefly and set "insufficient_evidence": true.

LANGUAGE: Match the user's query language entirely.

STYLE: Plain text only — no HTML, no URLs.

Return ONLY ONE valid JSON object:
{{
  "answer": "<your answer here>",
  "insufficient_evidence": false,
  "sources_used": []
}}

USER QUERY ({jurisdiction_ctx.canonical_name}):
{query}

Return JSON only.
"""

    def _answer_without_retrieval(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        intent: QueryIntent,
        *,
        stream: bool = False,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        prompt = self._build_direct_answer_prompt(query, intent, jurisdiction_ctx)
        if stream:
            full_text = self._stream_generate(prompt)
        else:
            resp = self.gemini_client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": Config.GEMINI_TEMPERATURE,
                    "max_output_tokens": Config.GEMINI_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
            full_text = resp.text or ""

        parsed = self._parse_answer_json(full_text)
        parsed["references"] = []
        parsed["sources_used"] = []
        parsed["retrieved_chunks"] = 0
        parsed["top_chunks"] = 0
        if include_debug:
            parsed["_eval_debug"] = self._build_eval_debug(
                jurisdiction=jurisdiction_ctx.canonical_name,
                all_chunks=[],
                top_chunks=[],
                context="",
                self_rag_retry_used=False,
            )
        return parsed

    def _direct_answer_events(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        intent: QueryIntent,
    ) -> Iterator[dict[str, Any]]:
        prompt = self._build_direct_answer_prompt(query, intent, jurisdiction_ctx)
        for fragment in self._stream_generate_iter(prompt):
            yield {"type": "token", "data": {"text": fragment}}
        full_text = getattr(self, "_last_stream_full_text", "")
        parsed = self._parse_answer_json(full_text)
        parsed["references"] = []
        parsed["sources_used"] = []
        parsed["retrieved_chunks"] = 0
        parsed["top_chunks"] = 0
        return parsed

    # ------------------------------------------------------------------
    # Step 2 — Qdrant filter building (jurisdiction-aware)
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        doc_type: Optional[str],
        jurisdiction: str,
        query: str,
        jurisdiction_key: str,
    ) -> qmodels.Filter:
        must: list[qmodels.Condition] = [
            qmodels.FieldCondition(
                key="jurisdiction", match=qmodels.MatchValue(value=jurisdiction)
            ),
        ]

        # Only add doc_type filter when a specific type is requested
        canonical_type: Optional[str] = None
        if doc_type is not None:
            canonical_type = self._canonical_doc_type(doc_type)
            must.insert(0, qmodels.FieldCondition(
                key="doc_type", match=qmodels.MatchValue(value=canonical_type)
            ))

        # Jurisdiction-specific legislation ID filtering
        if canonical_type == "Legislation":
            leg_conditions = get_legislation_qdrant_conditions(query, jurisdiction_key)
            if leg_conditions:
                must.append(
                    qmodels.Filter(
                        should=[
                            qmodels.FieldCondition(
                                key=cond["key"],
                                match=qmodels.MatchValue(value=cond["value"]),
                            )
                            for cond in leg_conditions
                        ]
                    )
                )

        return qmodels.Filter(must=must)

    # ------------------------------------------------------------------
    # Step 3 — Hybrid search (dense + sparse)
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        collection: str,
        vector: list[float],
        query_filter: qmodels.Filter,
        top_k: int,
    ) -> list[Any]:
        dense_name = getattr(Config, "DENSE_VECTOR_NAME", "text-dense")
        try:
            resp = self.q_client.query_points(
                collection_name=collection,
                query=vector,
                using=dense_name,
                query_filter=query_filter,
                limit=top_k,
            )
            return getattr(resp, "points", resp)
        except AttributeError:
            return self.q_client.search(
                collection_name=collection,
                query_vector=(dense_name, vector),
                query_filter=query_filter,
                limit=top_k,
            )

    def _sparse_search(
        self,
        collection: str,
        query_text: str,
        query_filter: qmodels.Filter,
        top_k: int,
    ) -> list[Any]:
        sparse_name = getattr(Config, "SPARSE_VECTOR_NAME", "text-sparse")
        try:
            sparse_vec = embed_query_sparse(query_text)
            resp = self.q_client.query_points(
                collection_name=collection,
                query=sparse_vec,
                using=sparse_name,
                query_filter=query_filter,
                limit=top_k,
            )
            return getattr(resp, "points", resp)
        except Exception as exc:
            logging.warning("Sparse search unavailable for '%s': %s", collection, exc)
            return []

    def _hybrid_search_collection(
        self,
        collection: str,
        sub_queries: list[str],
        query_filter: qmodels.Filter,
        top_k_per_query: int,
    ) -> tuple[list[list[str]], dict[str, dict[str, Any]]]:
        ranked_lists: list[list[str]] = []
        score_map: dict[str, dict[str, Any]] = {}

        # Batch-embed all sub-queries in one model call instead of one per loop
        all_vectors = self.embeddings.embed_documents(sub_queries)

        for sub_q, vector in zip(sub_queries, all_vectors):

            dense_results = self._dense_search(collection, vector, query_filter, top_k_per_query)
            dense_ids: list[str] = []
            for pt in dense_results:
                pid = str(pt.id)
                dense_ids.append(pid)
                if pid not in score_map:
                    payload = dict(getattr(pt, "payload", {}) or {})
                    payload["_dense_score"] = float(getattr(pt, "score", 0.0))
                    score_map[pid] = payload
            if dense_ids:
                ranked_lists.append(dense_ids)

            sparse_results = self._sparse_search(collection, sub_q, query_filter, top_k_per_query)
            sparse_ids: list[str] = []
            for pt in sparse_results:
                pid = str(pt.id)
                sparse_ids.append(pid)
                if pid not in score_map:
                    payload = dict(getattr(pt, "payload", {}) or {})
                    score_map[pid] = payload
                score_map[pid]["_sparse_score"] = float(getattr(pt, "score", 0.0))
            if sparse_ids:
                ranked_lists.append(sparse_ids)

        # Fallback: retry with jurisdiction-only filter if zero results
        if not score_map:
            logging.info("Zero results; retrying with jurisdiction-only filter.")
            jurisdiction_val = ""
            for must_cond in (query_filter.must or []):
                if (
                    isinstance(must_cond, qmodels.FieldCondition)
                    and must_cond.key == "jurisdiction"
                ):
                    jurisdiction_val = must_cond.match.value if hasattr(must_cond.match, "value") else ""
                    break

            if jurisdiction_val:
                fallback_filter = qmodels.Filter(must=[
                    qmodels.FieldCondition(key="jurisdiction", match=qmodels.MatchValue(value=jurisdiction_val))
                ])
                results = self._dense_search(collection, all_vectors[0], fallback_filter, top_k_per_query)
                fallback_ids: list[str] = []
                for pt in results:
                    pid = str(pt.id)
                    fallback_ids.append(pid)
                    if pid not in score_map:
                        payload = dict(getattr(pt, "payload", {}) or {})
                        payload["_dense_score"] = float(getattr(pt, "score", 0.0))
                        score_map[pid] = payload
                if fallback_ids:
                    ranked_lists.append(fallback_ids)

        return ranked_lists, score_map

    # ------------------------------------------------------------------
    # Step 4 — RRF Fusion
    # ------------------------------------------------------------------

    def _fuse_results(
        self,
        all_ranked_lists: list[list[str]],
        all_score_maps: list[dict[str, dict[str, Any]]],
        top_n: int = 100,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for sm in all_score_maps:
            for pid, payload in sm.items():
                if pid not in merged:
                    merged[pid] = payload
                else:
                    merged[pid].update(
                        {k: v for k, v in payload.items() if k not in merged[pid]}
                    )

        top_ids = reciprocal_rank_fusion(all_ranked_lists, merged, top_n=top_n)
        return [merged.get(pid, {}) for pid in top_ids]

    # ------------------------------------------------------------------
    # Step 6 — MMR Re-ranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
        lambda_mult: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Score-based MMR — uses RRF/dense scores already present on each chunk
        instead of re-embedding with BGE-M3. Eliminates the expensive
        embed_documents() call on the candidate pool.

        Relevance  = normalised dense score from Qdrant retrieval.
        Redundancy = fraction of already-selected chunks sharing the same doc_id,
                     so results are drawn from diverse source documents.
        """
        if lambda_mult is None:
            lambda_mult = Config.MMR_LAMBDA
        if len(chunks) <= top_k:
            return chunks

        dense_scores = [float(c.get("_dense_score", 0.0)) for c in chunks]
        max_dense = max(dense_scores) if max(dense_scores) > 0 else 1.0
        relevance = [s / max_dense for s in dense_scores]

        selected: list[int] = []
        selected_doc_ids: list[str] = []
        remaining = set(range(len(chunks)))

        for _ in range(top_k):
            best_idx, best_score = -1, -float("inf")
            for idx in remaining:
                doc_id = chunks[idx].get("doc_id") or chunks[idx].get("parent_id") or ""
                redundancy = (
                    selected_doc_ids.count(doc_id) / len(selected_doc_ids)
                    if selected_doc_ids else 0.0
                )
                score = lambda_mult * relevance[idx] - (1.0 - lambda_mult) * redundancy
                if score > best_score:
                    best_score, best_idx = score, idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_doc_ids.append(
                chunks[best_idx].get("doc_id") or chunks[best_idx].get("parent_id") or ""
            )
            remaining.discard(best_idx)

        logging.info("MMR: %d candidates → %d selected (λ=%.2f)", len(chunks), len(selected), lambda_mult)
        return [chunks[i] for i in selected]

    # ------------------------------------------------------------------
    # Step 7 — Answer generation prompts (per intent type)
    # ------------------------------------------------------------------

    def _build_answer_prompt(
        self,
        query: str,
        context: str,
        *,
        document_grounded: bool,
        intent: QueryIntent,
        jurisdiction_ctx: JurisdictionContext,
    ) -> str:
        intent_type = intent.intent_type
        allow_general = Config.ALLOW_GENERAL_LEGAL_ANSWERS
        general_mode = allow_general and not document_grounded

        # Select the right intent-specific addition
        intent_key = intent_type if intent_type in _INTENT_SYSTEM_ADDITIONS else "generic"
        intent_addition = _INTENT_SYSTEM_ADDITIONS[intent_key]

        if general_mode:
            grounding_rule = """
GENERAL LEGAL MODE:
- You may use careful general legal knowledge for definitions and topic-level questions.
- Do NOT invent specific case citations, statute numbers, dates, or jurisdiction-specific rules.
- If RETRIEVED SOURCES contain relevant information, you MUST cite them via "sources_used".
- Set "insufficient_evidence": false if you can provide a valid explanation.
"""
        else:
            grounding_rule = f"""
DOCUMENT-GROUNDED MODE (jurisdiction: {jurisdiction_ctx.canonical_name}):
- Every factual claim MUST be grounded in RETRIEVED SOURCES.
- Do NOT use outside knowledge. Cite [SOURCE N] ids in "sources_used".
- If sources do not support the answer, set "insufficient_evidence": true.
"""

        return f"""
You are Kanooni Jawab, a legal research assistant specialised in {jurisdiction_ctx.canonical_name} law.

Active jurisdiction: {jurisdiction_ctx.canonical_name}
Query intent: {intent_type}
Document-grounded mode: {document_grounded}

{intent_addition.strip()}

{grounding_rule.strip()}

CRITICAL — PREFER ANSWERING OVER REFUSING:
- If RETRIEVED SOURCES contain ANY relevant information, attempt an answer and cite those sources.
- Only return "Insufficient evidence" when sources are truly unrelated to the query.
- For partial information, answer what you can and note the gaps explicitly.

LANGUAGE: Match the user's query language entirely.

STYLE: Plain text only — no HTML, no URLs. You may use short headings ending with colons
and bullet lines starting with "- ".

Return ONLY ONE valid JSON object:
{{
  "answer": "<your answer here>",
  "insufficient_evidence": false,
  "sources_used": [1, 3, 5]
}}

RETRIEVED SOURCES (ranked by relevance):
{context}

USER QUERY ({jurisdiction_ctx.canonical_name}):
{query}

Return JSON only.
"""

    # ------------------------------------------------------------------
    # Step 8 — Self-RAG retry on insufficient evidence
    # ------------------------------------------------------------------

    def _self_rag_retry(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        intent: QueryIntent,
        stream: bool,
        attempt: int = 1,
        include_debug: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Broaden the search and retry answer generation once.
        Called when the primary pipeline returns insufficient_evidence=True.
        """
        logging.info("Self-RAG retry (attempt %d) for query: %.80s", attempt, query)

        # Broaden: force both collections, use original query only
        broadened_intent = QueryIntent(
            intent_type=intent.intent_type,
            is_document_grounded=False,  # relax grounding on retry
            needs_cases=True,
            needs_legislation=True,
            route_label="ambiguous",
            sub_queries=[query],
            rationale="self-rag-retry",
        )

        return self._run_vector_pipeline(
            query=query,
            jurisdiction_ctx=jurisdiction_ctx,
            intent=broadened_intent,
            top_k=getattr(Config, "RETRIEVAL_TOP_K", 10) * 2,
            stream=stream,
            allow_retry=False,  # prevent infinite recursion
            include_debug=include_debug,
        )

    def _ensure_filter_indexes(self) -> None:
        collections = Config.all_qdrant_collections()
        fields = [
            "doc_type", "jurisdiction", "jurisdiction_key", "parent_id",
            "cap_number", "cap_label", "legislation_number",  # HK
            "act_number", "statute_id", "ordinance_number",   # Pakistan / UK / Canada
            "chapter_number", "rsc_reference",               # Canada / UK
        ]
        for col in collections:
            for fname in fields:
                try:
                    self.q_client.create_payload_index(
                        collection_name=col,
                        field_name=fname,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass  # Already exists or collection missing — non-fatal

    # ------------------------------------------------------------------
    # Context + reference builders
    # ------------------------------------------------------------------

    def _build_grounded_context(self, chunks: list[dict[str, Any]]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            snippet = str(c.get("text", "")).strip().replace("\n", " ")
            blocks.append(
                "\n".join(filter(None, [
                    f"[SOURCE {i}]",
                    f"corpus: {c.get('doc_type', '')}",
                    f"file_name: {c.get('file_name', c.get('doc_name', 'unknown'))}",
                    f"doc_name: {c.get('doc_name', '')}",
                    f"jurisdiction: {c.get('jurisdiction', '')}",
                    f"chunk_index: {c.get('chunk_index', '')}",
                    f"text: {snippet[:1800]}",
                ]))
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_eval_debug(
        *,
        jurisdiction: str,
        all_chunks: list[dict[str, Any]],
        top_chunks: list[dict[str, Any]],
        context: str = "",
        self_rag_retry_used: bool = False,
    ) -> dict[str, Any]:
        """Debug payload for the offline evaluation harness only."""

        def _chunk_record(rank: int, chunk: dict[str, Any]) -> dict[str, Any]:
            return {
                "rank": rank,
                "parent_id": chunk.get("parent_id"),
                "doc_name": chunk.get("doc_name") or chunk.get("file_name") or "",
                "file_name": chunk.get("file_name") or "",
                "chunk_index": chunk.get("chunk_index"),
                "jurisdiction": chunk.get("jurisdiction", ""),
                "doc_type": chunk.get("doc_type", ""),
                "dense_score": chunk.get("_dense_score"),
                "sparse_score": chunk.get("_sparse_score"),
            }

        return {
            "all_chunk_jurisdictions": [c.get("jurisdiction", "") for c in all_chunks],
            "top_chunk_jurisdictions": [c.get("jurisdiction", "") for c in top_chunks],
            "all_chunk_doc_names": [
                c.get("doc_name") or c.get("file_name") or "" for c in all_chunks
            ],
            "top_chunk_doc_names": [
                c.get("doc_name") or c.get("file_name") or "" for c in top_chunks
            ],
            "top_chunk_records": [
                _chunk_record(i, c) for i, c in enumerate(top_chunks, start=1)
            ],
            "retrieved_context": context,
            "expected_jurisdiction": jurisdiction,
            "self_rag_retry_used": self_rag_retry_used,
        }

    def _build_references(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs = []
        for i, c in enumerate(chunks, start=1):
            doc_name = str(c.get("doc_name", "") or "").strip()
            source_file = str(c.get("file_name", "") or "").strip()
            highlights = c.get("highlights") or []
            if not isinstance(highlights, list):
                highlights = []
            parent_id = c.get("parent_id")
            file_url = c.get("file_url")
            if not file_url and parent_id:
                file_url = f"/api/documents/{parent_id}/file"
            page = None
            if highlights:
                first = highlights[0] if isinstance(highlights[0], dict) else {}
                page = first.get("page")
            refs.append({
                "source_id": i,
                "file_name": doc_name or source_file or "unknown",
                "doc_name": doc_name,
                "source_file_name": source_file,
                "doc_type": c.get("doc_type", ""),
                "jurisdiction": c.get("jurisdiction", ""),
                "chunk_index": c.get("chunk_index", ""),
                "parent_id": parent_id,
                "file_url": file_url,
                "highlights": highlights,
                "page": page,
                "has_pdf_view": bool(file_url and highlights),
                "dense_score": c.get("_dense_score", None),
                "sparse_score": c.get("_sparse_score", None),
            })
        return refs

    # ------------------------------------------------------------------
    # JSON / streaming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_json_object(raw_text: str) -> Optional[str]:
        clean = re.sub(r"```json|```", "", (raw_text or "").strip())
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return match.group(0)
        arr_match = re.search(r"\[\s*(\{.*\})\s*\]", clean, re.DOTALL)
        return arr_match.group(1) if arr_match else None

    @staticmethod
    def _insufficient_payload() -> dict[str, Any]:
        return {
            "answer": "Insufficient evidence from retrieved documents.",
            "insufficient_evidence": True,
            "references": [],
            "sources_used": [],
        }

    @staticmethod
    def _normalize_source_ids(sources_used: Any) -> list[int]:
        out: list[int] = []
        if not isinstance(sources_used, list):
            return out
        for s in sources_used:
            if isinstance(s, bool):
                continue
            if isinstance(s, int):
                out.append(s)
            elif isinstance(s, float) and float(s).is_integer():
                out.append(int(s))
            elif isinstance(s, str):
                for m in re.findall(r"\d+", s):
                    out.append(int(m))
        return sorted(set(out))

    def _parse_answer_json(self, raw_text: str) -> dict[str, Any]:
        json_blob = self._extract_first_json_object(raw_text or "")
        if not json_blob:
            return self._insufficient_payload()
        try:
            data = json.loads(json_blob)
            if not isinstance(data, dict):
                return self._insufficient_payload()
            answer_text = str(data.get("answer", "")).strip()
            if not answer_text:
                return self._insufficient_payload()
            return {
                "answer": answer_text,
                "insufficient_evidence": bool(data.get("insufficient_evidence", False)),
                "references": [],
                "sources_used": self._normalize_source_ids(data.get("sources_used", [])),
            }
        except json.JSONDecodeError:
            return self._insufficient_payload()

    @staticmethod
    def _find_answer_bounds(buffer: str) -> tuple[int, int, bool]:
        match = re.search(r'"answer"\s*:\s*"', buffer)
        if not match:
            return -1, 0, False
        start = match.end()
        i = start
        while i < len(buffer):
            if buffer[i] == '\\' and i + 1 < len(buffer):
                i += 2
                continue
            if buffer[i] == '"':
                return start, i, True
            i += 1
        safe = i
        if safe > start and buffer[safe - 1] == '\\':
            safe -= 1
        return start, safe, False

    @staticmethod
    def _unescape_json_fragment(text: str) -> str:
        result: list[str] = []
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                nxt = text[i + 1]
                result.append({'n': '\n', 't': '\t', '"': '"', '\\': '\\', '/': '/'}.get(nxt, text[i:i + 2]))
                i += 2
                continue
            result.append(text[i])
            i += 1
        return ''.join(result)

    def _stream_generate_iter(self, prompt: str) -> Iterator[str]:
        """Yield answer-text fragments as Gemini streams JSON; sets _last_stream_full_text."""
        logger = logging.getLogger()
        orig = logger.level
        logger.setLevel(logging.CRITICAL)
        buffer = ""
        answer_start = -1
        printed_up_to = 0
        try:
            for chunk in self.gemini_client.models.generate_content_stream(
                model=Config.GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": Config.GEMINI_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            ):
                buffer += chunk.text or ""
                start, safe_end, _ = self._find_answer_bounds(buffer)
                if start == -1:
                    continue
                if answer_start == -1:
                    answer_start = start
                new_end = safe_end - answer_start
                if new_end > printed_up_to:
                    new_raw = buffer[answer_start + printed_up_to: safe_end]
                    yield self._unescape_json_fragment(new_raw)
                    printed_up_to = new_end
        finally:
            logger.setLevel(orig)
        self._last_stream_full_text = buffer

    def _stream_generate(self, prompt: str) -> str:
        header_printed = False
        printed_any = False
        for fragment in self._stream_generate_iter(prompt):
            if not header_printed:
                sys.stdout.write("\nAnswer:\n")
                header_printed = True
            sys.stdout.write(fragment)
            sys.stdout.flush()
            printed_any = True
        if printed_any:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif not header_printed:
            sys.stdout.write("\nAnswer:\n")
            sys.stdout.flush()
        return getattr(self, "_last_stream_full_text", "")

    # ------------------------------------------------------------------
    # Core vector pipeline (factored out for Self-RAG reuse)
    # ------------------------------------------------------------------

    def _run_vector_pipeline(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        intent: QueryIntent,
        top_k: int,
        stream: bool,
        allow_retry: bool = True,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """
        include_debug: when True, attaches an "_eval_debug" block with the
        jurisdiction tag of every retrieved chunk (pre- and post-MMR). This is
        used exclusively by the offline evaluation harness (evaluation/) to
        verify that no cross-jurisdiction content ever reaches the LLM context
        window for a given user. It is never set by the live chat path
        (answer_with_events), so normal traffic is completely unaffected.
        """
        jurisdiction = jurisdiction_ctx.canonical_name
        jkey = jurisdiction_ctx.registry_key
        search_top_k = max(top_k * 3, 30)

        # ── Hybrid search ──
        all_ranked_lists: list[list[str]] = []
        all_score_maps: list[dict[str, dict[str, Any]]] = []

        def _run_collection(dt: str) -> None:
            # dt="any" → no doc_type filter (cases+legislation share one collection)
            collection = self._collection_for(jkey, "case")
            effective_dt: Optional[str] = None if dt == "any" else dt
            q_filter = self._build_filter(effective_dt, jurisdiction, query, jkey)
            ranked_lists, score_map = self._hybrid_search_collection(
                collection=collection,
                sub_queries=intent.sub_queries,
                query_filter=q_filter,
                top_k_per_query=search_top_k,
            )
            all_ranked_lists.extend(ranked_lists)
            all_score_maps.append(score_map)
            logging.info(
                "Collection '%s': %d ranked lists, %d candidates",
                collection, len(ranked_lists), len(score_map),
            )

        if intent.needs_cases and intent.needs_legislation:
            # Both doc types live in one collection — one search pass is enough
            _run_collection("any")
        elif intent.needs_cases:
            _run_collection("case")
        elif intent.needs_legislation:
            _run_collection("legislation")

        # ── RRF Fusion ──
        fused_chunks = self._fuse_results(all_ranked_lists, all_score_maps, top_n=100)
        all_chunks = fused_chunks

        if not all_chunks:
            empty_result: dict[str, Any] = {
                "answer": "Insufficient evidence from retrieved documents.",
                "insufficient_evidence": True,
                "references": [],
                "retrieved_chunks": 0,
                "top_chunks": 0,
            }
            if include_debug:
                empty_result["_eval_debug"] = self._build_eval_debug(
                    jurisdiction=jurisdiction,
                    all_chunks=[],
                    top_chunks=[],
                    context="",
                    self_rag_retry_used=False,
                )
            return empty_result

        # ── MMR re-rank ──
        final_top_k = getattr(Config, "ANSWER_TOP_K", 10)
        mmr_pool = min(len(all_chunks), final_top_k * 5)
        top_chunks = self._mmr_rerank(query, all_chunks[:mmr_pool], top_k=final_top_k)

        # ── Gemini Call #2 ──
        context = self._build_grounded_context(top_chunks)
        prompt = self._build_answer_prompt(
            query=query,
            context=context,
            document_grounded=intent.is_document_grounded,
            intent=intent,
            jurisdiction_ctx=jurisdiction_ctx,
        )

        if stream:
            full_text = self._stream_generate(prompt)
        else:
            resp = self.gemini_client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": Config.GEMINI_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
            full_text = resp.text or ""

        parsed = self._parse_answer_json(full_text)

        self_rag_retry_used = False
        # ── Self-RAG: retry on insufficient evidence ──
        if (
            allow_retry
            and parsed.get("insufficient_evidence")
            and len(intent.sub_queries) > 0
        ):
            retry_result = self._self_rag_retry(
                query, jurisdiction_ctx, intent, stream, attempt=1, include_debug=include_debug
            )
            if retry_result and not retry_result.get("insufficient_evidence"):
                parsed = retry_result
                self_rag_retry_used = True

        # ── Source clamping for grounded queries ──
        source_ids = set(parsed.get("sources_used", []))
        if not source_ids:
            source_ids = {
                int(m)
                for m in re.findall(
                    r"(?:source|src)\s*#?\s*(\d+)", parsed.get("answer", ""), re.IGNORECASE
                )
            }
            if source_ids:
                parsed["sources_used"] = sorted(source_ids)

        if intent.is_document_grounded and not parsed.get("insufficient_evidence"):
            ans = (parsed.get("answer") or "").strip().lower()
            if ans and "insufficient evidence" not in ans and not source_ids:
                logging.warning("Grounded answer cited no sources — clamping.")
                parsed["answer"] = "Insufficient evidence from retrieved documents."
                parsed["insufficient_evidence"] = True
                parsed["sources_used"] = []
                source_ids = set()

        # ── Build references ──
        all_references = self._build_references(top_chunks)
        if source_ids:
            parsed["references"] = [
                r for r in all_references if int(r.get("source_id", 0)) in source_ids
            ]
        else:
            parsed["references"] = all_references

        parsed["retrieved_chunks"] = len(fused_chunks)
        parsed["top_chunks"] = len(top_chunks)

        # Attach jurisdiction-leakage debug info for the evaluation harness only.
        # Skipped if a Self-RAG retry already attached its own (more accurate)
        # debug block for the chunks that actually produced the returned answer.
        if include_debug and "_eval_debug" not in parsed:
            parsed["_eval_debug"] = self._build_eval_debug(
                jurisdiction=jurisdiction,
                all_chunks=all_chunks,
                top_chunks=top_chunks,
                context=context,
                self_rag_retry_used=self_rag_retry_used,
            )
        elif include_debug and parsed.get("_eval_debug") is not None:
            parsed["_eval_debug"]["self_rag_retry_used"] = self_rag_retry_used

        return parsed

    def _postprocess_parsed(
        self,
        parsed: dict[str, Any],
        intent: QueryIntent,
        top_chunks: list[dict[str, Any]],
        fused_len: int,
    ) -> dict[str, Any]:
        """Source clamping, reference filtering, and chunk counts."""
        source_ids = set(parsed.get("sources_used", []))
        if not source_ids:
            source_ids = {
                int(m)
                for m in re.findall(
                    r"(?:source|src)\s*#?\s*(\d+)", parsed.get("answer", ""), re.IGNORECASE
                )
            }
            if source_ids:
                parsed["sources_used"] = sorted(source_ids)

        if intent.is_document_grounded and not parsed.get("insufficient_evidence"):
            ans = (parsed.get("answer") or "").strip().lower()
            if ans and "insufficient evidence" not in ans and not source_ids:
                logging.warning("Grounded answer cited no sources — clamping.")
                parsed["answer"] = "Insufficient evidence from retrieved documents."
                parsed["insufficient_evidence"] = True
                parsed["sources_used"] = []
                source_ids = set()

        all_references = self._build_references(top_chunks)
        if source_ids:
            parsed["references"] = [
                r for r in all_references if int(r.get("source_id", 0)) in source_ids
            ]
        else:
            parsed["references"] = all_references

        parsed["retrieved_chunks"] = fused_len
        parsed["top_chunks"] = len(top_chunks)
        return parsed

    def _run_vector_pipeline_events(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        intent: QueryIntent,
        top_k: int,
        allow_retry: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Generator variant: yields thinking/token events; returns final parsed dict."""
        jurisdiction = jurisdiction_ctx.canonical_name
        jkey = jurisdiction_ctx.registry_key
        search_top_k = max(top_k * 3, 30)

        all_ranked_lists: list[list[str]] = []
        all_score_maps: list[dict[str, dict[str, Any]]] = []

        def _run_collection(dt: str) -> None:
            # dt="any" → no doc_type filter (cases+legislation share one collection)
            collection = self._collection_for(jkey, "case")
            effective_dt: Optional[str] = None if dt == "any" else dt
            q_filter = self._build_filter(effective_dt, jurisdiction, query, jkey)
            ranked_lists, score_map = self._hybrid_search_collection(
                collection=collection,
                sub_queries=intent.sub_queries,
                query_filter=q_filter,
                top_k_per_query=search_top_k,
            )
            all_ranked_lists.extend(ranked_lists)
            all_score_maps.append(score_map)

        if intent.needs_cases and intent.needs_legislation:
            yield {"type": "thinking", "data": {"step": "Searching case law & legislation (dense + sparse)..."}}
            _run_collection("any")
        elif intent.needs_cases:
            yield {"type": "thinking", "data": {"step": "Searching case law (dense + sparse)..."}}
            _run_collection("case")
        elif intent.needs_legislation:
            yield {"type": "thinking", "data": {"step": "Searching legislation (dense + sparse)..."}}
            _run_collection("legislation")

        yield {"type": "thinking", "data": {"step": "Fusing and re-ranking results..."}}
        fused_chunks = self._fuse_results(all_ranked_lists, all_score_maps, top_n=100)
        all_chunks = fused_chunks

        if not all_chunks:
            return {
                "answer": "Insufficient evidence from retrieved documents.",
                "insufficient_evidence": True,
                "references": [],
                "retrieved_chunks": 0,
                "top_chunks": 0,
            }

        final_top_k = getattr(Config, "ANSWER_TOP_K", 10)
        mmr_pool = min(len(all_chunks), final_top_k * 5)
        top_chunks = self._mmr_rerank(query, all_chunks[:mmr_pool], top_k=final_top_k)

        context = self._build_grounded_context(top_chunks)
        prompt = self._build_answer_prompt(
            query=query,
            context=context,
            document_grounded=intent.is_document_grounded,
            intent=intent,
            jurisdiction_ctx=jurisdiction_ctx,
        )

        yield {"type": "thinking", "data": {"step": "Generating grounded answer..."}}
        for fragment in self._stream_generate_iter(prompt):
            yield {"type": "token", "data": {"text": fragment}}

        full_text = getattr(self, "_last_stream_full_text", "")
        parsed = self._parse_answer_json(full_text)

        if (
            allow_retry
            and parsed.get("insufficient_evidence")
            and len(intent.sub_queries) > 0
        ):
            yield {"type": "thinking", "data": {"step": "Broadening search and retrying..."}}
            broadened = QueryIntent(
                intent_type=intent.intent_type,
                is_document_grounded=False,
                needs_cases=True,
                needs_legislation=True,
                route_label="ambiguous",
                sub_queries=[query],
                rationale="self-rag-retry",
            )
            retry_parsed = yield from self._run_vector_pipeline_events(
                query=query,
                jurisdiction_ctx=jurisdiction_ctx,
                intent=broadened,
                top_k=getattr(Config, "RETRIEVAL_TOP_K", 10) * 2,
                allow_retry=False,
            )
            if retry_parsed and not retry_parsed.get("insufficient_evidence"):
                return retry_parsed

        return self._postprocess_parsed(parsed, intent, top_chunks, len(fused_chunks))

    def answer_with_events(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-friendly events: thinking, token, done, error."""
        pipeline_start = time.perf_counter()
        jurisdiction = jurisdiction_ctx.canonical_name

        try:
            yield {"type": "thinking", "data": {"step": "Analyzing your query..."}}

            if MetadataDB.is_structured_query(query):
                yield {"type": "thinking", "data": {"step": "Running metadata search..."}}
                struct_result = self.metadata_db.structured_query(query, jurisdiction)
                if struct_result["answered"]:
                    answer_text = self._format_structured_result(struct_result, jurisdiction)
                    yield {"type": "token", "data": {"text": answer_text}}
                    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
                    yield {
                        "type": "done",
                        "data": {
                            "answer": answer_text,
                            "insufficient_evidence": False,
                            "references": [],
                            "retrieved_chunks": struct_result["count"],
                            "top_chunks": struct_result["count"],
                            "document_grounded": True,
                            "intent_type": "structured",
                            "retrieval_route": "structured_sql",
                            "sub_queries_used": [query],
                            "retrieval_latency_ms": round(elapsed_ms, 2),
                        },
                    }
                    return

            yield {"type": "thinking", "data": {"step": "Classifying intent and expanding query..."}}
            intent = self.classify_and_expand(query, jurisdiction_ctx)

            if doc_type is not None:
                canonical = self._canonical_doc_type(doc_type)
                intent.needs_cases = canonical == "Case"
                intent.needs_legislation = canonical == "Legislation"
                intent.route_label = "manual"

            if self.should_skip_retrieval(intent):
                yield {
                    "type": "thinking",
                    "data": {"step": "Answering from general legal knowledge (no corpus search)..."},
                }
                parsed = yield from self._direct_answer_events(
                    query, jurisdiction_ctx, intent
                )
                elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
                yield {
                    "type": "done",
                    "data": {
                        **parsed,
                        "document_grounded": False,
                        "intent_type": intent.intent_type,
                        "retrieval_route": "direct_llm",
                        "sub_queries_used": [query],
                        "retrieval_latency_ms": round(elapsed_ms, 2),
                    },
                }
                return

            per_query_k = top_k or self._resolve_top_k(
                intent.is_document_grounded, intent.route_label
            )

            yield {
                "type": "thinking",
                "data": {
                    "step": (
                        f"Intent: {intent.intent_type} — "
                        f"searching {jurisdiction} corpus..."
                    ),
                },
            }

            parsed = yield from self._run_vector_pipeline_events(
                query=query,
                jurisdiction_ctx=jurisdiction_ctx,
                intent=intent,
                top_k=per_query_k,
                allow_retry=True,
            )

            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
            parsed.update({
                "document_grounded": intent.is_document_grounded,
                "intent_type": intent.intent_type,
                "retrieval_route": intent.route_label,
                "sub_queries_used": intent.sub_queries,
                "retrieval_latency_ms": round(elapsed_ms, 2),
            })

            yield {"type": "done", "data": parsed}

        except Exception as exc:
            logging.exception("Pipeline error: %s", exc)
            yield {"type": "error", "data": {"message": str(exc)}}

    # ------------------------------------------------------------------
    # Top-level resolve helpers
    # ------------------------------------------------------------------

    def _resolve_top_k(self, document_grounded: bool, route: str) -> int:
        if route == "ambiguous":
            return getattr(Config, "RETRIEVAL_TOP_K_AMBIGUOUS", 20)
        if document_grounded:
            return getattr(Config, "RETRIEVAL_TOP_K_DOCUMENT", 15)
        return getattr(Config, "RETRIEVAL_TOP_K", 10)

    # ------------------------------------------------------------------
    # Public API — answer()
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        jurisdiction_ctx: JurisdictionContext,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
        stream: bool = False,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """
        Full production pipeline.

        Args:
            query:            The user's legal question.
            jurisdiction_ctx: Server-enforced JurisdictionContext (from middleware).
            top_k:            Override per-collection retrieval budget.
            doc_type:         Force "Case" or "Legislation" retrieval only.
            stream:           Stream answer tokens to stdout.
            include_debug:    Attach "_eval_debug" jurisdiction-leakage info.
                              Used by the offline evaluation harness only —
                              leave False (default) for all live traffic.
        """
        pipeline_start = time.perf_counter()

        jurisdiction = jurisdiction_ctx.canonical_name
        jkey = jurisdiction_ctx.registry_key

        logging.info(
            "Pipeline start: user=%s jurisdiction=%s query=%.80s",
            jurisdiction_ctx.user_id, jurisdiction, query,
        )

        # ── Structured query shortcut ──
        if MetadataDB.is_structured_query(query):
            struct_result = self.metadata_db.structured_query(query, jurisdiction)
            if struct_result["answered"]:
                elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
                structured_response: dict[str, Any] = {
                    "answer": self._format_structured_result(struct_result, jurisdiction),
                    "insufficient_evidence": False,
                    "references": [],
                    "retrieved_chunks": struct_result["count"],
                    "top_chunks": struct_result["count"],
                    "document_grounded": True,
                    "intent_type": "structured",
                    "retrieval_route": "structured_sql",
                    "sub_queries_used": [query],
                    "retrieval_latency_ms": round(elapsed_ms, 2),
                    "_streamed": False,
                }
                if include_debug:
                    structured_response["_eval_debug"] = self._build_eval_debug(
                        jurisdiction=jurisdiction,
                        all_chunks=[],
                        top_chunks=[],
                        context=structured_response["answer"],
                        self_rag_retry_used=False,
                    )
                return structured_response

        # ── Step 1: Intent classification ──
        intent = self.classify_and_expand(query, jurisdiction_ctx)

        # Manual doc_type override
        if doc_type is not None:
            canonical = self._canonical_doc_type(doc_type)
            intent.needs_cases = canonical == "Case"
            intent.needs_legislation = canonical == "Legislation"
            intent.route_label = "manual"

        if self.should_skip_retrieval(intent):
            logging.info(
                "Skipping retrieval for generic query (direct LLM): %.80s", query
            )
            parsed = self._answer_without_retrieval(
                query,
                jurisdiction_ctx,
                intent,
                stream=stream,
                include_debug=include_debug,
            )
            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
            parsed.update({
                "document_grounded": False,
                "intent_type": intent.intent_type,
                "retrieval_route": "direct_llm",
                "sub_queries_used": [query],
                "retrieval_latency_ms": round(elapsed_ms, 2),
                "_streamed": stream,
            })
            return parsed

        per_query_k = top_k or self._resolve_top_k(intent.is_document_grounded, intent.route_label)

        logging.info(
            "Intent: type=%s grounded=%s structured=%s route=%s sub_queries=%d k=%d",
            intent.intent_type, intent.is_document_grounded, intent.is_structured,
            intent.route_label, len(intent.sub_queries), per_query_k,
        )

        # ── Steps 2–8: Vector pipeline ──
        parsed = self._run_vector_pipeline(
            query=query,
            jurisdiction_ctx=jurisdiction_ctx,
            intent=intent,
            top_k=per_query_k,
            stream=stream,
            allow_retry=True,
            include_debug=include_debug,
        )

        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        parsed.update({
            "document_grounded": intent.is_document_grounded,
            "intent_type": intent.intent_type,
            "retrieval_route": intent.route_label,
            "sub_queries_used": intent.sub_queries,
            "retrieval_latency_ms": round(elapsed_ms, 2),
            "_streamed": stream,
        })

        logging.info(
            "Pipeline complete: route=%s intent=%s chunks=%d latency=%.2fms",
            intent.route_label, intent.intent_type,
            parsed.get("retrieved_chunks", 0), elapsed_ms,
        )
        return parsed

    # ------------------------------------------------------------------
    # Format structured SQL results as natural-language answer
    # ------------------------------------------------------------------

    @staticmethod
    def _format_structured_result(struct_result: dict, jurisdiction: str) -> str:
        rtype = struct_result.get("result_type", "")
        results = struct_result.get("results", [])
        count = struct_result.get("count", 0)

        if rtype == "cases":
            lines = [f"Found {count} case(s) in {jurisdiction}:\n"]
            for r in results[:20]:
                lines.append(
                    f"- {r.case_name} ({r.year}) | {r.court} | Citation: {r.citation or 'N/A'}"
                )
            return "\n".join(lines)
        elif rtype == "legislation":
            lines = [f"Found {count} legislation record(s) in {jurisdiction}:\n"]
            for r in results[:20]:
                eff = r.effective_date.isoformat() if r.effective_date else "Unknown"
                lines.append(f"- {r.title} | Act No.: {r.act_number} | Effective: {eff}")
            return "\n".join(lines)
        return f"Found {count} result(s) in {jurisdiction}."


# ---------------------------------------------------------------------------
# CLI entry-point (uses dev_context — not for production)
# ---------------------------------------------------------------------------

def main() -> None:
    from middleware import dev_context

    print(
        "Kanooni Jawab Multi-Jurisdiction CLI\n"
        "Pipeline: Jurisdiction -> Intent -> SQL/Vector -> RRF -> Self-RAG -> Answer\n"
    )

    supported = [
        "Hong Kong", "Pakistan", "United Kingdom",
        "Canada", "United States", "Australia", "India",
    ]
    print("Supported jurisdictions:", ", ".join(supported))
    jname = input("Your jurisdiction: ").strip()

    try:
        ctx = dev_context(jname)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\nInitialising retriever (models + DB + Qdrant). This may take 1-2 minutes on first run...")
    try:
        retriever = LegalRetriever()
    except Exception as exc:
        logging.exception("Retriever initialisation failed: %s", exc)
        print(f"Error: {exc}")
        return
    print(f"\nContext: {ctx.canonical_name} | user={ctx.user_id}\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        try:
            result = retriever.answer(query=query, jurisdiction_ctx=ctx, stream=True)

            if not result.get("_streamed"):
                print("\nAnswer:")
                print(result.get("answer", ""))

            print(f"\nIntent:            {result.get('intent_type', '')}")
            print(f"Route:             {result.get('retrieval_route', '')}")
            print(f"Grounded:          {result.get('document_grounded', False)}")
            print(f"Sub-queries:       {len(result.get('sub_queries_used', []))}")
            print(f"Chunks (RRF pool): {result.get('retrieved_chunks', 0)}")
            print(f"Top-K to Gemini:   {result.get('top_chunks', 0)}")
            print(f"Latency (ms):      {result.get('retrieval_latency_ms', 0)}")

            refs = result.get("references", [])
            if refs:
                print("\nReferences:")
                for ref in refs:
                    print(
                        f"  SOURCE {ref.get('source_id')} | {ref.get('doc_type')} | "
                        f"{ref.get('file_name')}"
                    )

        except JurisdictionAccessError as e:
            print(f"Access denied: {e}")
        except Exception as exc:
            logging.exception("Pipeline error: %s", exc)
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()