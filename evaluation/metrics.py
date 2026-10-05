from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from config import Config
from jurisdiction_registry import normalize_jurisdiction


def normalize_doc_name(name: str) -> str:
    cleaned = re.sub(r"[_\-\.]+", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def doc_name_matches(expected: str, retrieved: str) -> bool:
    expected_norm = normalize_doc_name(expected)
    retrieved_norm = normalize_doc_name(retrieved)
    if not expected_norm or not retrieved_norm:
        return False
    return expected_norm in retrieved_norm or retrieved_norm in expected_norm


def context_precision_at_k(
    expected_docs: list[str],
    retrieved_doc_names: list[str],
    k: int | None = None,
) -> float:
    """Fraction of top-K retrieved docs that match any expected document."""
    if not expected_docs:
        return 1.0
    names = retrieved_doc_names[:k] if k else retrieved_doc_names
    if not names:
        return 0.0
    hits = sum(
        1
        for name in names
        if any(doc_name_matches(expected, name) for expected in expected_docs)
    )
    return hits / len(names)


def hit_at_k(
    expected_docs: list[str],
    retrieved_doc_names: list[str],
    k: int = 10,
) -> float:
    """1.0 if any expected doc appears in top-K, else 0.0."""
    if not expected_docs:
        return 1.0
    top = retrieved_doc_names[:k]
    return (
        1.0
        if any(
            doc_name_matches(expected, name)
            for expected in expected_docs
            for name in top
        )
        else 0.0
    )


def jurisdiction_leak_detected(
    expected_jurisdiction: str,
    chunk_jurisdictions: list[str],
) -> bool:
    expected = normalize_jurisdiction(expected_jurisdiction).lower()
    for jurisdiction in chunk_jurisdictions:
        if not jurisdiction:
            continue
        if normalize_jurisdiction(jurisdiction).lower() != expected:
            return True
    return False


def citation_present(sources_used: list[Any]) -> bool:
    return bool(sources_used)


@dataclass
class ItemScores:
    item_id: str
    query: str
    jurisdiction: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float = 0.0
    hit_at_k: float = 0.0
    jurisdiction_leak: bool = False
    latency_ms: float = 0.0
    insufficient_evidence: bool = False
    self_rag_retry_used: bool = False
    cited_sources: bool = False
    judge_rationale: str = ""
    pass_faithfulness: bool = False
    pass_relevancy: bool = False
    pass_precision: bool = False
    pass_overall: bool = False
    error: str = ""


@dataclass
class JurisdictionMatrixRow:
    jurisdiction: str
    registry_key: str
    item_count: int
    faithfulness_mean: float | None = None
    answer_relevancy_mean: float | None = None
    context_precision_mean: float = 0.0
    hit_at_k_mean: float = 0.0
    jurisdiction_leak_rate: float = 0.0
    latency_ms_p50: float = 0.0
    pass_rate: float = 0.0
    pass_faithfulness: bool = False
    pass_relevancy: bool = False
    pass_precision: bool = False
    pass_overall: bool = False


def score_item(
    *,
    item_id: str,
    query: str,
    jurisdiction: str,
    expected_docs: list[str],
    must_cite: bool,
    pipeline_result: dict[str, Any],
    judge_scores: dict[str, Any] | None = None,
    k: int | None = None,
) -> ItemScores:
    k = k or getattr(Config, "ANSWER_TOP_K", 10)
    debug = pipeline_result.get("_eval_debug") or {}
    top_names = list(debug.get("top_chunk_doc_names") or [])
    top_jurisdictions = list(debug.get("top_chunk_jurisdictions") or [])

    faithfulness = None
    answer_relevancy = None
    rationale = ""
    if judge_scores:
        faithfulness = _as_score(judge_scores.get("faithfulness"))
        answer_relevancy = _as_score(judge_scores.get("answer_relevancy"))
        rationale = str(judge_scores.get("rationale", ""))

    precision = context_precision_at_k(expected_docs, top_names, k=k)
    hit = hit_at_k(expected_docs, top_names, k=k)
    leak = jurisdiction_leak_detected(jurisdiction, top_jurisdictions)
    sources_used = pipeline_result.get("sources_used") or []
    cited = citation_present(sources_used)

    scores = ItemScores(
        item_id=item_id,
        query=query,
        jurisdiction=jurisdiction,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=precision,
        hit_at_k=hit,
        jurisdiction_leak=leak,
        latency_ms=float(pipeline_result.get("retrieval_latency_ms") or 0),
        insufficient_evidence=bool(pipeline_result.get("insufficient_evidence")),
        self_rag_retry_used=bool(debug.get("self_rag_retry_used")),
        cited_sources=cited,
        judge_rationale=rationale,
    )

    if faithfulness is not None:
        scores.pass_faithfulness = faithfulness >= Config.EVAL_PASS_THRESHOLD_FAITHFULNESS
    if answer_relevancy is not None:
        scores.pass_relevancy = answer_relevancy >= Config.EVAL_PASS_THRESHOLD_RELEVANCY
    scores.pass_precision = precision >= Config.EVAL_PASS_THRESHOLD_PRECISION

    overall_checks = [scores.pass_precision, not scores.jurisdiction_leak]
    if faithfulness is not None:
        overall_checks.append(scores.pass_faithfulness)
    if answer_relevancy is not None:
        overall_checks.append(scores.pass_relevancy)
    if must_cite:
        overall_checks.append(cited and not scores.insufficient_evidence)
    scores.pass_overall = all(overall_checks)

    return scores


def _as_score(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return statistics.mean(nums)


def build_matrix_rows(
    item_scores: list[ItemScores],
    golden_sets: list[Any],
) -> list[JurisdictionMatrixRow]:
    key_to_name = {s.registry_key: s.jurisdiction for s in golden_sets}
    by_jurisdiction: dict[str, list[ItemScores]] = {}
    for score in item_scores:
        by_jurisdiction.setdefault(score.jurisdiction, []).append(score)

    rows: list[JurisdictionMatrixRow] = []
    for jurisdiction, scores in sorted(by_jurisdiction.items()):
        registry_key = next(
            (k for k, v in key_to_name.items() if v == jurisdiction),
            jurisdiction.lower().replace(" ", "_"),
        )
        faithfulness_mean = _mean([s.faithfulness for s in scores])
        relevancy_mean = _mean([s.answer_relevancy for s in scores])
        precision_mean = statistics.mean(s.context_precision for s in scores) if scores else 0.0
        hit_mean = statistics.mean(s.hit_at_k for s in scores) if scores else 0.0
        leak_rate = (
            sum(1 for s in scores if s.jurisdiction_leak) / len(scores) if scores else 0.0
        )
        latencies = [s.latency_ms for s in scores]
        latency_p50 = statistics.median(latencies) if latencies else 0.0
        pass_rate = sum(1 for s in scores if s.pass_overall) / len(scores) if scores else 0.0

        row = JurisdictionMatrixRow(
            jurisdiction=jurisdiction,
            registry_key=registry_key,
            item_count=len(scores),
            faithfulness_mean=faithfulness_mean,
            answer_relevancy_mean=relevancy_mean,
            context_precision_mean=precision_mean,
            hit_at_k_mean=hit_mean,
            jurisdiction_leak_rate=leak_rate,
            latency_ms_p50=latency_p50,
            pass_rate=pass_rate,
        )
        row.pass_precision = precision_mean >= Config.EVAL_PASS_THRESHOLD_PRECISION
        overall_checks = [row.pass_precision, leak_rate == 0.0]
        if faithfulness_mean is not None:
            row.pass_faithfulness = faithfulness_mean >= Config.EVAL_PASS_THRESHOLD_FAITHFULNESS
            overall_checks.append(row.pass_faithfulness)
        if relevancy_mean is not None:
            row.pass_relevancy = relevancy_mean >= Config.EVAL_PASS_THRESHOLD_RELEVANCY
            overall_checks.append(row.pass_relevancy)
        row.pass_overall = all(overall_checks)
        rows.append(row)
    return rows
