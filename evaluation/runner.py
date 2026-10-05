from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from config import Config
from evaluation.judge import GeminiJudge
from evaluation.loader import GoldenItem, GoldenSet, load_golden_sets
from evaluation.metrics import ItemScores, JurisdictionMatrixRow, build_matrix_rows, score_item
from middleware import dev_context

log = logging.getLogger(__name__)

_run_lock = threading.Lock()
_latest_run: dict[str, Any] | None = None
_run_in_progress = False


def get_latest_run() -> dict[str, Any] | None:
    with _run_lock:
        return _latest_run


def is_run_in_progress() -> bool:
    with _run_lock:
        return _run_in_progress


def _set_run_state(run: dict[str, Any] | None, in_progress: bool) -> None:
    global _latest_run, _run_in_progress
    with _run_lock:
        _latest_run = run
        _run_in_progress = in_progress


def _evaluate_item(
    retriever: Any,
    judge: GeminiJudge | None,
    item: GoldenItem,
) -> tuple[ItemScores, dict[str, Any]]:
    ctx = dev_context(item.jurisdiction, user_id=f"eval_{item.id}")
    doc_type = item.doc_type

    pipeline_result = retriever.answer(
        query=item.query,
        jurisdiction_ctx=ctx,
        doc_type=doc_type,
        stream=False,
        include_debug=True,
    )

    debug = pipeline_result.get("_eval_debug") or {}
    judge_scores = None
    if judge and not pipeline_result.get("insufficient_evidence"):
        judge_scores = judge.score(
            query=item.query,
            context=str(debug.get("retrieved_context") or ""),
            answer=str(pipeline_result.get("answer") or ""),
        )
    elif judge:
        judge_scores = judge.score(
            query=item.query,
            context=str(debug.get("retrieved_context") or ""),
            answer=str(pipeline_result.get("answer") or ""),
        )

    item_score = score_item(
        item_id=item.id,
        query=item.query,
        jurisdiction=item.jurisdiction,
        expected_docs=item.expected_docs,
        must_cite=item.must_cite,
        pipeline_result=pipeline_result,
        judge_scores=judge_scores,
    )
    return item_score, pipeline_result


def run_evaluation(
    retriever: Any,
    *,
    jurisdiction_keys: list[str] | None = None,
    skip_judge: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = str(uuid.uuid4())
    golden_sets = load_golden_sets(jurisdiction_keys)
    if not golden_sets:
        raise FileNotFoundError(
            f"No golden sets found in {Config.EVAL_GOLDEN_DIR}. "
            "Add JSON files under retrieval/evaluation/golden/."
        )

    all_items: list[tuple[GoldenItem, GoldenSet]] = []
    for golden in golden_sets:
        for item in golden.items:
            if not item.id:
                item.id = f"{golden.registry_key}_{len(all_items) + 1}"
            all_items.append((item, golden))

    judge = None if skip_judge else GeminiJudge()
    item_scores: list[ItemScores] = []
    raw_results: list[dict[str, Any]] = []
    errors: list[str] = []

    def _notify(msg: str) -> None:
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    concurrency = max(1, Config.EVAL_CONCURRENCY)
    _notify(f"Evaluating {len(all_items)} item(s) across {len(golden_sets)} jurisdiction(s)...")

    if concurrency == 1:
        for index, (item, golden) in enumerate(all_items, start=1):
            _notify(f"[{index}/{len(all_items)}] {golden.registry_key}: {item.query[:80]}")
            try:
                scored, pipeline = _evaluate_item(retriever, judge, item)
                item_scores.append(scored)
                raw_results.append(
                    {
                        "item_id": item.id,
                        "jurisdiction": item.jurisdiction,
                        "query": item.query,
                        "scores": asdict(scored),
                        "answer": pipeline.get("answer"),
                        "references": pipeline.get("references"),
                        "intent_type": pipeline.get("intent_type"),
                        "retrieval_route": pipeline.get("retrieval_route"),
                    }
                )
            except Exception as exc:
                log.exception("Eval item failed: %s", item.id)
                errors.append(f"{item.id}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_evaluate_item, retriever, judge, item): (item, golden)
                for item, golden in all_items
            }
            for index, future in enumerate(as_completed(futures), start=1):
                item, golden = futures[future]
                _notify(f"[{index}/{len(all_items)}] completed: {item.id}")
                try:
                    scored, pipeline = future.result()
                    item_scores.append(scored)
                    raw_results.append(
                        {
                            "item_id": item.id,
                            "jurisdiction": item.jurisdiction,
                            "query": item.query,
                            "scores": asdict(scored),
                            "answer": pipeline.get("answer"),
                            "references": pipeline.get("references"),
                            "intent_type": pipeline.get("intent_type"),
                            "retrieval_route": pipeline.get("retrieval_route"),
                        }
                    )
                except Exception as exc:
                    log.exception("Eval item failed: %s", item.id)
                    errors.append(f"{item.id}: {exc}")

    matrix = build_matrix_rows(item_scores, golden_sets)
    finished = datetime.now(timezone.utc)
    report = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "golden_dir": Config.EVAL_GOLDEN_DIR,
        "jurisdiction_keys": jurisdiction_keys,
        "skip_judge": skip_judge,
        "item_count": len(all_items),
        "errors": errors,
        "items": raw_results,
        "matrix": [asdict(row) for row in matrix],
    }
    return report


def run_evaluation_background(
    retriever: Any,
    *,
    jurisdiction_keys: list[str] | None = None,
    skip_judge: bool = False,
) -> str:
    if is_run_in_progress():
        raise RuntimeError("An evaluation run is already in progress")

    run_id = str(uuid.uuid4())

    def _worker() -> None:
        _set_run_state({"run_id": run_id, "status": "running", "started_at": time.time()}, True)
        try:
            report = run_evaluation(
                retriever,
                jurisdiction_keys=jurisdiction_keys,
                skip_judge=skip_judge,
            )
            path = save_report(report)
            report["status"] = "completed"
            report["report_path"] = str(path)
            _set_run_state(report, False)
        except Exception as exc:
            log.exception("Background evaluation failed")
            _set_run_state(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(exc),
                },
                False,
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id


def save_report(report: dict[str, Any]) -> Path:
    out_dir = Path(Config.EVAL_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"eval_{stamp}_{report.get('run_id', 'run')[:8]}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    log.info("Evaluation report saved: %s", path)
    return path
