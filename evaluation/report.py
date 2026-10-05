from __future__ import annotations

from typing import Any


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _pass_icon(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_matrix(report: dict[str, Any]) -> None:
    matrix: list[dict[str, Any]] = report.get("matrix") or []
    if not matrix:
        print("No matrix rows to display.")
        return

    headers = [
        "Jurisdiction",
        "Items",
        "Faithfulness",
        "Relevancy",
        "Precision@K",
        "Hit@K",
        "Leak%",
        "Latency p50",
        "Pass rate",
        "Overall",
    ]
    print("\n" + "=" * 110)
    print("Kanooni Jawab Evaluation Matrix")
    print("=" * 110)
    print(
        f"{'Jurisdiction':<18} {'Items':>5} {'Faith':>8} {'Rel':>8} {'Prec':>8} "
        f"{'Hit@K':>8} {'Leak%':>8} {'Lat ms':>8} {'Pass%':>8} {'Overall':>8}"
    )
    print("-" * 110)
    for row in matrix:
        leak_pct = float(row.get("jurisdiction_leak_rate") or 0) * 100
        pass_pct = float(row.get("pass_rate") or 0) * 100
        print(
            f"{row.get('jurisdiction', ''):<18} "
            f"{int(row.get('item_count', 0)):>5} "
            f"{_fmt(row.get('faithfulness_mean')):>8} "
            f"{_fmt(row.get('answer_relevancy_mean')):>8} "
            f"{_fmt(row.get('context_precision_mean')):>8} "
            f"{_fmt(row.get('hit_at_k_mean')):>8} "
            f"{leak_pct:>7.1f}% "
            f"{_fmt(row.get('latency_ms_p50'), 0):>8} "
            f"{pass_pct:>7.1f}% "
            f"{_pass_icon(bool(row.get('pass_overall'))):>8}"
        )
    print("=" * 110)

    errors = report.get("errors") or []
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")


def matrix_summary(report: dict[str, Any]) -> dict[str, Any]:
    matrix = report.get("matrix") or []
    return {
        "run_id": report.get("run_id"),
        "status": report.get("status", "completed"),
        "item_count": report.get("item_count"),
        "duration_seconds": report.get("duration_seconds"),
        "errors": report.get("errors") or [],
        "matrix": matrix,
        "report_path": report.get("report_path"),
    }
