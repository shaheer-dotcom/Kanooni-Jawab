from __future__ import annotations

import argparse
import logging
import sys

from evaluation.report import print_matrix
from evaluation.runner import run_evaluation, save_report
from retriever import LegalRetriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Kanooni Jawab golden-set evaluation")
    parser.add_argument(
        "--jurisdiction",
        action="append",
        dest="jurisdictions",
        help="Registry key(s) to evaluate, e.g. hong_kong pakistan",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Gemini judge (retrieval metrics only)",
    )
    parser.add_argument(
        "--output",
        help="Optional explicit report JSON path (also writes to EVAL_OUTPUT_DIR)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("Initialising retriever...")
    retriever = LegalRetriever()

    report = run_evaluation(
        retriever,
        jurisdiction_keys=args.jurisdictions,
        skip_judge=args.skip_judge,
        progress_callback=lambda msg: print(msg),
    )
    report["status"] = "completed"

    path = save_report(report)
    if args.output:
        import json
        from pathlib import Path

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"Report also written to {out}")

    print_matrix(report)
    print(f"\nReport saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
