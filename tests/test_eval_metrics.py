import unittest

from evaluation.loader import load_golden_file
from evaluation.metrics import (
    ItemScores,
    build_matrix_rows,
    context_precision_at_k,
    doc_name_matches,
    hit_at_k,
    jurisdiction_leak_detected,
    score_item,
)
from pathlib import Path


class MetricsTests(unittest.TestCase):
    def test_doc_name_matches_partial(self) -> None:
        self.assertTrue(doc_name_matches("Weapons Ordinance", "Weapons_Ordinance_english.pdf"))
        self.assertTrue(doc_name_matches("Cap 4", "Cap 4 Consolidated version"))

    def test_context_precision_at_k(self) -> None:
        precision = context_precision_at_k(
            ["Weapons Ordinance"],
            ["Weapons_Ordinance_english.pdf", "Unrelated Act", "Another Doc"],
            k=3,
        )
        self.assertAlmostEqual(precision, 1 / 3)

    def test_hit_at_k(self) -> None:
        self.assertEqual(
            hit_at_k(["Weapons Ordinance"], ["Foo", "Weapons_Ordinance_english.pdf"], k=2),
            1.0,
        )
        self.assertEqual(hit_at_k(["Weapons Ordinance"], ["Foo", "Bar"], k=2), 0.0)

    def test_jurisdiction_leak_detected(self) -> None:
        self.assertFalse(
            jurisdiction_leak_detected("Hong Kong", ["Hong Kong", "Hong Kong"])
        )
        self.assertTrue(
            jurisdiction_leak_detected("Hong Kong", ["Hong Kong", "Pakistan"])
        )

    def test_score_item_overall_pass(self) -> None:
        pipeline = {
            "answer": "The Commissioner may order delivery under section 2.",
            "sources_used": [1],
            "insufficient_evidence": False,
            "retrieval_latency_ms": 1200,
            "_eval_debug": {
                "top_chunk_doc_names": ["Weapons_Ordinance_english.pdf"],
                "top_chunk_jurisdictions": ["Hong Kong"],
                "self_rag_retry_used": False,
            },
        }
        scored = score_item(
            item_id="t1",
            query="Weapons question",
            jurisdiction="Hong Kong",
            expected_docs=["Weapons Ordinance"],
            must_cite=True,
            pipeline_result=pipeline,
            judge_scores={"faithfulness": 0.9, "answer_relevancy": 0.85, "rationale": "ok"},
        )
        self.assertTrue(scored.pass_overall)
        self.assertFalse(scored.jurisdiction_leak)

    def test_build_matrix_rows(self) -> None:
        class _Set:
            registry_key = "hong_kong"
            jurisdiction = "Hong Kong"

        rows = build_matrix_rows(
            [
                ItemScores(
                    item_id="a",
                    query="q",
                    jurisdiction="Hong Kong",
                    faithfulness=0.9,
                    answer_relevancy=0.8,
                    context_precision=0.7,
                    pass_precision=True,
                    pass_faithfulness=True,
                    pass_relevancy=True,
                    pass_overall=True,
                )
            ],
            [_Set()],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].jurisdiction, "Hong Kong")
        self.assertTrue(rows[0].pass_overall)


class LoaderTests(unittest.TestCase):
    def test_load_golden_file(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evaluation" / "golden" / "hong_kong.json"
        golden = load_golden_file(path)
        self.assertEqual(golden.jurisdiction, "Hong Kong")
        self.assertGreaterEqual(len(golden.items), 1)
        self.assertTrue(golden.items[0].query)


if __name__ == "__main__":
    unittest.main()
