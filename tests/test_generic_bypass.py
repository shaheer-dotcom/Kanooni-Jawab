import unittest

from retriever import LegalRetriever, QueryIntent


class GenericBypassTests(unittest.TestCase):
    def test_should_skip_generic(self) -> None:
        intent = QueryIntent(
            intent_type="generic",
            is_document_grounded=False,
            is_structured=False,
            needs_cases=False,
            needs_legislation=False,
            route_label="generic",
            sub_queries=["who is criminal?"],
        )
        self.assertTrue(LegalRetriever.should_skip_retrieval(intent))

    def test_should_not_skip_topic(self) -> None:
        intent = QueryIntent(
            intent_type="topic",
            is_document_grounded=False,
            needs_cases=True,
            needs_legislation=True,
            route_label="hybrid",
        )
        self.assertFalse(LegalRetriever.should_skip_retrieval(intent))

    def test_should_not_skip_document_grounded(self) -> None:
        intent = QueryIntent(
            intent_type="generic",
            is_document_grounded=True,
            needs_cases=False,
            needs_legislation=False,
            route_label="generic",
        )
        self.assertFalse(LegalRetriever.should_skip_retrieval(intent))

    def test_should_not_skip_manual_doc_type(self) -> None:
        intent = QueryIntent(
            intent_type="generic",
            is_document_grounded=False,
            needs_cases=True,
            needs_legislation=False,
            route_label="manual",
        )
        self.assertFalse(LegalRetriever.should_skip_retrieval(intent))

    def test_normalize_generic_routing(self) -> None:
        intent = QueryIntent(
            intent_type="generic",
            is_document_grounded=False,
            needs_cases=True,
            needs_legislation=True,
            route_label="ambiguous",
        )
        normalized = LegalRetriever._normalize_intent_routing(intent)
        self.assertFalse(normalized.needs_cases)
        self.assertFalse(normalized.needs_legislation)
        self.assertEqual(normalized.route_label, "generic")


if __name__ == "__main__":
    unittest.main()
