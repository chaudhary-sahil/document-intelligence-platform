import unittest
from unittest.mock import patch

from backend import evaluation
from backend.services import embedding_service, rag_service


class NormalizeMarkdownTests(unittest.TestCase):
    def test_inline_bullets_become_lines(self):
        out = rag_service.normalize_markdown("Dates: • one [1] • two [2] • three [3]")
        self.assertEqual(out.split("\n")[1:], ["- one [1]", "- two [2]", "- three [3]"])

    def test_leading_bullet_and_blank_lines(self):
        self.assertEqual(rag_service.normalize_markdown("• a\n\n\n\n• b"), "- a\n\n- b")

    def test_plain_text_untouched(self):
        self.assertEqual(rag_service.normalize_markdown("30 days [1]"), "30 days [1]")


class PromptTests(unittest.TestCase):
    def test_grounding_rules_present_in_all_prompts(self):
        for prompt in (rag_service.QA_SYSTEM_PROMPT, rag_service.INSIGHT_SYSTEM_PROMPT):
            self.assertIn("either party", prompt)
            self.assertIn("Never invent", prompt)
        self.assertIn(rag_service.NOT_FOUND, rag_service.QA_SYSTEM_PROMPT)


class EmbeddingBackendTests(unittest.TestCase):
    def setUp(self):
        embedding_service.backend_name.cache_clear()
        self.addCleanup(embedding_service.backend_name.cache_clear)

    def test_auto_uses_onnx_without_sentence_transformers(self):
        with patch.object(embedding_service, "EMBEDDING_BACKEND", "auto"), \
             patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(embedding_service.backend_name(), "onnx")

    def test_auto_prefers_sentence_transformers_when_installed(self):
        with patch.object(embedding_service, "EMBEDDING_BACKEND", "auto"), \
             patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(embedding_service.backend_name(), "sentence-transformers")

    def test_explicit_choice_wins(self):
        with patch.object(embedding_service, "EMBEDDING_BACKEND", "onnx"):
            self.assertEqual(embedding_service.backend_name(), "onnx")


def hit(page, text, score):
    return {"page": page, "text": text, "score": score}


class EvaluationTests(unittest.TestCase):
    ITEM = {"question": "q", "page": 3, "fact": "30 days"}

    def test_hit_requires_right_page_and_fact(self):
        hits = [hit(2, "30 days here", 0.6), hit(3, "either party gives 30 days notice", 0.5)]
        r = evaluation.score_retrieval(self.ITEM, hits, 0.2)
        self.assertTrue(r["ok"])
        self.assertEqual(r["rank"], 2)

    def test_miss_when_fact_absent(self):
        r = evaluation.score_retrieval(self.ITEM, [hit(3, "nothing relevant", 0.5)], 0.2)
        self.assertFalse(r["ok"])
        self.assertIsNone(r["rank"])

    def test_fact_matching_ignores_commas_and_case(self):
        self.assertTrue(evaluation.contains_fact("fee of 8,000 USD", "8000"))
        self.assertTrue(evaluation.contains_fact("NEW YORK", "new york"))

    def test_abstention_scored_on_threshold(self):
        item = {"question": "cake?", "answerable": False}
        self.assertTrue(evaluation.score_retrieval(item, [hit(1, "x", 0.05)], 0.2)["ok"])
        self.assertFalse(evaluation.score_retrieval(item, [hit(1, "x", 0.45)], 0.2)["ok"])
        self.assertTrue(evaluation.score_retrieval(item, [], 0.2)["ok"])

    def test_summary(self):
        results = [
            {"answerable": True, "ok": True, "rank": 1},
            {"answerable": True, "ok": True, "rank": 2},
            {"answerable": True, "ok": False, "rank": None},
            {"answerable": False, "ok": True, "rank": None},
        ]
        s = evaluation.summarize(results)
        self.assertEqual((s["retrieval_hits"], s["answerable_total"]), (2, 3))
        self.assertAlmostEqual(s["mrr"], (1 + 0.5) / 3)
        self.assertEqual((s["abstain_correct"], s["abstain_total"]), (1, 1))


if __name__ == "__main__":
    unittest.main()


class CitationGuardTests(unittest.TestCase):
    def test_out_of_range_citations_removed(self):
        out = rag_service.clean_citations("Fee is 8,000 USD [2] and law is NY [8].", 3)
        self.assertEqual(out, "Fee is 8,000 USD [2] and law is NY.")

    def test_valid_citations_kept(self):
        self.assertEqual(rag_service.clean_citations("ok [1][3]", 3), "ok [1][3]")

    def test_citation_only_reply_has_no_content(self):
        self.assertFalse(rag_service.has_content("[1]"))
        self.assertFalse(rag_service.has_content(" [2][3]. "))
        self.assertTrue(rag_service.has_content("[2] 5 years"))

    def test_leaked_instruction_line_dropped(self):
        text = "- What it is, exact value, excerpt number\n- Notice: 60 days [1]"
        self.assertEqual(rag_service.drop_leaked_lines(text), "- Notice: 60 days [1]")


class RetryTests(unittest.TestCase):
    HITS = [{"text": "t", "source": "a.pdf", "page": 1, "chunk_index": 0, "score": 0.6}]

    def test_citation_only_reply_is_retried(self):
        with patch.object(rag_service, "retrieve", return_value=self.HITS), \
             patch("backend.services.llm_service.generate", side_effect=["[1]", "30 days [1]"]) as llm:
            result = rag_service.answer_question("notice?")
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result["answer"], "30 days [1]")
        self.assertIn("no actual answer", llm.call_args_list[1].args[1])

    def test_two_empty_replies_give_clear_note(self):
        with patch.object(rag_service, "retrieve", return_value=self.HITS), \
             patch("backend.services.llm_service.generate", side_effect=["[1]", "[1]"]):
            result = rag_service.answer_question("notice?")
        self.assertEqual(result["answer"], rag_service.EMPTY_NOTE)
        self.assertEqual(len(result["sources"]), 1)

    def test_invented_excerpt_numbers_are_stripped_from_insights(self):
        hits = self.HITS * 3
        with patch.object(rag_service, "retrieve", return_value=hits), \
             patch("backend.services.llm_service.generate", return_value="- Law: New York [8]\n- Fee: 8,000 [2]"):
            result = rag_service.generate_insight("key_dates", "a.pdf")
        self.assertNotIn("[8]", result["answer"])
        self.assertIn("[2]", result["answer"])

    def test_prompts_separate_clause_numbers_from_excerpt_numbers(self):
        self.assertIn("NOT excerpt numbers", rag_service.QA_SYSTEM_PROMPT)
        self.assertIn("only of citations", rag_service.QA_SYSTEM_PROMPT.replace("made only of citations", "only of citations"))
