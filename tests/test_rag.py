"""Tests the RAG logic with the vector store and the LLM replaced by fakes."""
import unittest
from unittest.mock import patch

from backend.services import rag_service


def hit(text, score, page=1, idx=0, source="a.pdf"):
    return {"text": text, "source": source, "page": page, "chunk_index": idx, "score": score}


class RagTests(unittest.TestCase):
    def test_irrelevant_results_skip_the_llm(self):
        with patch.object(rag_service, "retrieve", return_value=[hit("x", 0.05)]), \
             patch("backend.services.llm_service.generate") as llm:
            result = rag_service.answer_question("anything?")
        llm.assert_not_called()
        self.assertEqual(result["answer"], rag_service.NOT_FOUND)
        self.assertEqual(result["sources"], [])

    def test_answer_includes_numbered_context_and_sources(self):
        hits = [hit("Termination needs 30 days notice.", 0.7, page=3),
                hit("Fees are due in 45 days.", 0.4, page=2, idx=1)]
        with patch.object(rag_service, "retrieve", return_value=hits), \
             patch("backend.services.llm_service.generate", return_value="30 days [1]") as llm:
            result = rag_service.answer_question("How do I terminate?")
        prompt = llm.call_args[0][1]
        self.assertIn("[1] (Source: a.pdf, page 3)", prompt)
        self.assertIn("[2] (Source: a.pdf, page 2)", prompt)
        self.assertEqual(result["answer"], "30 days [1]")
        self.assertEqual([s["id"] for s in result["sources"]], [1, 2])

    def test_weak_hits_are_filtered_out(self):
        hits = [hit("good", 0.6), hit("bad", 0.1, idx=1)]
        with patch.object(rag_service, "retrieve", return_value=hits), \
             patch("backend.services.llm_service.generate", return_value="ok"):
            result = rag_service.answer_question("q?")
        self.assertEqual(len(result["sources"]), 1)

    def test_insight_orders_context_by_page(self):
        hits = [hit("late", 0.5, page=9, idx=5), hit("early", 0.4, page=1, idx=0)]
        with patch.object(rag_service, "retrieve", return_value=hits), \
             patch("backend.services.llm_service.generate", return_value="done") as llm:
            result = rag_service.generate_insight("risks", "a.pdf")
        prompt = llm.call_args[0][1]
        self.assertLess(prompt.index("early"), prompt.index("late"))
        self.assertEqual(result["sources"][0]["page"], 1)

    def test_all_insight_kinds_are_defined(self):
        self.assertEqual(set(rag_service.INSIGHTS), {"summary", "risks", "key_dates", "obligations"})


if __name__ == "__main__":
    unittest.main()
