import unittest

from backend.services.chunk_service import chunk_pages, split_text

SAMPLE = (
    "The Provider shall deliver the report each quarter. "
    "The Client must pay within forty five days. "
    "Either party may terminate with thirty days notice. "
) * 40


class ChunkingTests(unittest.TestCase):
    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(split_text("   \n  "), [])

    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("Hello world.", 800, 150), ["Hello world."])

    def test_chunks_respect_size_limit(self):
        for chunk in split_text(SAMPLE, 300, 50):
            self.assertLessEqual(len(chunk), 300)

    def test_chunks_do_not_start_or_end_mid_word(self):
        words = set(SAMPLE.split())
        for chunk in split_text(SAMPLE, 300, 50):
            for word in (chunk.split()[0], chunk.split()[-1]):
                self.assertIn(word, words)

    def test_consecutive_chunks_overlap(self):
        chunks = split_text(SAMPLE, 300, 60)
        self.assertGreater(len(chunks), 3)
        for a, b in zip(chunks, chunks[1:]):
            self.assertTrue(any(b.startswith(a[i:]) for i in range(len(a))),
                            "next chunk should begin inside the previous one")

    def test_all_text_is_covered(self):
        normalized = " ".join(SAMPLE.split())
        chunks = split_text(SAMPLE, 300, 50)
        self.assertTrue(normalized.startswith(chunks[0]))
        self.assertTrue(normalized.endswith(chunks[-1]))

    def test_bad_overlap_rejected(self):
        with self.assertRaises(ValueError):
            split_text("abc", 100, 80)

    def test_pages_keep_their_page_number_and_global_index(self):
        pages = [{"page": 2, "text": SAMPLE}, {"page": 5, "text": "Short page."}]
        chunks = chunk_pages(pages, 300, 50)
        self.assertEqual([c["chunk_index"] for c in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[-1]["page"], 5)
        self.assertEqual(chunks[0]["page"], 2)


if __name__ == "__main__":
    unittest.main()
