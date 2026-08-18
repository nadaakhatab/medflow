import os
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from day1.ingest import load_pdfs, naive_chunk_documents, section_aware_chunk_documents, clean_medical_text


class TestDay1Ingestion(unittest.TestCase):

    def test_clean_medical_text(self):
        raw = "American Thyroid Association\xa0guidelines www.thyroid.org\n\n\n\nDiagnosis."
        cleaned = clean_medical_text(raw)
        self.assertNotIn("American Thyroid Association", cleaned)
        self.assertNotIn("www.thyroid.org", cleaned)
        self.assertNotIn("\xa0", cleaned)
        self.assertIn("Diagnosis.", cleaned)

    def test_pdf_loading(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        docs = load_pdfs(data_dir)
        self.assertGreaterEqual(len(docs), 200, "Expected at least 200 non-empty pages across the 11 ingested thyroid PDFs")

    def test_chunking_strategies(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        docs = load_pdfs(data_dir)
        
        # Test Naive Chunking
        naive_chunks = naive_chunk_documents(docs[:5], chunk_size=500, chunk_overlap=50)
        self.assertTrue(len(naive_chunks) > 0)
        self.assertEqual(naive_chunks[0].metadata["chunking_strategy"], "naive_fixed_size")

        # Test Section-Aware Chunking
        section_chunks = section_aware_chunk_documents(docs[:5], chunk_size=550, chunk_overlap=70)
        self.assertTrue(len(section_chunks) > 0)
        self.assertEqual(section_chunks[0].metadata["chunking_strategy"], "section_aware")
        self.assertIn("section_title", section_chunks[0].metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
