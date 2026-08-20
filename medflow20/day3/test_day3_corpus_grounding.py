"""Corpus-integrity checks for Day 3 examples.

These tests ensure the supported treatment fixture is tied to text that actually exists
in the submitted thyroid corpus and that an unsupported exact numeric dose is not
silently treated as source-backed just because the drug name appears.
"""
from pathlib import Path
import unittest

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


class TestDay3CorpusGrounding(unittest.TestCase):
    def test_hypothyroidism_treatment_example_exists_in_real_pdf(self):
        pdf = ROOT / "data" / "Hypothyroidism_web_booklet.pdf"
        reader = PdfReader(str(pdf))
        page4 = (reader.pages[3].extract_text() or "").lower()
        self.assertIn("hypothyroidism is treated", page4)
        self.assertIn("synthetic thyroxine", page4)

    def test_exact_1_6_mcg_per_kg_dose_is_not_in_hypothyroidism_booklet(self):
        pdf = ROOT / "data" / "Hypothyroidism_web_booklet.pdf"
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).lower()
        self.assertNotIn("1.6 mcg/kg/day", text)
        self.assertNotIn("1.6 mcg/kg", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
