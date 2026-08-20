import os
import sys
import json
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator import (
    generate_answer,
    build_prompt,
    validate_with_json_schema,
    ClinicalAnswer,
    ConfidenceLevel,
    GROUNDING_SYSTEM_PROMPT
)
from pydantic import ValidationError
from langchain_core.documents import Document


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.invoked = False

    def invoke(self, messages):
        self.invoked = True
        return MockLLMResponse(self.response_content)


class TestDay3GroundedGeneration(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            Document(
                page_content="Levothyroxine is the standard initial therapy for primary hypothyroidism.",
                metadata={
                    "document_name": "Hypothyroidism_web_booklet.pdf",
                    "page_number": 4,
                    "section_title": "Treatment & Management",
                    "similarity_score": 0.8850
                }
            )
        ]

    def test_prompt_has_all_components(self):
        self.assertIn("citation-bound clinical evidence assistant", GROUNDING_SYSTEM_PROMPT.lower())
        self.assertIn("answer only using the context passages", GROUNDING_SYSTEM_PROMPT.lower())
        self.assertIn("recommendation", GROUNDING_SYSTEM_PROMPT)
        self.assertIn("evidence", GROUNDING_SYSTEM_PROMPT)
        self.assertIn("citations", GROUNDING_SYSTEM_PROMPT)
        self.assertIn("confidence", GROUNDING_SYSTEM_PROMPT)

    def test_schema_valid_and_invalid(self):
        good = {
            "recommendation": "Start levothyroxine.",
            "evidence": "Levothyroxine is standard therapy.",
            "citations": [{"document": "Hypothyroidism_web_booklet.pdf", "section": "Treatment", "page": 4}],
            "confidence": "high"
        }
        self.assertTrue(validate_with_json_schema(good))

        broken = {
            "recommendation": "Start drug.",
            "evidence": "",
            "citations": [],
            "confidence": "high"
        }
        with self.assertRaises((ValidationError, ValueError)):
            ClinicalAnswer.model_validate(broken)

    def test_refusal_cases(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "day3_refusal_test_cases.json")
        with open(csv_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        for c in cases:
            mock_refusal = json.dumps({
                "recommendation": f"I couldn't find enough information in the indexed guideline to answer this query.",
                "evidence": "",
                "citations": [],
                "confidence": "insufficient"
            })
            res = generate_answer(c["query"], self.sample_chunks, llm=MockLLM(mock_refusal))
            self.assertEqual(res["confidence"], "insufficient")
            self.assertEqual(res["evidence"], "")
            self.assertEqual(res["citations"], [])
            self.assertTrue(validate_with_json_schema(res))


if __name__ == "__main__":
    unittest.main(verbosity=2)
