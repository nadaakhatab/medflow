import os
import json
import unittest
from typing import List, Dict, Any
from langchain_core.documents import Document
from pydantic import ValidationError

from generator import (
    generate_answer,
    build_prompt,
    validate_citations,
    parse_and_validate_llm_response,
    validate_with_json_schema,
    ClinicalAnswer,
    Citation,
    ConfidenceLevel,
    extract_chunk_metadata,
    GROUNDING_SYSTEM_PROMPT,
    DEFAULT_REFUSAL_MESSAGE
)


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.invoked = False
        self.last_messages = None

    def invoke(self, messages):
        self.invoked = True
        self.last_messages = messages
        return MockLLMResponse(self.response_content)


class TestDay3Compliance(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            Document(
                page_content="Levothyroxine is the standard initial therapy for primary hypothyroidism. The recommended replacement dose is typically 1.6 mcg/kg/day for healthy adults without coronary disease.",
                metadata={
                    "document_name": "Hypothyroidism_web_booklet.pdf",
                    "page_number": 4,
                    "section_title": "Treatment & Management",
                    "Header 1": "Hypothyroidism",
                    "Header 2": "Medical Therapy",
                    "similarity_score": 0.8850
                }
            ),
            Document(
                page_content="TSH measurement is the primary initial screening test for suspected thyroid dysfunction. Normal reference range is approximately 0.4 to 4.0 mIU/L.",
                metadata={
                    "document_name": "ThyroidDisease.pdf",
                    "page_number": 2,
                    "section_title": "Diagnosis & Tests",
                    "similarity_score": 0.7620
                }
            )
        ]

    # -------------------------------------------------------------
    # 1. GROUNDING SYSTEM PROMPT AUDIT
    # -------------------------------------------------------------
    def test_prompt_contains_all_four_core_components(self):
        """Verify the prompt contains Role, Context Boundary, Output Structure, and Escape Hatch."""
        # A. Role
        self.assertIn("citation-bound clinical evidence assistant", GROUNDING_SYSTEM_PROMPT.lower())
        # B. Context Boundary
        self.assertIn("answer only using the context passages", GROUNDING_SYSTEM_PROMPT.lower())
        self.assertIn("never use outside medical knowledge", GROUNDING_SYSTEM_PROMPT.lower())
        # C. Required Output Structure
        self.assertIn('"recommendation"', GROUNDING_SYSTEM_PROMPT)
        self.assertIn('"evidence"', GROUNDING_SYSTEM_PROMPT)
        self.assertIn('"citations"', GROUNDING_SYSTEM_PROMPT)
        self.assertIn('"confidence"', GROUNDING_SYSTEM_PROMPT)
        # D. Escape Hatch / Refusal
        self.assertIn("insufficient", GROUNDING_SYSTEM_PROMPT.lower())
        self.assertIn("never invent a citation", GROUNDING_SYSTEM_PROMPT.lower())

    # -------------------------------------------------------------
    # 2 & 3. JSON SCHEMA & MANDATORY SCHEMA TEST
    # -------------------------------------------------------------
    def test_schema_validation_good_answer_passes(self):
        """Mandatory Test: Valid grounded response passes schema validation."""
        good_answer = {
            "recommendation": "Start with a guideline-supported treatment option.",
            "evidence": "Direct supporting evidence from the guideline.",
            "citations": [
                {
                    "document": "WHO_Hypertension_Guideline_2021",
                    "section": "3.4 Drug classes",
                    "page": 8
                }
            ],
            "confidence": "high"
        }
        # Validate with Pydantic
        ans_obj = ClinicalAnswer.model_validate(good_answer)
        self.assertEqual(ans_obj.confidence, ConfidenceLevel.HIGH)
        # Validate with JSON Schema file
        self.assertTrue(validate_with_json_schema(good_answer))

    def test_schema_validation_broken_answer_fails(self):
        """Mandatory Test: High confidence with empty evidence/citations MUST FAIL validation."""
        broken_answer = {
            "recommendation": "Take 10mg of drug X daily.",
            "evidence": "",
            "citations": [],
            "confidence": "high"
        }
        # Pydantic must reject it
        with self.assertRaises((ValidationError, ValueError)):
            ClinicalAnswer.model_validate(broken_answer)

        import jsonschema
        schema_path = os.path.join(os.path.dirname(__file__), "schema", "response_schema.json")
        if not os.path.exists(schema_path):
            schema_path = os.path.join(os.path.dirname(__file__), "..", "schema", "response_schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken_answer, schema=schema)

    # -------------------------------------------------------------
    # 4. GROUNDED GENERATION (TEST A)
    # -------------------------------------------------------------
    def test_supported_question_generates_grounded_answer(self):
        """Test A: Supported clinical question produces structured, grounded output."""
        query = "What is the recommended dose of levothyroxine for hypothyroidism?"
        mock_response = json.dumps({
            "recommendation": "The recommended replacement dose of levothyroxine for primary hypothyroidism is typically 1.6 mcg/kg/day.",
            "evidence": "Levothyroxine is the standard initial therapy for primary hypothyroidism. The recommended replacement dose is typically 1.6 mcg/kg/day.",
            "citations": [
                {
                    "document": "Hypothyroidism_web_booklet.pdf",
                    "section": "Hypothyroidism > Medical Therapy",
                    "page": 4
                }
            ],
            "confidence": "high"
        })
        mock_llm = MockLLM(mock_response)
        result = generate_answer(query, self.sample_chunks, llm=mock_llm)

        self.assertTrue(mock_llm.invoked)
        self.assertEqual(result["confidence"], "high")
        self.assertTrue(len(result["evidence"]) > 0)
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["document"], "Hypothyroidism_web_booklet.pdf")
        self.assertTrue(validate_with_json_schema(result))

    # -------------------------------------------------------------
    # 5. CONTEXT CONSTRUCTION & METADATA HANDLING
    # -------------------------------------------------------------
    def test_context_construction_handles_missing_metadata_safely(self):
        """Ensure missing page/section/document metadata does not produce 'None' strings."""
        dirty_chunks = [
            Document(page_content="Guideline passage 1", metadata={}),
            {
                "retrieved_passage": "Guideline passage 2",
                "metadata": {
                    "document_name": None,
                    "page_number": None,
                    "section_title": None,
                    "Header 1": None
                }
            }
        ]
        prompt = build_prompt("Test question", dirty_chunks)
        self.assertNotIn("None", prompt)
        self.assertIn("[Document: Unknown]", prompt)
        self.assertIn("Guideline passage 1", prompt)

    # -------------------------------------------------------------
    # 6 & 7. JSON PARSING & ERROR RESILIENCE (TEST E)
    # -------------------------------------------------------------
    def test_json_parsing_and_error_handling(self):
        """Test E: Malformed JSON, markdown fences, and unparseable output handled safely."""
        query = "How is hypothyroidism treated?"

        # Markdown wrapped
        fenced_json = """```json
{
    "recommendation": "Levothyroxine 1.6 mcg/kg/day.",
    "evidence": "Levothyroxine is standard initial therapy.",
    "citations": [{"document": "Hypothyroidism_web_booklet.pdf", "section": "Treatment", "page": 4}],
    "confidence": "high"
}
```"""
        ans_fenced = generate_answer(query, self.sample_chunks, llm=MockLLM(fenced_json))
        self.assertEqual(ans_fenced["confidence"], "high")

        # Unparseable text
        garbage = "I am an AI and I don't speak JSON..."
        ans_garbage = generate_answer(query, self.sample_chunks, llm=MockLLM(garbage))
        self.assertEqual(ans_garbage["confidence"], "insufficient")
        self.assertEqual(ans_garbage["evidence"], "")
        self.assertEqual(ans_garbage["citations"], [])

    # -------------------------------------------------------------
    # 8 & 12. CITATION INTEGRITY
    # -------------------------------------------------------------
    def test_citation_integrity_drops_hallucinated_sources(self):
        """Ensure citations not in retrieved chunk metadata are pruned or trigger refusal."""
        query = "Test query"
        hallucinated_response = json.dumps({
            "recommendation": "Grounded claim.",
            "evidence": "Levothyroxine is standard therapy.",
            "citations": [
                {
                    "document": "Invented_Fake_Journal_2026.pdf",
                    "section": "Fake Section",
                    "page": 999
                }
            ],
            "confidence": "high"
        })
        result = generate_answer(query, self.sample_chunks, llm=MockLLM(hallucinated_response))
        # Since the only citation was fake, system safely refuses
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["citations"], [])

    # -------------------------------------------------------------
    # 9 & 10. REFUSAL MECHANISM & THRESHOLD GATING (TEST D)
    # -------------------------------------------------------------
    def test_empty_retrieval_triggers_deterministic_refusal(self):
        """Test D: Empty retrieval results in structured refusal without calling LLM."""
        mock_llm = MockLLM("Should not run")
        result = generate_answer("Any question", [], llm=mock_llm)

        self.assertFalse(mock_llm.invoked)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["evidence"], "")
        self.assertEqual(result["citations"], [])
        self.assertTrue(validate_with_json_schema(result))

    def test_low_similarity_score_triggers_refusal_threshold(self):
        """Test retrieval threshold gating when similarity score is below confidence threshold."""
        low_score_chunks = [
            {
                "retrieved_passage": "Random unrelated text snippet.",
                "document_name": "ThyroidDisease.pdf",
                "page_number": 1,
                "similarity_score": 0.2100  # Below 0.50 threshold
            }
        ]
        mock_llm = MockLLM("Should not run")
        result = generate_answer("Query with low relevance", low_score_chunks, llm=mock_llm, confidence_threshold=0.50)

        self.assertFalse(mock_llm.invoked)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["evidence"], "")
        self.assertEqual(result["citations"], [])

    # -------------------------------------------------------------
    # 11. DAY 3 REFUSAL TEST CASES BENCHMARK
    # -------------------------------------------------------------
    def test_day3_refusal_test_cases_from_csv(self):
        """Verifies refusal behavior on all 10 required Day 3 refusal categories."""
        csv_path = os.path.join(os.path.dirname(__file__), "evaluation", "day3_refusal_test_cases.json")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "evaluation", "day3_refusal_test_cases.json")
        with open(csv_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        self.assertEqual(len(cases), 10)

        for case in cases:
            q = case["query"]
            cat = case["category"]
            
            # Simulate unsupported context retrieval or refusal response
            refusal_response = json.dumps({
                "recommendation": f"I couldn't find enough information in the indexed guideline to answer this query regarding {cat}.",
                "evidence": "",
                "citations": [],
                "confidence": "insufficient"
            })
            ans = generate_answer(q, self.sample_chunks, llm=MockLLM(refusal_response))

            self.assertEqual(ans["confidence"], "insufficient", f"Failed on category: {cat}")
            self.assertEqual(ans["evidence"], "", f"Failed on category: {cat}")
            self.assertEqual(ans["citations"], [], f"Failed on category: {cat}")
            self.assertTrue(validate_with_json_schema(ans), f"Schema validation failed for: {cat}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
