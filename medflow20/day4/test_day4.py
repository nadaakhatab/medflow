import unittest

from day4.claim_validator import evaluate_faithfulness
from day4.evaluation_metrics import citation_accuracy, precision_at_k, refusal_metrics
from day4.safety_guardrails import apply_posthoc_guard, uncertainty_language
from day4.threshold_calibration import calibrate_threshold, calibrate_stratified_thresholds, threshold_for_query


class Day4CoreTests(unittest.TestCase):
    def test_threshold_calibration_separates_answerable_and_unsupported(self):
        samples = [
            {"expected_answerable": True, "top_score": 0.82},
            {"expected_answerable": True, "top_score": 0.76},
            {"expected_answerable": True, "top_score": 0.72},
            {"expected_answerable": False, "top_score": 0.41},
            {"expected_answerable": False, "top_score": 0.35},
            {"expected_answerable": False, "top_score": 0.28},
        ]
        result = calibrate_threshold(samples, max_unsafe_accept_rate=0.0)
        self.assertTrue(result["clean_separation"])
        self.assertEqual(result["selected_metrics"]["unsafe_accept_rate"], 0.0)
        self.assertEqual(result["selected_metrics"]["false_refusal_rate"], 0.0)
        self.assertGreater(result["selected_threshold"], 0.41)
        self.assertLessEqual(result["selected_threshold"], 0.72)

    def test_supported_numeric_claim_is_faithful(self):
        answer = "The guideline reports a levothyroxine dose of 1.6 mcg/kg/day."
        chunks = [{"retrieved_passage": "The guideline reports a levothyroxine dose of 1.6 mcg/kg/day for replacement therapy."}]
        result = evaluate_faithfulness(answer, chunks, lexical_threshold=0.35)
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertEqual(result["unsupported_claim_count"], 0)

    def test_unsupported_dosage_is_flagged_even_with_shared_drug_words(self):
        answer = "Levothyroxine should be given at 1.6 mcg/kg/day."
        chunks = [{"retrieved_passage": "Levothyroxine is the standard thyroid hormone replacement therapy."}]
        result = evaluate_faithfulness(answer, chunks, lexical_threshold=0.20)
        self.assertLess(result["faithfulness"], 1.0)
        detail = result["claim_details"][0]
        self.assertTrue(detail["missing_numbers"] or detail["missing_units"])

    def test_precision_at_4_uses_day2_relevance_rule(self):
        gt = {
            "expected_document": "A.pdf",
            "expected_page_range": [10, 12],
            "acceptable_alternative_sources": ["B.pdf"],
        }
        chunks = [
            {"document_name": "A.pdf", "page_number": 10, "retrieved_passage": "x"},
            {"document_name": "X.pdf", "page_number": 10, "retrieved_passage": "x"},
            {"document_name": "B.pdf", "page_number": 12, "retrieved_passage": "x"},
            {"document_name": "A.pdf", "page_number": 40, "retrieved_passage": "x"},
        ]
        result = precision_at_k(chunks, gt, 4)
        self.assertEqual(result["precision_at_k"], 0.5)

    def test_citation_accuracy_requires_exact_page_and_support(self):
        chunks = [{
            "document_name": "Hypo.pdf",
            "page_number": 4,
            "section_title": "Treatment",
            "retrieved_passage": "Levothyroxine is standard therapy for primary hypothyroidism.",
        }]
        answer = {
            "recommendation": "Levothyroxine is standard therapy.",
            "evidence": "Levothyroxine is standard therapy for primary hypothyroidism.",
            "citations": [{"document": "Hypo.pdf", "section": "Treatment", "page": 4}],
            "confidence": "high",
        }
        good = citation_accuracy(answer, chunks)
        self.assertEqual(good["citation_accuracy"], 1.0)
        bad = dict(answer)
        bad["citations"] = [{"document": "Hypo.pdf", "section": "Treatment", "page": 5}]
        self.assertEqual(citation_accuracy(bad, chunks)["citation_accuracy"], 0.0)

    def test_posthoc_guard_refuses_unsupported_numeric_claim(self):
        chunks = [{
            "document_name": "Graves.pdf", "page_number": 2, "section_title": "Treatment",
            "similarity_score": 0.80,
            "retrieved_passage": "Antithyroid drugs include methimazole and propylthiouracil.",
        }]
        answer = {
            "recommendation": "Propylthiouracil should be given at 5 mg/kg/day.",
            "evidence": "Antithyroid drugs include methimazole and propylthiouracil.",
            "citations": [{"document": "Graves.pdf", "section": "Treatment", "page": 2}],
            "confidence": "high",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.60)
        self.assertTrue(guarded["safety"]["guard_triggered"])
        self.assertEqual(guarded["answer"]["confidence"], "insufficient")

    def test_refusal_metrics_expose_unsafe_accepts_and_false_refusals(self):
        rows = [
            {"expected_answerable": True, "answered": True},
            {"expected_answerable": True, "answered": False},
            {"expected_answerable": False, "answered": False},
            {"expected_answerable": False, "answered": True},
        ]
        m = refusal_metrics(rows)
        self.assertEqual(m["fp_unsafe_accept"], 1)
        self.assertEqual(m["fn_false_refusal"], 1)
        self.assertEqual(m["answerability_accuracy"], 0.5)

    def test_stratified_thresholds_calibrate_only_when_labels_are_sufficient(self):
        samples = [
            {"query": "How is condition treated?", "query_family": "treatment_management", "expected_answerable": True, "top_score": 0.80},
            {"query": "What therapy is used?", "query_family": "treatment_management", "expected_answerable": True, "top_score": 0.76},
            {"query": "What medication for headache?", "query_family": "treatment_management", "expected_answerable": False, "top_score": 0.40},
            {"query": "Which antihypertensive today?", "query_family": "treatment_management", "expected_answerable": False, "top_score": 0.45},
            {"query": "What dose?", "query_family": "dosage", "expected_answerable": False, "top_score": 0.50},
        ]
        global_result = calibrate_threshold(samples, max_unsafe_accept_rate=0.0)
        result = calibrate_stratified_thresholds(samples, global_result, min_positive=2, min_negative=2, max_unsafe_accept_rate=0.0)
        self.assertEqual(result["family_thresholds"]["treatment_management"]["mode"], "family_calibrated")
        self.assertIn("global_fallback", result["family_thresholds"]["dosage"]["mode"])
        self.assertIsInstance(threshold_for_query("What treatment is recommended?", result), float)

    def test_partial_evidence_calibrates_answer_wording(self):
        chunks = [{
            "document_name": "Hypo.pdf", "page_number": 4, "section_title": "Treatment",
            "similarity_score": 0.66,
            "retrieved_passage": "Hypothyroidism is treated with synthetic thyroxine pills.",
        }]
        answer = {
            "recommendation": "Hypothyroidism is treated with synthetic thyroxine pills.",
            "evidence": "Hypothyroidism is treated with synthetic thyroxine pills.",
            "citations": [{"document": "Hypo.pdf", "section": "Treatment", "page": 4}],
            "confidence": "medium",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.60)
        self.assertTrue(guarded["safety"]["safe_to_return_original"])
        self.assertEqual(guarded["safety"]["evidence_strength"], "partial")
        self.assertIn("suggests", guarded["answer"]["recommendation"].lower())
        self.assertIn("disclaimer", guarded["safety"])

    def test_uncertainty_language_refuses_below_threshold(self):
        result = uncertainty_language(0.50, 0.65, 1.0, 1.0)
        self.assertEqual(result["evidence_strength"], "insufficient")
        self.assertIn("Refuse", result["language_guidance"])


    def test_unrelated_negation_in_long_chunk_does_not_false_flag_claim(self):
        answer = "Common symptoms include fatigue, cold intolerance, dry skin, constipation, and mild weight gain."
        chunks = [{
            "retrieved_passage": (
                "Common symptoms include fatigue, feeling cold, dry skin, constipation, and mild weight gain. "
                "Hypothyroidism does not cause obesity."
            )
        }]
        result = evaluate_faithfulness(answer, chunks, lexical_threshold=0.35)
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertFalse(result["claim_details"][0]["negation_conflict"])

    def test_true_local_negation_conflict_is_still_detected(self):
        answer = "Diagnostic FNA is indicated for purely cystic nodules."
        chunks = [{"retrieved_passage": "Diagnostic FNA is not indicated for purely cystic nodules."}]
        result = evaluate_faithfulness(answer, chunks, lexical_threshold=0.20)
        self.assertEqual(result["faithfulness"], 0.0)
        self.assertTrue(result["claim_details"][0]["negation_conflict"])

    def test_citation_accuracy_considers_all_chunks_on_same_page(self):
        chunks = [
            {
                "document_name": "Guide.pdf", "page_number": 2, "section_title": "General Content",
                "retrieved_passage": "Unrelated introduction on the same PDF page.",
            },
            {
                "document_name": "Guide.pdf", "page_number": 2, "section_title": "General Content",
                "retrieved_passage": "Hyperthyroidism is confirmed with T4, T3 and TSH blood tests.",
            },
        ]
        answer = {
            "recommendation": "Hyperthyroidism is confirmed with T4, T3 and TSH blood tests.",
            "evidence": "combined evidence",
            "citations": [{"document": "Guide.pdf", "section": "General Content", "page": 2}],
            "confidence": "high",
        }
        result = citation_accuracy(answer, chunks)
        self.assertEqual(result["citation_accuracy"], 1.0)
        self.assertEqual(result["claim_coverage"], 1.0)

    def test_citations_are_scored_against_recommendation_claims_not_aggregate_evidence(self):
        chunks = [
            {
                "document_name": "A.pdf", "page_number": 1, "section_title": "General Content",
                "retrieved_passage": "Antithyroid drugs are a treatment option.",
            },
            {
                "document_name": "B.pdf", "page_number": 3, "section_title": "General Content",
                "retrieved_passage": "Beta blockers can control hyperthyroid symptoms.",
            },
        ]
        answer = {
            "recommendation": "Antithyroid drugs are a treatment option. Beta blockers can control hyperthyroid symptoms.",
            "evidence": "Antithyroid drugs are a treatment option and beta blockers control symptoms.",
            "citations": [
                {"document": "A.pdf", "section": "General Content", "page": 1},
                {"document": "B.pdf", "section": "General Content", "page": 3},
            ],
            "confidence": "high",
        }
        result = citation_accuracy(answer, chunks)
        self.assertEqual(result["citation_accuracy"], 1.0)
        self.assertEqual(result["claim_coverage"], 1.0)

    def test_guard_repairs_partial_answer_instead_of_refusing_everything(self):
        chunks = [{
            "document_name": "Graves.pdf", "page_number": 2, "section_title": "General Content",
            "similarity_score": 0.82,
            "retrieved_passage": "Hyperthyroidism is confirmed with T4, T3 and TSH blood tests.",
        }]
        answer = {
            "recommendation": (
                "Hyperthyroidism is confirmed with T4, T3 and TSH blood tests. "
                "Every patient must receive exactly 99 mg of medicine daily."
            ),
            "evidence": "mixed",
            "citations": [{"document": "Wrong.pdf", "section": "General Content", "page": 99}],
            "confidence": "high",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.72)
        self.assertNotEqual(guarded["answer"]["confidence"], "insufficient")
        self.assertTrue(guarded["safety"]["repair_applied"])
        self.assertIn("Hyperthyroidism", guarded["answer"]["recommendation"])
        self.assertNotIn("99 mg", guarded["answer"]["recommendation"])
        self.assertEqual(guarded["safety"]["faithfulness"]["faithfulness"], 1.0)
        self.assertEqual(guarded["safety"]["citation_accuracy"]["citation_accuracy"], 1.0)

    def test_structured_refusal_is_not_treated_as_unsupported_medical_claim(self):
        chunks = [{"similarity_score": 0.40, "retrieved_passage": "thyroid evidence"}]
        answer = {
            "recommendation": "I couldn't find enough information in the indexed guideline to answer this confidently.",
            "evidence": "", "citations": [], "confidence": "insufficient",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.72)
        self.assertFalse(guarded["safety"]["guard_triggered"])
        self.assertEqual(guarded["safety"]["faithfulness"]["total_claims"], 0)
        self.assertEqual(guarded["answer"]["confidence"], "insufficient")

    def test_same_page_citation_unions_claim_coverage_across_chunks(self):
        chunks = [
            {
                "document_name": "Graves.pdf", "page_number": 2, "section_title": "General Content",
                "retrieved_passage": "Symptoms and physical exam findings plus T4, T3 and TSH support the diagnosis.",
            },
            {
                "document_name": "Graves.pdf", "page_number": 2, "section_title": "General Content",
                "retrieved_passage": "TRAb or TSI antibodies are measured for confirmation of Graves disease.",
            },
        ]
        answer = {
            "recommendation": (
                "Graves disease is diagnosed from symptoms and physical exam findings with T4, T3 and TSH. "
                "Confirmation uses TRAb or TSI antibodies."
            ),
            "evidence": "",
            "citations": [{"document": "Graves.pdf", "section": "General Content", "page": 2}],
            "confidence": "high",
        }
        result = citation_accuracy(answer, chunks)
        self.assertEqual(result["citation_accuracy"], 1.0)
        self.assertEqual(result["claim_coverage"], 1.0)
        self.assertEqual(result["covered_claims"], 2)

    def test_guideline_exception_in_same_sentence_does_not_negate_positive_rule(self):
        answer = (
            "Fine-needle aspiration biopsy is recommended for thyroid nodules that are at least 2 cm in size "
            "and for cervical lymph nodes that appear sonographically suspicious for cancer; "
            "it is not indicated for purely cystic nodules."
        )
        chunks = [{
            "retrieved_passage": (
                "A nodule should be at least 2 cm for FNA, observation without FNA may be considered for nodules <2 cm, "
                "purely cystic nodules are unlikely to be malignant and FNA is not indicated, and suspicious cervical "
                "lymph nodes should be biopsied with FNA."
            )
        }]
        result = evaluate_faithfulness(answer, chunks, lexical_threshold=0.35)
        self.assertEqual(result["faithfulness"], 1.0)

    def test_guard_removes_unsupported_parenthetical_identifier_but_keeps_treatment_claim(self):
        chunks = [{
            "document_name": "Graves.pdf", "page_number": 2, "section_title": "General Content",
            "similarity_score": 0.82,
            "retrieved_passage": (
                "Treatment options for Graves disease include antithyroid drugs such as methimazole and propylthiouracil, "
                "radioactive iodine therapy, thyroid surgery, and beta blockers to control symptoms."
            ),
        }]
        answer = {
            "recommendation": (
                "The treatment options for Graves disease include antithyroid drugs (such as methimazole or propylthiouracil), "
                "radioactive iodine therapy (I-131), thyroid surgery, and beta-blockers to control symptoms."
            ),
            "evidence": "", "citations": [], "confidence": "high",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.66)
        self.assertNotEqual(guarded["answer"]["confidence"], "insufficient")
        self.assertNotIn("I-131", guarded["answer"]["recommendation"])
        self.assertEqual(guarded["safety"]["faithfulness"]["faithfulness"], 1.0)
        self.assertEqual(guarded["safety"]["citation_accuracy"]["citation_accuracy"], 1.0)

    def test_multi_chunk_claim_can_be_jointly_supported_and_cited(self):
        chunks = [
            {
                "document_name": "CancerGuide.pdf", "page_number": 33, "section_title": "General Content",
                "similarity_score": 0.86,
                "retrieved_passage": (
                    "Low- to intermediate-risk patients include unifocal tumors <4 cm with no extrathyroidal extension "
                    "or lymph node metastases."
                ),
            },
            {
                "document_name": "CancerGuide.pdf", "page_number": 112, "section_title": "General Content",
                "similarity_score": 0.84,
                "retrieved_passage": (
                    "Response-to-therapy variables after total thyroidectomy and radioactive iodine ablation modify the initial risk estimate."
                ),
            },
        ]
        answer = {
            "recommendation": (
                "Recurrence risk is first stratified using features such as unifocal tumor size (<4 cm), absence of "
                "extrathyroidal extension or lymph-node metastases, and then refined after total thyroidectomy using "
                "response-to-therapy variables that modify the initial risk estimate."
            ),
            "evidence": "", "citations": [], "confidence": "high",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.72)
        self.assertNotEqual(guarded["answer"]["confidence"], "insufficient")
        self.assertEqual(guarded["safety"]["faithfulness"]["faithfulness"], 1.0)
        self.assertEqual(guarded["safety"]["citation_accuracy"]["citation_accuracy"], 1.0)
        self.assertEqual(guarded["safety"]["citation_accuracy"]["claim_coverage"], 1.0)


    def test_duplicate_passage_prefers_filename_year_consistent_with_content(self):
        content = (
            "Executive Summary of the 2025 American Thyroid Association guidelines: "
            "the updated 2025 guidelines place greater emphasis on active surveillance."
        )
        chunks = [
            {
                "document_name": "ATA_2016_Hyperthyroidism_Thyrotoxicosis_Guidelines.pdf",
                "page_number": 1, "section_title": "General Content", "similarity_score": 0.81,
                "retrieved_passage": content,
            },
            {
                "document_name": "praw-et-al-2025-executive-summary.pdf",
                "page_number": 1, "section_title": "General Content", "similarity_score": 0.80,
                "retrieved_passage": content,
            },
        ]
        answer = {
            "recommendation": "The updated 2025 guidelines place greater emphasis on active surveillance.",
            "evidence": "", "citations": [], "confidence": "high",
        }
        guarded = apply_posthoc_guard(answer, chunks, confidence_threshold=0.69)
        self.assertNotEqual(guarded["answer"]["confidence"], "insufficient")
        self.assertEqual(len(guarded["answer"]["citations"]), 1)
        self.assertIn("2025", guarded["answer"]["citations"][0]["document"])



if __name__ == "__main__":
    unittest.main(verbosity=2)
