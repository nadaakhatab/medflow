import os
import sys
import json
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestDay2RetrievalOptimization(unittest.TestCase):

    def setUp(self):
        self.ground_truth_file = os.path.join(os.path.dirname(__file__), "..", "evaluation", "thyroid_ground_truth.json")
        self.best_config_file = os.path.join(os.path.dirname(__file__), "..", "results", "best_retrieval_config.json")

    def test_ground_truth_dataset_exists_and_valid(self):
        """Verify the 16 Ground Truth evaluation questions are properly formatted."""
        self.assertTrue(os.path.exists(self.ground_truth_file))
        with open(self.ground_truth_file, "r", encoding="utf-8") as f:
            gt = json.load(f)
        self.assertEqual(len(gt), 16)
        for item in gt:
            self.assertIn("query_id", item)
            self.assertIn("question", item)
            self.assertIn("expected_document", item)
            self.assertIn("expected_page_range", item)

    def test_best_retrieval_configuration_frozen(self):
        """Verify the final selected configuration frozen in Day 2."""
        self.assertTrue(os.path.exists(self.best_config_file))
        with open(self.best_config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        sel = config_data.get("selected_configuration", {})
        self.assertEqual(sel.get("embedding_model"), "BAAI/bge-small-en-v1.5")
        self.assertEqual(sel.get("top_k_retrieval"), 4)
        self.assertEqual(sel.get("chunk_size_tokens"), 200)
        self.assertFalse(sel.get("reranker_enabled"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
