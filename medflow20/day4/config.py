"""Day 4 configuration derived from the frozen Day 2 retriever."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = PROJECT_ROOT / "results" / "day4"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "best_retrieval_config.json"
GROUND_TRUTH_PATH = EVALUATION_DIR / "thyroid_ground_truth.json"
REFUSAL_CASES_PATH = EVALUATION_DIR / "day3_refusal_test_cases.json"
LIVE_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
LIVE_COLLECTION_NAME = "thyroid_section_aware"
FROZEN_DAY4_PERSIST_DIR = PROJECT_ROOT / "chroma_db_day2_frozen"
FROZEN_DAY4_COLLECTION_NAME = "thyroid_day2_frozen"

TARGET_FAITHFULNESS = 0.90
CLAIM_SUPPORT_THRESHOLD = 0.35
MIN_CITATION_ACCURACY = 1.00
MAX_UNSAFE_ACCEPT_RATE = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
STRICT_INDEX_MATCH = True

CLINICAL_DISCLAIMER = (
    "MedFlow provides guideline-grounded clinical decision support and does not "
    "replace individualized clinical judgment or professional medical evaluation."
)


def load_frozen_retriever_config() -> Dict[str, Any]:
    if not FROZEN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Frozen Day 2 config not found: {FROZEN_CONFIG_PATH}")
    with FROZEN_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


FROZEN_RETRIEVER = load_frozen_retriever_config()
EMBEDDING_MODEL_NAME = FROZEN_RETRIEVER.get("embedding_model", "BAAI/bge-small-en-v1.5")
BGE_QUERY_PREFIX = FROZEN_RETRIEVER.get(
    "query_instruction",
    "Represent this sentence for searching relevant passages: ",
)
TOP_K = int(FROZEN_RETRIEVER.get("top_k_retrieval", FROZEN_RETRIEVER.get("top_k", 4)))
EXPECTED_INDEXED_CHUNKS = int(FROZEN_RETRIEVER.get("total_indexed_chunks", 1470))
CHUNK_SIZE_TOKENS = int(FROZEN_RETRIEVER.get("chunk_size_tokens", 200))
CHUNK_OVERLAP_TOKENS = int(FROZEN_RETRIEVER.get("chunk_overlap_tokens", 0))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
