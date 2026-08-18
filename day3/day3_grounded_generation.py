import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator import generate_answer, build_prompt, validate_with_json_schema, GROUNDING_SYSTEM_PROMPT
from rag_pipeline import ask_clinical_question, semantic_retrieval, get_embeddings
from langchain_chroma import Chroma
from config import settings


def run_day3_demo():
    print("=======================================================")
    print("DAY 3 — GROUNDED GENERATION & CITATION DEMO")
    print("=======================================================\n")

    # 1. Supported Clinical Question
    q1 = "What are the clinical symptoms of hypothyroidism?"
    print(f"--- 1. Testing Supported Question: '{q1}' ---")
    ans1 = ask_clinical_question(q1, top_k=4)
    print(json.dumps(ans1, indent=2))
    validate_with_json_schema(ans1)
    print("--> Schema Validation: PASSED\n")

    # 2. Refusal Case (Out of Scope)
    q2 = "What screening interval does this guideline recommend for breast cancer?"
    print(f"--- 2. Testing Refusal Case (Out of Scope): '{q2}' ---")
    ans2 = ask_clinical_question(q2, top_k=4)
    print(json.dumps(ans2, indent=2))
    validate_with_json_schema(ans2)
    print("--> Schema Validation: PASSED\n")


if __name__ == "__main__":
    run_day3_demo()
