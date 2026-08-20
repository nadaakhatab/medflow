# Day 3 — Grounded Generation & Citation

## 1. Overview & Objective
Day 3 implements the clinical answer generation and grounding safety layer:
* **Grounding System Prompt:** Restricts the model strictly to retrieved guideline evidence with zero outside clinical knowledge.
* **Response JSON Schema (`schema/response_schema.json`):** Enforces structured JSON output with `recommendation`, `evidence`, `citations`, and `confidence`.
* **Schema Validation & Invariants:** Enforces that high-confidence answers *must* contain supporting evidence and at least one citation.
* **Citation Integrity Check:** Cross-verifies generated document names, pages, and sections against actual retrieved chunks and eliminates hallucinations.
* **Deterministic Refusal Mechanism:** Pre-LLM threshold gating (`CONFIDENCE_THRESHOLD = 0.50`) and empty retrieval checks that safely refuse when evidence is lacking.
* **Refusal Benchmarks:** Evaluated against 10 distinct refusal categories (off-topic, personal advice, prompt injection, etc.).

---

## 2. Directory Contents
* **`generator.py`**: Core answer generation engine, Pydantic validation models (`ClinicalAnswer`, `Citation`, `ConfidenceLevel`), prompt builder, citation validator, and JSON Schema checker.
* **`day3_grounded_generation.py`**: Standalone executable demonstrating grounded answering on supported questions and safe refusal on out-of-scope queries.
* **`day3_grounded_generation.ipynb`**: Interactive Jupyter Notebook demonstrating grounding prompts, schema validation, and live answer generation.
* **`test_day3.py`**: Automated unit tests for prompt rules, schema validation pass/fail invariants, and refusal triggers.
* **`test_day3_compliance.py`**: Full 10-suite Day 3 compliance benchmark test suite.
* **`config.py`**: Local configuration resolving LLM parameters, paths, and refusal thresholds.
* **`.env`**: Local environment variables containing the Groq API key and model name.

---

## 3. How to Run Independently

### A. Run Grounded Generation & Refusal Demo:
```bash
python day3/day3_grounded_generation.py
```

### B. Run Automated Unit Tests:
```bash
python -m unittest day3/test_day3.py -v
python -m unittest day3/test_day3_compliance.py -v
```

### C. Run Interactive Notebook:
Open `day3/day3_grounded_generation.ipynb` in Jupyter Lab or VS Code and run all cells.
