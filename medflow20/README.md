# MedFlow: Evidence-Grounded Medical RAG for Thyroid Diseases

> **MedFlow** is a citation-bound clinical decision support system designed to retrieve traceable, verified medical evidence from official clinical guidelines before any language-model answer generation occurs.

$$\mathbf{Fluent\ Answer \neq Safe\ Answer}$$

In clinical decision support, large language models (LLMs) often generate syntactically fluent, authoritative-sounding answers that mask subtle factual hallucinations, obsolete clinical guidelines, or hazardous dosing recommendations. MedFlow solves this foundational safety failure by enforcing **vector indexing traceability, empirical ranking validation, strict JSON schema validation, and deterministic refusal gating** before any generative response reaches the clinician.

---

## 📁 Modular 5-Day Hackathon Architecture

The implemented source currently covers **Days 1–4**, with **Day 5 reserved for final UI integration, live demo, presentation, and judge evaluation**. Day 4 is intentionally implemented as a non-destructive safety/evaluation wrapper over the Day 1–3 pipeline:

```text
medflow20/
├── README.md                                # Master project documentation & architectural guide
├── config.py                                # Master configuration (LLM, retrieval parameters, thresholds)
├── rag_pipeline.py                          # Master end-to-end clinical QA entry point
├── requirements.txt                         # Unified project dependencies
├── .env.example                             # Safe environment template; local .env is excluded
│
├── day1/                                    # 🟢 DAY 1: DOCUMENT INGESTION & BASELINE
│   ├── README.md                            # Day 1 technical guide & execution commands
│   ├── config.py                            # Day 1 local configuration & path resolution
│   ├── ingest.py                            # PDF loading, text cleaning, naive & section-aware chunking
│   ├── day1_pipeline.py                     # Runnable Day 1 baseline pipeline
│   ├── day1_task1_ingestion.ipynb           # Interactive Jupyter Notebook for Day 1
│   └── test_day1.py                         # Automated unit tests for Day 1
│
├── day2/                                    # 🔵 DAY 2: RETRIEVAL OPTIMIZATION & BENCHMARKING
│   ├── README.md                            # Day 2 technical guide & benchmark results
│   ├── config.py                            # Day 2 local configuration & retrieval settings
│   ├── evaluate_retrieval.py                # Ground Truth evaluation engine (Hit@k, Precision, MRR)
│   ├── chunk_experiments.py                 # Chunk size experiments (100 vs 200 vs 500 tokens)
│   ├── embedding_benchmark.py               # Embedding model comparison benchmark
│   ├── reranker_experiment.py               # Cross-Encoder re-ranker trade-off analysis
│   ├── validate_top_k.py                    # Top-K empirical validation (K=3, 4, 5)
│   ├── day2_retrieval_optimization.ipynb    # Interactive Jupyter Notebook for Day 2
│   └── test_day2.py                         # Automated unit tests for Day 2
│
├── day3/                                    # 🟣 DAY 3: GROUNDED GENERATION & CITATION
│   ├── README.md                            # Day 3 technical guide & grounding rules
│   ├── config.py                            # Day 3 local configuration & refusal thresholds
│   ├── generator.py                         # Citation-grounded generator & schema validator
│   ├── day3_grounded_generation.py          # Runnable Day 3 grounded QA & refusal demo
│   ├── day3_grounded_generation.ipynb       # Interactive Jupyter Notebook for Day 3
│   ├── test_day3.py                         # Automated unit test suite for Day 3
│   ├── test_day3_compliance.py              # Day 3 refusal/schema compliance suite
│   └── DAY3_REVIEW.md                       # Requirement-by-requirement review before Day 4
│
├── day4/                                    # 🟠 DAY 4: SAFETY, GUARDRAILS & INTERNAL EVALUATION
│   ├── README.md                            # Day 4 guide, metrics, commands & judge explanation
│   ├── RESPONSIBLE_AI_CHECKLIST.md          # Clinical Responsible-AI controls & team sign-off
│   ├── config.py                            # Frozen Day 2 contract + Day 4 safety targets
│   ├── index_audit.py                       # Read-only persisted-index reproducibility audit
│   ├── frozen_index_builder.py              # Separate exact 200-token/0-overlap index builder
│   ├── threshold_calibration.py             # Answerable-vs-unsupported threshold sweep
│   ├── claim_validator.py                   # Post-hoc unsupported-claim detection
│   ├── evaluation_metrics.py                # Precision@K, citation accuracy, faithfulness
│   ├── risk_classifier.py                   # ALLOWED / NEEDS_CAUTION / REFUSE_REDIRECT
│   ├── safety_guardrails.py                 # Fail-closed post-hoc guard + uncertainty language
│   ├── day4_pipeline.py                     # Live safe wrapper around Day 3
│   ├── evaluate_day4.py                     # Full internal evaluation and artifact logger
│   ├── day4_safety_evaluation.ipynb         # Judge/demo notebook
│   ├── test_day4.py                         # Day 4 metric/guard tests
│   └── test_day4_compliance.py              # Competition checklist tests
│
├── day5/                                    # ⏳ DAY 5: UI integration, live demo & presentation (next stage)
│
├── data/                                    # 11 Official Thyroid Guidelines & Brochures (202 extracted pages in verified run)
├── schema/
│   └── response_schema.json                 # Master JSON Schema draft-07 for clinical answers
├── evaluation/                              # Master Evaluation Datasets (Ground Truth & Refusal Benchmarks)
│   ├── thyroid_ground_truth.json            # 16 official Ground Truth QA pairs
│   ├── thyroid_ground_truth.csv             # Tabular Ground Truth
│   ├── day3_refusal_test_cases.json         # 10 benchmark refusal test queries
│   └── day3_refusal_test_cases.csv          # Tabular refusal benchmark
├── results/                                 # Official Benchmark Scorecards & Frozen Configs
│   ├── best_retrieval_config.json           # Final frozen retriever configuration
│   └── retrieval_summary.csv                # Complete experiment comparison table
└── chroma_db/                               # Persisted ChromaDB Vector Store
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create or edit `.env` in the root folder:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
CONFIDENCE_THRESHOLD=0.50
```

---

## 🔬 Daily Workflows & Command Matrix

| Day | Focus Area | Key Module | Runnable Script | Interactive Notebook | Automated Tests |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Day 1** | **Ingestion & Baseline** | [`day1/ingest.py`](file:///c:/Users/user/Desktop/medflow20/day1/ingest.py) | `python day1/day1_pipeline.py` | [`day1_task1_ingestion.ipynb`](file:///c:/Users/user/Desktop/medflow20/day1/day1_task1_ingestion.ipynb) | `python -m unittest day1/test_day1.py` |
| **Day 2** | **Retrieval Optimization** | [`day2/evaluate_retrieval.py`](file:///c:/Users/user/Desktop/medflow20/day2/evaluate_retrieval.py) | `python day2/evaluate_retrieval.py` | [`day2_retrieval_optimization.ipynb`](file:///c:/Users/user/Desktop/medflow20/day2/day2_retrieval_optimization.ipynb) | `python -m unittest day2/test_day2.py` |
| **Day 3** | **Grounded Generation** | [`day3/generator.py`](file:///c:/Users/user/Desktop/medflow20/day3/generator.py) | `python day3/day3_grounded_generation.py` | [`day3_grounded_generation.ipynb`](file:///c:/Users/user/Desktop/medflow20/day3/day3_grounded_generation.ipynb) | `python -m unittest day3/test_day3_compliance.py` |
| **Day 4** | **Safety & Evaluation** | `day4/evaluate_day4.py` | `python day4/evaluate_day4.py --audit-index` / `--full` | `day4/day4_safety_evaluation.ipynb` | `python -m unittest discover -s day4 -p "test_*.py" -v` |
| **Day 5** | **Integration, Evidence Panel & Live Demo** | **Pending / next stage** | — | — | — |

---

## 💡 Master End-to-End Clinical QA

You can run the entire RAG pipeline from python:

```python
from rag_pipeline import ask_clinical_question

# 1. Ask a supported thyroid question
result = ask_clinical_question("What are the clinical symptoms of hypothyroidism?")
print(result)

# 2. Ask an unsupported question (triggers safe refusal)
refusal = ask_clinical_question("What is the surgical chemotherapy protocol for glioblastoma?")
print(refusal)
```

### Example Structured Output:
```json
{
  "recommendation": "The clinical symptoms of hypothyroidism include fatigue, cold intolerance, dry skin, coarse brittle hair, mild weight gain (5-20 lbs), memory impairment, muscle cramps, peripheral edema, constipation, hoarse voice, and goiter.",
  "evidence": "The booklet lists a comprehensive set of hypothyroidism symptoms covering energy loss, temperature intolerance, skin and hair changes, weight changes, cognitive changes, respiratory and musculoskeletal symptoms...",
  "citations": [
    {
      "document": "Hypothyroidism_web_booklet.pdf",
      "section": "Symptoms & Clinical Presentation",
      "page": 5
    }
  ],
  "confidence": "high"
}
```

---

## 📊 Key Verified Benchmarks

### Day 2 Retrieval Optimization Scorecard (16 Ground Truth Questions)
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (L2 Normalized, Cosine Space)
* **Chunking Strategy:** Token-Aware Recursive Splitter (200 tokens / 0 overlap)
* **Hit@1:** `0.5625` (+200.0% over Day 1 Baseline)
* **Hit@3:** `0.8125`
* **Hit@4:** `0.8750` (+55.6% over Day 1 Baseline)
* **Hit@5:** `0.8750`
* **Precision@3:** `0.5417`
* **Precision@4:** `0.5312` (+82.1% over Day 1 Baseline)
* **Precision@5:** `0.5000`
* **MRR:** `0.7031` (+82.4% over Day 1 Baseline)
* **Mean Latency:** `20.53 ms` (-36.1% faster)

### Day 3 Grounding & Citation Compliance
* **JSON Schema Enforcement:** Structured responses are validated by Pydantic and `schema/response_schema.json`; missing schema now fails closed.
* **Citation Integrity:** Generated citations are restricted to retrieved source metadata; fabricated sources trigger safe refusal.
* **Refusal Compliance Coverage:** 10/10 refusal categories exercise the structured refusal contract with controlled mock outputs. End-to-end refusal precision is measured in Day 4 after threshold calibration.

### Day 4 Safety & Internal Evaluation
* **Threshold Calibration:** sweeps answerable Day 2 cases versus unsupported/refusal cases instead of choosing a threshold by intuition; query-family thresholds are used only when subgroup labels are sufficient, otherwise the global threshold is the explicit fallback.
* **Unsupported-Claim Detection:** independent post-hoc claim support check with conservative numeric/dosage protection.
* **Named Day 4 Metrics:** Precision@4, Citation Accuracy, Faithfulness, plus refusal/unsafe-accept metrics.
* **Input Risk Classification:** explicit `ALLOWED`, `NEEDS_CAUTION`, and `REFUSE_REDIRECT` states before live retrieval.
* **Responsible AI:** evidence-strength language, fail-closed refusal, and a visible clinical-judgment disclaimer.
* **Reproducibility Guard:** the legacy `chroma_db/thyroid_section_aware` collection contains 1,696 embeddings and is retained only as an older artifact. Day 4 uses the separate audited `chroma_db_day2_frozen/thyroid_day2_frozen` index with exactly **1,470 chunks**, matching the frozen Day 2 configuration.

---

## 🧪 Running the Global Test Suite

Run the source test suites from the activated project environment:

```bash
python -m unittest discover -s day1 -p "test_*.py" -v
python -m unittest discover -s day2 -p "test_*.py" -v
python -m unittest discover -s day3 -p "test_*.py" -v
python -m unittest discover -s day4 -p "test_*.py" -v
```

Then run the Day 4 reproducibility audit before reporting evaluation numbers:

```bash
python day4/evaluate_day4.py --audit-index --persist-dir chroma_db_day2_frozen --collection thyroid_day2_frozen
```

The repository does not pre-fill or fabricate Day 4 benchmark values. The full metrics are written to `results/day4/` only after a real run against an audited index.


---

## ✅ Agenda Alignment Snapshot

The project intentionally preserves empirically selected settings rather than changing them to mirror illustrative slide values.

- **Day 1:** 11 official thyroid PDFs, 202 extracted pages in the verified run, cleaning, character-based baseline chunking, metadata, and vector indexing.
- **Day 2:** 16-query ground truth, chunk/embedding/reranker experiments, Precision@3/4/5, Hit@K, MRR, and frozen `200 tokens / 0 overlap / Top-K 4` retriever.
- **Day 3:** grounded structured generation, supporting evidence, page-level citations, confidence, schema enforcement, and refusal behavior.
- **Day 4:** explicit input-risk state, empirically calibrated retrieval threshold, unsupported-claim checks, numerical/negation safeguards, citation verification, faithfulness, repair-before-refuse, Responsible-AI disclaimer, and 26-query internal evaluation.
- **Day 5:** still to be completed as the presentation/UI stage: Evidence Panel, three predefined live-demo cases (success / complex multi-step / safe refusal), frozen prototype, pitch, and scalability roadmap.

### Internal-evaluation claims

When reporting Day 4's final 100% metrics, use the precise wording:

> **On the 26-query internal safety and grounding evaluation set**, the final guarded outputs achieved 100% answerability accuracy, 100% final citation accuracy, 100% final faithfulness, 0% unsafe accepts, and 0% false refusals.

These are **internal benchmark results, not a claim of universal clinical accuracy or production readiness**. External clinical validation and a larger independent held-out set are required before real-world deployment.
