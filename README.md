<<<<<<< HEAD
---
title: Medflow Medical AI Assistant
emoji: ??
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Medflow Medical AI Assistant (Medflow20 Core RAG Engine)

Specialized Thyroid Medical AI Platform built for clinical decision support, ATA guideline-grounded retrieval, lab interpretation (functioning, post-thyroidectomy, congenital), and PDF document vector search.

## Features
- **Medflow20 Core RAG Engine**: Hybrid Dense (BAAI/bge-small-en-v1.5 + ChromaDB) + Sparse (BM25) with Reciprocal Rank Fusion (RRF).
- **Grounded Evidence & Citations**: Section-aware page-level citations and verified ATA guideline grounding.
- **Specialized Lab Interpreter**: Multi-pathway assessment for functioning thyroid, post-thyroidectomy/ablated, congenital hypothyroidism, and pediatric cases.
- **Dynamic PDF Search & Vector Indexing**: Upload custom medical PDFs to index into ChromaDB & BM25 in real-time.
- **Production Single-Container Architecture**: FastAPI backend serving both API endpoints (/api/v1/*) and static SPA frontend (index.html) on port 7860.

## API Endpoints
- GET / � Medflow Single-Page Application (Web Interface)
- GET /health � System Health & RAG Engine Readiness
- POST /api/v1/query � Live Hybrid RAG Query & Synthesis
- POST /api/v1/interpret-labs � Specialized Thyroid Lab Interpretation
- GET /api/v1/search-docs � Vector & BM25 Document Search
- POST /api/v1/upload-pdf � Dynamic Medical PDF Indexing
- GET /api/v1/imported-documents � List Imported Documents
=======
# MedFlow: Evidence-Grounded Medical RAG for Thyroid Diseases

> **MedFlow** is a citation-bound clinical decision support system designed to retrieve traceable, verified medical evidence from official clinical guidelines before any language-model answer generation occurs.

$$\mathbf{Fluent\ Answer \neq Safe\ Answer}$$

In clinical decision support, large language models (LLMs) often generate syntactically fluent, authoritative-sounding answers that mask subtle factual hallucinations, obsolete clinical guidelines, or hazardous dosing recommendations. MedFlow solves this foundational safety failure by enforcing **vector indexing traceability, empirical ranking validation, strict JSON schema validation, and deterministic refusal gating** before any generative response reaches the clinician.

---

## 📁 Modular 3-Day Project Architecture

The project is structured into **3 independent, self-contained day modules**, each equipped with its own `README.md`, `config.py`, `.env`, interactive Jupyter Notebook, and automated test suite:

```text
medflow10/
├── README.md                                # Master project documentation & architectural guide
├── README_PROJECT_ANALYSIS.md               # In-depth engineering benchmark & analytical report
├── config.py                                # Master configuration (LLM, retrieval parameters, thresholds)
├── rag_pipeline.py                          # Master end-to-end clinical QA entry point
├── requirements.txt                         # Unified project dependencies
├── .env                                     # Configured Groq API Key & Model settings
│
├── day1/                                    # 🟢 DAY 1: DOCUMENT INGESTION & BASELINE
│   ├── README.md                            # Day 1 technical guide & execution commands
│   ├── config.py                            # Day 1 local configuration & path resolution
│   ├── .env                                 # Day 1 environment variables
│   ├── ingest.py                            # PDF loading, text cleaning, naive & section-aware chunking
│   ├── day1_pipeline.py                     # Runnable Day 1 baseline pipeline
│   ├── day1_task1_ingestion.ipynb           # Interactive Jupyter Notebook for Day 1
│   └── test_day1.py                         # Automated unit tests for Day 1
│
├── day2/                                    # 🔵 DAY 2: RETRIEVAL OPTIMIZATION & BENCHMARKING
│   ├── README.md                            # Day 2 technical guide & benchmark results
│   ├── config.py                            # Day 2 local configuration & retrieval settings
│   ├── .env                                 # Day 2 environment variables
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
│   ├── .env                                 # Day 3 environment variables
│   ├── generator.py                         # Citation-grounded generator & schema validator
│   ├── day3_grounded_generation.py          # Runnable Day 3 grounded QA & refusal demo
│   ├── day3_grounded_generation.ipynb       # Interactive Jupyter Notebook for Day 3
│   ├── test_day3.py                         # Automated unit test suite for Day 3
│   └── test_day3_compliance.py              # Full 10-suite Day 3 compliance test suite
│
├── data/                                    # 11 Official Thyroid Guidelines & Brochures (204 pages)
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
| **Day 1** | **Ingestion & Baseline** | [`day1/ingest.py`](file:///c:/Users/user/Desktop/medflow10/day1/ingest.py) | `python day1/day1_pipeline.py` | [`day1_task1_ingestion.ipynb`](file:///c:/Users/user/Desktop/medflow10/day1/day1_task1_ingestion.ipynb) | `python -m unittest day1/test_day1.py` |
| **Day 2** | **Retrieval Optimization** | [`day2/evaluate_retrieval.py`](file:///c:/Users/user/Desktop/medflow10/day2/evaluate_retrieval.py) | `python day2/evaluate_retrieval.py` | [`day2_retrieval_optimization.ipynb`](file:///c:/Users/user/Desktop/medflow10/day2/day2_retrieval_optimization.ipynb) | `python -m unittest day2/test_day2.py` |
| **Day 3** | **Grounded Generation** | [`day3/generator.py`](file:///c:/Users/user/Desktop/medflow10/day3/generator.py) | `python day3/day3_grounded_generation.py` | [`day3_grounded_generation.ipynb`](file:///c:/Users/user/Desktop/medflow10/day3/day3_grounded_generation.ipynb) | `python -m unittest day3/test_day3_compliance.py` |

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
* **Chunking Strategy:** Section-Aware Context-Enriched (200 tokens)
* **Hit@1:** `0.5625` (+200.0% over Day 1 Baseline)
* **Hit@4:** `0.8750` (+55.6% over Day 1 Baseline)
* **Precision@4:** `0.5312` (+82.1% over Day 1 Baseline)
* **MRR:** `0.7031` (+82.4% over Day 1 Baseline)
* **Mean Latency:** `20.53 ms` (-36.1% faster)

### Day 3 Grounding & Citation Compliance
* **JSON Schema Enforcement:** 100% compliant with `schema/response_schema.json`.
* **Citation Integrity:** 100% verified against retrieved chunks (fabricated sources automatically purged).
* **Refusal Precision:** 10/10 refusal benchmark categories pass with `confidence = "insufficient"`.

---

## 🧪 Running the Global Test Suite

To run all automated test suites across all 3 days simultaneously:

```bash
python -m unittest discover -s day1 -p "test_*.py" -v
python -m unittest discover -s day2 -p "test_*.py" -v
python -m unittest discover -s day3 -p "test_*.py" -v
```

All 18 automated tests pass with `OK`.
>>>>>>> origin/main
