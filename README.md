# Medflow Medical AI Assistant (Medflow20 Core RAG Engine)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Medflow** is an advanced, production-grade Clinical AI platform engineered for evidence-based medical decision support, American Thyroid Association (ATA) guideline-grounded retrieval, interactive thyroid lab interpretation, and dynamic PDF document search.

---

## 🌟 Key Features

- **Hybrid Dense-Sparse RAG Engine (Medflow20)**: Combines dense vector retrieval (`BAAI/bge-small-en-v1.5` + ChromaDB) with sparse lexical search (BM25) via Reciprocal Rank Fusion (RRF).
- **Verifiable Clinical Citations**: Provides section-aware and page-level citations from official clinical guidelines for every answer.
- **Specialized Lab Interpreter**: Multi-pathway assessment engine for functioning thyroid, post-thyroidectomy/ablated, congenital hypothyroidism, and pediatric cases.
- **Dynamic PDF Ingestion & Indexing**: Real-time upload, parsing, and vector indexing of custom medical PDFs.
- **Single-Port Desktop & Web Architecture**: Integrated FastAPI backend serving both REST API endpoints (`/api/v1/*`) and modern SPA UI (`index.html`) on port `7860`.
- **Comprehensive Evaluation & Safety Framework**: Includes 4-day benchmark notebooks and test suites covering retrieval optimization, grounded generation, citation compliance, and refusal guardrails.

---

## 🏗️ System Architecture

```
                                  +-----------------------+
                                  |   Web UI (index.html) |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |   FastAPI Backend     |
                                  |     (port 7860)       |
                                  +-----------+-----------+
                                              |
                        +---------------------+---------------------+
                        |                                           |
                        v                                           v
         +-----------------------------+             +-----------------------------+
         |   Hybrid Retriever Engine   |             |   Specialized Lab Engine    |
         |  ChromaDB Dense + BM25 RRF  |             | Assessment & Risk Logic     |
         +--------------+--------------+             +-----------------------------+
                        |
                        v
         +-----------------------------+
         | Citation-Grounded Generator |
         |   Groq / Llama-3 / GPT-OSS  |
         +-----------------------------+
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Environment Setup
Clone the repository and activate a virtual environment:

```bash
git clone https://github.com/nadaakhatab/medflow.git
cd medflow

# Create and activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
CONFIDENCE_THRESHOLD=0.50
PORT=7860
```

### 4. Running the Application

#### Option A: One-Click Desktop Launcher (Windows)
Double-click `start_medflow.bat` or execute in PowerShell:
```powershell
.\start_medflow.ps1
```
*This launches the backend, checks system health, and automatically opens `http://127.0.0.1:7860` in your web browser.*

#### Option B: Public HTTPS Tunnel Launcher
To generate a secure public HTTPS URL (powered by Cloudflare Tunnel):
```cmd
start_medflow_public.bat
```

#### Option C: Direct Python Execution
```bash
python run_app.py
```

---

## 📡 API Documentation

When the application is running, full interactive OpenAPI documentation is available at `http://127.0.0.1:7860/docs`.

### Primary API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Medflow Single-Page Application |
| `GET` | `/health` | System readiness and health check probe |
| `POST` | `/api/v1/query` | Hybrid RAG query synthesis with verifiable citations |
| `POST` | `/api/v1/interpret-labs` | Diagnostic evaluation of thyroid panel inputs |
| `GET` | `/api/v1/search-docs` | Semantic vector and BM25 document search |
| `POST` | `/api/v1/upload-pdf` | Upload and index custom PDF medical documents |
| `GET` | `/api/v1/imported-documents` | Retrieve list of user-imported documents |

---

## 📂 Project Structure

```
medflow/
├── backend/                   # FastAPI REST services, database models, and auth
│   ├── services/              # Medflow20 core engine wrapper
│   ├── auth.py                # Authentication & JWT handler
│   ├── main.py                # Application entry point & route handlers
│   ├── models.py              # SQLAlchemy database schemas
│   └── pdf_processor.py       # PDF extraction & text cleaner
├── medflow20/                 # Medflow20 core RAG engine & study benchmarks
│   ├── data/                  # Official ATA guidelines & medical brochures (11 PDFs)
│   ├── day1/                  # Day 1: Document Ingestion & Baseline
│   ├── day2/                  # Day 2: Retrieval Optimization & Benchmarking
│   ├── day3/                  # Day 3: Grounded Generation & Citation Compliance
│   ├── day4/                  # Day 4: Responsible AI, Safety & Risk Classifier
│   ├── evaluation/            # Master Ground Truth datasets & test benchmarks
│   ├── rag_pipeline.py        # Standalone RAG execution pipeline
│   └── requirements.txt       # Engine-specific dependencies
├── index.html                 # Modern responsive SPA Web Interface
├── run_app.py                 # Single-port unified service launcher
├── start_medflow.bat          # Desktop batch launcher
├── start_medflow.ps1          # Desktop PowerShell launcher
├── start_medflow_public.bat   # Public HTTPS tunnel launcher
├── Dockerfile                 # Docker container specification
├── docker-compose.yml         # Compose deployment configuration
└── requirements.txt           # Master project dependencies
```

---

## 📊 Empirical Benchmarks & Evaluation

### Day 2 Retrieval Optimization Scorecard (16 Ground Truth Questions)
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (L2 Normalized, Cosine Space)
- **Chunking Strategy**: Section-Aware Context-Enriched (200 tokens)
- **Hit@1**: `0.5625` (+200.0% improvement over baseline)
- **Hit@4**: `0.8750` (+55.6% improvement over baseline)
- **Precision@4**: `0.5312` (+82.1% improvement over baseline)
- **MRR**: `0.7031` (+82.4% improvement over baseline)
- **Mean Latency**: `20.53 ms` (-36.1% latency reduction)

### Day 3 Grounding & Citation Compliance
- **Schema Enforcement**: 100% compliant with JSON Schema validation.
- **Citation Integrity**: 100% verified source matching against retrieved document chunks.
- **Refusal Precision**: 10/10 out-of-domain queries successfully triggered safe refusal.

---

## 🧪 Automated Testing

Run the automated test suites:

```bash
# Run Medflow20 study module tests
python -m unittest discover -s medflow20/day1 -p "test_*.py" -v
python -m unittest discover -s medflow20/day2 -p "test_*.py" -v
python -m unittest discover -s medflow20/day3 -p "test_*.py" -v
python -m unittest discover -s medflow20/day4 -p "test_*.py" -v
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
