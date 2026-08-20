# Day 1 — Document Ingestion & Baseline Retrieval

## 1. Overview & Objective
Day 1 establishes the evidence ingestion foundation for MedFlow:
* **PDF Ingestion:** Parsing 11 clinical guidelines and brochures (202 extracted pages in the verified local run).
* **Text Cleaning:** Removing copyright footers, URL noise, and non-breaking spaces.
* **Day 1 Baseline Chunking:** Comparing Naive Recursive Chunking (500 characters / 50 overlap) against Section-Aware Chunking (550 characters / 70 overlap) enriched with clinical headings.
* **Important:** the later **Day 2 optimized/frozen retriever** is a separate configuration: **200 tokens / 0 overlap**. Day 1's character-based baseline is preserved for reproducibility and is not relabeled as the final configuration.
* **Vector Indexing:** Persisting embeddings into ChromaDB vector database using `BAAI/bge-small-en-v1.5` in cosine space.
* **Baseline Retrieval:** Running 6 clinical queries to verify recall and score distributions.

---

## 2. Directory Contents
* **`ingest.py`**: Core ingestion module containing `load_pdfs()`, `clean_medical_text()`, `naive_chunk_documents()`, `section_aware_chunk_documents()`, and `build_index()`.
* **`day1_pipeline.py`**: Standalone pipeline executing the Day 1 baseline retrieval on 6 clinical questions.
* **`day1_task1_ingestion.ipynb`**: Interactive Jupyter Notebook demonstrating each ingestion step.
* **`test_day1.py`**: Automated unit tests for text cleaning, page extraction, and chunking strategies.
* **`config.py`**: Local configuration resolving data directories and embedding models.
* **`.env.example` / root `.env`**: environment-variable template / local secrets. Real API keys are not part of the submitted source package.

---

## 3. How to Run Independently

### A. Run Pipeline Script:
```bash
python day1/day1_pipeline.py
```

### B. Run Automated Unit Tests:
```bash
python -m unittest day1/test_day1.py -v
```

### C. Run Interactive Notebook:
Open `day1/day1_task1_ingestion.ipynb` in Jupyter Lab or VS Code and run all cells.
