# Day 2 — Retrieval Optimization & Evaluation

## 1. Overview & Objective
Day 2 rigorously evaluates and optimizes the retrieval pipeline using empirical metrics:
* **Ground Truth Dataset:** 16 clinically curated questions with exact target guidelines and page ranges.
* **Top-K Optimization:** Empirically validating $K=4$ against $K=3$ and $K=5$ (balancing recall vs. context noise).
* **Chunking Experiments:** Comparing 100 vs. 200 vs. 500 token chunk sizes.
* **Embedding Model Benchmarks:** Benchmarking `all-MiniLM-L6-v2` vs. `BAAI/bge-small-en-v1.5` vs. `PubMedBERT`.
* **Cross-Encoder Re-ranker Analysis:** Testing `ms-marco-MiniLM-L-6-v2` and evaluating the latency/precision trade-off.
* **Frozen Optimal Configuration:** Storing the verified winning configuration in `results/best_retrieval_config.json`.

---

## 2. Directory Contents
* **`evaluate_retrieval.py`**: Evaluation engine computing Hit@1, Hit@3, Hit@5, Precision@k, and MRR.
* **`chunk_experiments.py`**: Automated comparison across chunk size variations.
* **`embedding_benchmark.py`**: Benchmark runner evaluating the 3 embedding models.
* **`reranker_experiment.py`**: Cross-Encoder evaluation script.
* **`validate_top_k.py`**: Empirical top-k trade-off script.
* **`day2_retrieval_optimization.ipynb`**: Interactive Jupyter Notebook walking through the optimization experiments.
* **`test_day2.py`**: Automated unit tests for Ground Truth validation and frozen configuration integrity.
* **`config.py`**: Local configuration resolving paths and retrieval parameters.
* **`.env`**: Local environment variables.

---

## 3. How to Run Independently

### A. Run Retrieval Evaluation:
```bash
python day2/evaluate_retrieval.py
```

### B. Run Automated Unit Tests:
```bash
python -m unittest day2/test_day2.py -v
```

### C. Run Interactive Notebook:
Open `day2/day2_retrieval_optimization.ipynb` in Jupyter Lab or VS Code and execute the evaluation cells.
