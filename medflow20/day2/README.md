# Day 2 — Retrieval Optimization & Evaluation

## 1. Overview & Objective
Day 2 rigorously evaluates and optimizes the retrieval pipeline using empirical metrics:
* **Ground Truth Dataset:** 16 clinically curated questions with exact target guidelines and page ranges.
* **Top-K Optimization:** Empirically validating $K=4$ against $K=3$ and $K=5$ (balancing recall vs. context noise).
* **Chunking Experiments:** Comparing 200/0, 400/50, 600/100 token configurations plus a 500-character naive baseline.
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
* **Root `.env` / `.env.example`**: local secrets / safe template; real API keys are not committed.

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


---

## 4. Final Frozen Retrieval Scorecard

The agenda asks for measured **Precision@K** and an explicit retrieval trade-off. The saved `evaluation/top_k_validation_results.json` gives:

| Metric | K=3 | **K=4 (selected)** | K=5 |
|---|---:|---:|---:|
| Hit@K | 0.8125 | **0.8750** | 0.8750 |
| Precision@K | **0.5417** | **0.5312** | 0.5000 |
| MRR | 0.6875 | **0.7031** | 0.7031 |
| Noise ratio | 45.8% | **46.9%** | 50.0% |

**Why K=4?** It increases evidence coverage over K=3 (Hit@K 81.25% → 87.50%). Moving to K=5 adds no Hit@K gain, lowers precision, and raises retrieval noise. Therefore K=4 is the best measured coverage/noise operating point for the frozen system.

Judge-ready values:

```text
Precision@3 = 54.17%
Precision@4 = 53.12%
Precision@5 = 50.00%
Hit@4       = 87.50%
MRR         = 70.31%
```

Run the saved-artifact summary without rebuilding embeddings:

```bash
python day2/agenda_retrieval_summary.py
```

### Chunk-size decision

The agenda blueprint illustrates 400–800-token section-aware chunks, but the project treated chunk size as an **empirical hyperparameter**, not a mandatory constant. The saved chunking experiment showed the 200-token / 0-overlap configuration had the strongest Precision@3 among the tested token configurations:

```text
200/0   → P@3 0.5833
400/50  → P@3 0.4583
600/100 → P@3 0.3750
```

For that reason the 200-token configuration was frozen rather than changed later to match an illustrative slide range.
