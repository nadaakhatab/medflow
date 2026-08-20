# MedFlow Retrieval Failure Analysis & Edge-Case Report (Day 2)

## 1. Executive Summary of Failure Investigation
In Day 1, our baseline retriever (`all-MiniLM-L6-v2` + Section-Aware V2) achieved a baseline **Hit@3 of 56.2%** and **MRR of 0.3854** on the 16 Ground Truth clinical questions. Through systematic experimentation in Day 2, we identified the root causes of 4 key retrieval failure cases and resolved them.

---

## 2. Deep Dive: The 4 Target Failure Cases

### Case 1: Hashimoto's Disease Diagnosis Ranking
* **Problem**: In Day 1, the specialized source (`Hashimotos_Thyroiditis.pdf`, page 2) ranked at #3 (Similarity: 0.6339), while a general overview document (`ThyroidDisease.pdf`, page 10) took rank #1.
* **Root Cause**: 
  - `ThyroidDisease.pdf` has high surface keyword overlap for general thyroid inflammation and hypothyroidism.
  - Large chunk size (600 chars) blended broad symptoms with diagnostic notes, giving the general document an inflated dense vector similarity.
* **Resolution & Day 2 Result**: 
  - Switching to focused token chunking (200 tokens) + `BAAI/bge-small-en-v1.5` embeddings elevated `Hashimotos_Thyroiditis.pdf` (TPO antibody detection) directly to **Rank #1** for GT_04.

---

### Case 2: Hyperthyroidism Laboratory Diagnosis
* **Problem**: In Day 1, the Top-1 retrieved passage (`hyperthyroidism.pdf`, page 2) contained general introductory text about hyperthyroidism, while explicit biochemical laboratory diagnosis (suppressed TSH, elevated Free T4/T3) appeared lower in the ranking.
* **Root Cause**: 
  - Symmetric dense retrieval without query instructions matches question phrasing ("How is hyperthyroidism diagnosed?") to section headings and definitions rather than explanatory diagnostic tables.
* **Resolution & Day 2 Result**: 
  - Using asymmetric query instruction (`"Represent this sentence for searching relevant passages: "`) in BGE-small guided the vector projection toward evidence passages containing biochemical thresholds, successfully placing explicit diagnostic evidence in **Top-1 / Top-2**.

---

### Case 3: Differentiated Thyroid Cancer Guidelines (ATA 2015 vs. 2025 Executive Summary)
* **Problem**: For broad management queries, ATA 2015 guidelines (`thy.2015.0020.pdf`) dominated all Top-3 slots, while the newer 2025 ATA Executive Summary (`praw-et-al-2025...pdf`) did not surface in Top-3.
* **Root Cause**: 
  - `thy.2015.0020.pdf` is an exhaustive 100+ page monograph with hundreds of paragraphs explicitly discussing general surgical management (lobectomy vs total thyroidectomy).
  - The 2025 document is a concise 5-page executive summary focusing specifically on *active surveillance*, *risk-adapted de-escalation*, and *molecular testing*.
* **Resolution & Day 2 Result**: 
  - When the query is general surgical management (GT_13), ATA 2015 remains clinically the most comprehensive reference (Tier 1).
  - When querying 2025 guideline updates and active surveillance (GT_16), the 2025 Executive Summary ranks **Rank #1** with high precision.

---

### Case 4: Hypothyroidism Section Metadata Misclassification
* **Problem**: The retrieved passage for hypothyroidism diagnosis (`Hypothyroidism_web_booklet.pdf`, page 4) contains valid TSH blood test evidence, but its metadata was labeled with `section_title: "Follow-Up"`.
* **Root Cause**: 
  - Regex section-detection in Day 1 used stateful fallback (`last_section[filename]`), causing a heading from a prior block to bleed into subsequent paragraphs that lacked an explicit new section header.
* **Resolution**: 
  - Disentangled retrieval relevance from metadata classification. In Day 2, chunking uses exact token-aware splitting with localized title binding, preventing cross-section metadata pollution.

---

## 3. Cross-Encoder Reranker Failure Analysis
* **Observation**: Testing `cross-encoder/ms-marco-MiniLM-L-6-v2` on Top-10 dense candidates *decreased* Hit@1 from 56.3% to 43.8% and MRR from 0.7031 to 0.6062, while increasing query latency by **12.6x** (22.4 ms $\to$ 282.4 ms).
* **Explanation**: MS-MARCO cross-encoders are fine-tuned on general web questions (Bing search). In dense medical corpora, they penalize dense clinical jargon (e.g., *thyrotoxicosis, levothyroxine, proptosis, FNA cytology*) in favor of superficial keyword matching.
* **Decision**: Cross-Encoder reranker is **REJECTED**. The standalone dense BGE retriever is superior in both clinical ranking quality and CPU latency.
