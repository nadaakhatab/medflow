# MedFlow — Hackathon Agenda Compliance Review

Status date: 2026-08-19

This file maps the current source package to the supplied hackathon agenda. It separates **implemented Days 1–4** from **Day 5 presentation work that is intentionally still pending**.

## Day 1 — Research, Scope & Ingestion

| Agenda item | Status | Evidence in project |
|---|---|---|
| Narrow clinical scope | PASS | Thyroid diseases / official thyroid guidelines |
| Official public guideline PDFs | PASS | 11 PDFs in local `data/` corpus |
| PDF cleaning | PASS | `day1/ingest.py::clean_medical_text` |
| Chunking strategy | PASS | Day 1 baseline 500-char naive and 550-char section-aware; Day 2 later optimizes token chunking |
| Metadata | PASS | document/filename, page number, section title, chunk id, strategy |
| Searchable vector DB | PASS | Chroma + normalized BGE embeddings |

Verified local extraction: **202 pages**. Do not relabel the Day 1 character baseline as the final 200-token configuration.

## Day 2 — Retrieval Optimization

| Agenda item | Status | Result |
|---|---|---:|
| Labeled evaluation set | PASS | 16 queries |
| Chunk experiments | PASS | 200/0, 400/50, 600/100 tokens + character baseline |
| Embedding comparison | PASS | MiniLM, BGE-small, PubMedBERT experiments |
| Reranker experiment | PASS | measured and rejected on quality/latency trade-off |
| Precision@K | PASS | P@3 54.17%, P@4 53.12%, P@5 50.00% |
| Hit@K | PASS | Hit@3 81.25%, Hit@4 87.50%, Hit@5 87.50% |
| MRR | PASS | 70.31% at selected K=4 |
| Frozen retriever | PASS | BGE-small, 200 tokens, 0 overlap, K=4, 1,470 chunks |

**K=4 rationale:** Hit@K improves from 81.25% at K=3 to 87.50% at K=4. K=5 gives no additional Hit@K gain, reduces precision, and raises noise to 50%.

The agenda's 400–800-token example is treated as an illustrative range; the project keeps the empirically stronger tested 200-token setting.

## Day 3 — Grounded Generation & Citation

| Agenda item | Status |
|---|---|
| Direct recommendation | PASS |
| Supporting retrieved evidence | PASS |
| Document/page citation | PASS |
| Structured output | PASS |
| Confidence / insufficient evidence | PASS |
| Refusal on unsupported evidence | PASS |
| Schema validation | PASS |
| Citation integrity checks | PASS |

## Day 4 — Safety, Guardrails & Internal Evaluation

| Agenda item | Status |
|---|---|
| Input risk classification | PASS — explicit ALLOWED / NEEDS_CAUTION / REFUSE_REDIRECT |
| Calibrated retrieval threshold | PASS — global 0.72, plus conservative family thresholds where labels suffice |
| Unsupported claim detection | PASS |
| Numeric/dosage guard | PASS |
| Negation guard | PASS |
| Citation accuracy | PASS |
| Faithfulness | PASS |
| Repair-before-refuse safety layer | PASS |
| Responsible-AI disclaimer | PASS |
| Internal benchmark | PASS — 26 queries (16 supported + 10 refusal) |

Latest reported full evaluation on the audited frozen index:

```text
Final Citation Accuracy = 100%
Final Faithfulness       = 100%
Answerability Accuracy   = 100%
Answer Precision         = 100%
Answer Recall            = 100%
Refusal Precision        = 100%
Refusal Recall           = 100%
Unsafe Accept Rate       = 0%
False Refusal Rate       = 0%
```

Use the qualifier **“on our 26-query internal evaluation set”** whenever presenting these numbers. They are not a claim of universal clinical accuracy.

## Day 5 — Final Presentation & Judge Evaluation

Not yet marked complete. The remaining agenda-facing deliverables are:

1. Final integrated UI / Evidence Panel.
2. Show recommendation, evidence excerpt, document, page, confidence, and safety/risk status.
3. Three predefined live demo cases:
   - Case A: supported success.
   - Case B: complex / multi-step synthesis.
   - Case C: safe refusal / out-of-scope.
4. Frozen prototype and reproducible demo commands.
5. Judge pitch slides / architecture diagram.
6. Clinical safety disclaimer visible in the interface.
7. Scalability roadmap: larger multi-guideline corpus, version monitoring, external clinical validation, independent held-out evaluation, human-in-the-loop review.

## Presentation wording to avoid

Do **not** say:

- “MedFlow is 100% clinically accurate.”
- “Production ready.”
- “Validated for clinical deployment.”

Prefer:

> “MedFlow is a hackathon prototype. On our internal 26-query safety and grounding evaluation set, the final guarded outputs achieved 100% citation accuracy and faithfulness with zero unsafe accepts. Larger independent and clinical validation is still required.”
