# Day 4 — Safety, Guardrails & Internal Evaluation

## Goal

Day 4 moves MedFlow from **cited answers** to **measurably safe answers**. It extends the existing Day 1–3 pipeline without replacing the frozen retrieval design or changing the Day 3 response schema.

The official Day 4 deliverables implemented here are:

1. Calibrated retrieval confidence threshold
2. Live unsupported-claim detection
3. Retrieval Precision@k, Citation Accuracy, and Faithfulness
4. Uncertainty language matched to evidence strength
5. Responsible-AI clinical disclaimer and fail-safe refusal behavior
6. Full internal evaluation log ready for the Day 5 demo

## Non-destructive architecture

```text
Clinical Question
      ↓
Explicit Input Risk Classification
(ALLOWED / NEEDS_CAUTION / REFUSE_REDIRECT)
      ├─ REFUSE_REDIRECT → Safe redirect/refusal
      └─ ALLOWED or NEEDS_CAUTION
                    ↓
Frozen Day 2 Retriever
BGE-small + 200 tokens + 0 overlap + Top-K 4
      ↓
Top-4 Evidence + Similarity Scores
      ↓
Calibrated Threshold Gate
   ├─ below threshold → Refuse
   └─ above threshold → Day 3 Grounded Generator
                              ↓
                       JSON + Citations
                              ↓
                   Day 4 Post-hoc Guard
                 ┌──────────┴──────────┐
                 ↓                     ↓
          Citation Accuracy      Claim Faithfulness
                 └──────────┬──────────┘
                            ↓
                  Unsupported claim?
                    ├─ Unsupported details → Repair supported content first
                    ├─ No supported content → Refuse
                    └─ Verified content → Return answer
                            ↓
                    Evaluation Logger
```

Day 4 is a **wrapper layer**. The original `rag_pipeline.py`, Day 1 code, Day 2 experiments, PDFs, and original Chroma database are not deleted or overwritten.

## Files

| File | Purpose |
|---|---|
| `RESPONSIBLE_AI_CHECKLIST.md` | Clinical Responsible-AI implementation checklist and team sign-off |
| `config.py` | Reads the frozen Day 2 configuration and centralizes Day 4 safety targets |
| `index_audit.py` | Non-destructively checks whether the persisted Chroma collection matches the frozen Day 2 index |
| `frozen_index_builder.py` | Optional separate builder for the exact 200-token / 0-overlap Day 2 index; never overwrites `chroma_db/` |
| `risk_classifier.py` | Explicit agenda-aligned input-risk classification: ALLOWED / NEEDS_CAUTION / REFUSE_REDIRECT |
| `threshold_calibration.py` | Sweeps thresholds using answerable + unsupported queries and selects a safety-constrained operating point |
| `claim_validator.py` | Splits generated answers into claims and checks retrieved support, numbers, units, and negation polarity |
| `evaluation_metrics.py` | Precision@4, Citation Accuracy, Faithfulness, refusal/safety metrics |
| `safety_guardrails.py` | Post-hoc guard, honest uncertainty language, visible disclaimer |
| `day4_pipeline.py` | Live safe wrapper around the Day 3 generation pipeline |
| `evaluate_day4.py` | Full Day 4 evaluator and artifact logger |
| `day4_safety_evaluation.ipynb` | Judge/demo notebook |
| `test_day4.py` | Unit tests for metrics, calibration, and guard behavior |
| `test_day4_compliance.py` | Competition-checklist and non-destructive architecture tests |

## Metric definitions

### Retrieval Precision@K (frozen live window: K=4)

```text
relevant retrieved chunks in Top-4 / 4
```

Day 4 deliberately uses **K=4**, because that is the frozen Day 2 retrieval window. Day 2 also reports the agenda-facing comparison: **P@3=0.5417, P@4=0.5312, P@5=0.5000**; K=4 was selected because it improved Hit@K over K=3 while K=5 added no hit-rate gain and more noise.

### Citation Accuracy

```text
correct citations / total citations produced
```

A Day 4 citation is counted as correct only if it:

- resolves to a retrieved document,
- resolves to the exact normalized page,
- has a compatible section when a section is available,
- and the cited passage supports at least one unit of the answer evidence.

This is intentionally stricter than merely checking that a filename exists.

### Faithfulness

```text
claims supported by retrieved text / total generated claims
```

Target: **>= 0.90** across answerable evaluation cases.

The claim validator is transparent and deterministic. It uses lexical support as a baseline, plus conservative checks for unsupported numbers/units and negation mismatches. It is a safety net, not a standalone medical fact checker.

### Additional safety metrics

Day 4 also logs:

- Answerability Accuracy
- Refusal Precision
- Refusal Recall
- Unsafe Accept Rate
- False Refusal Rate
- Unsupported Claim Count

These make the threshold trade-off defendable in front of judges.



## Explicit input-risk classification

Before the live Day 4 wrapper loads the retriever, `risk_classifier.py` assigns one of three visible states required by the safety workflow:

- **ALLOWED** — general in-scope thyroid-guideline question.
- **NEEDS_CAUTION** — in-scope but patient-specific, dosing, pregnancy, or urgent wording; retrieval may continue, but the UI must show caution and normal grounding controls still apply.
- **REFUSE_REDIRECT** — prompt injection or clearly out-of-scope request; the live wrapper refuses/redirects before generation.

This is a transparent hackathon guardrail, **not** a medical triage model.

## Threshold calibration

The threshold is **not copied from the slide deck** and is not hardcoded to 0.65 or 0.70. The evaluator collects the Top-1 retrieval score for:

- all 16 answerable Day 2 ground-truth queries, and
- all 10 unsupported/refusal cases.

It then sweeps candidate thresholds and selects an operating point under a clinical safety constraint (`MAX_UNSAFE_ACCEPT_RATE`, default 5%). The selected threshold and the full sweep are saved so the team can defend the choice quantitatively.

To address the Day 4 warning about using one threshold blindly for every query type, the evaluator also attempts **query-family calibration** (diagnosis/evaluation, treatment/management, dosage, procedure, symptoms, other). A family-specific threshold is used only when that family has enough answerable **and** unsupported labels; otherwise it explicitly falls back to the global threshold rather than overfitting a tiny subgroup.

## Frozen-index reproducibility status

Two Chroma collections may exist locally and they must not be confused:

```text
Legacy / diagnostic only:
  chroma_db / thyroid_section_aware = 1,696 chunks

Frozen Day 2 index used for Day 4 evaluation:
  chroma_db_day2_frozen / thyroid_day2_frozen = 1,470 chunks
```

The **1,470-chunk frozen index** matches the selected Day 2 configuration:

```text
BAAI/bge-small-en-v1.5
200 tokens
0 overlap
Top-K = 4
```

`verify_project.py` now audits the frozen collection explicitly so the older 1,696-chunk database does not create a misleading warning during final verification. The legacy database is not deleted because the project preserves prior experiment artifacts non-destructively.

If the frozen index ever needs to be reconstructed on a clean machine:

```bash
python day4/frozen_index_builder.py
```

Then verify it with:

```bash
python day4/evaluate_day4.py --audit-index \\
  --persist-dir chroma_db_day2_frozen \\
  --collection thyroid_day2_frozen
```

## How to run Day 4

### 1. Activate the existing environment

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run source/unit tests

```bash
python -m unittest day4/test_day4.py -v
python -m unittest day4/test_day4_compliance.py -v
```

### 3. Audit the index

```bash
python day4/evaluate_day4.py --audit-index \\
  --persist-dir chroma_db_day2_frozen \\
  --collection thyroid_day2_frozen
```

### 4. Calibrate threshold only

```bash
python day4/evaluate_day4.py --calibrate-only \
  --persist-dir chroma_db_day2_frozen \
  --collection thyroid_day2_frozen
```

### 5. Run the full internal evaluation

```bash
python day4/evaluate_day4.py --full \
  --persist-dir chroma_db_day2_frozen \
  --collection thyroid_day2_frozen
```

## Generated evaluation evidence

A real full run writes:

```text
results/day4/
├── index_audit.json
├── threshold_calibration.json
├── threshold_calibration_sweep.csv
├── day4_evaluation_log.json
├── day4_evaluation_log.csv
└── day4_evaluation_summary.json
```

No final Day 4 metric is pre-filled or fabricated. Numbers are created from the actual local corpus/index/model run.

## Responsible AI behavior

Every Day 4 diagnostic includes a visible statement that MedFlow does not replace clinical judgment. The live safety stack first classifies input risk, then applies the calibrated retrieval gate, and finally uses a **repair-before-refuse** post-hoc guard. It refuses rather than guessing when:

- retrieval is below the calibrated threshold,
- faithfulness falls below the target,
- citation accuracy fails the configured minimum,
- or a generated numerical/dosage claim is unsupported by retrieved evidence.

## Judge-ready 30-second explanation

> “Day 2 optimized retrieval and Day 3 constrained generation to cited evidence. Day 4 measures whether the system knows when it has enough evidence to answer. We calibrate the threshold on answerable versus unsupported queries, calculate Precision@4, Citation Accuracy and Faithfulness, and run a second post-hoc claim check. If a claim, citation, or numerical detail is unsupported, the safety layer fails closed to a refusal instead of letting a fluent answer through.”
