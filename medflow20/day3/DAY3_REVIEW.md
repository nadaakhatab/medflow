# Day 3 Review — Grounded Generation & Citation

## Review scope

This review checks the Day 3 implementation that is actually present in this project, without replacing the Day 1/Day 2 retriever. The available Day 3 scope requires a grounded prompt, structured cited answers, a working insufficient-evidence/refusal path, and safe handling of malformed or fabricated model output.

## Requirement-to-code check

| Day 3 requirement | Implementation evidence | Review result |
|---|---|---|
| Retriever → LLM integration | `rag_pipeline.ask_clinical_question()` retrieves Top-K evidence then calls `generate_answer()` | PASS |
| Evidence-only prompting | `GROUNDING_SYSTEM_PROMPT` explicitly forbids outside medical knowledge and guessing | PASS |
| Structured response | Pydantic `ClinicalAnswer` + JSON Schema enforce `recommendation`, `evidence`, `citations`, `confidence` | PASS |
| Citation integrity | `validate_citations()` resolves model citations only against retrieved document/page metadata and removes fabricated sources | PASS, metadata-level |
| Insufficient-evidence behavior | Empty retrieval and below-threshold retrieval refuse before the LLM; malformed/invalid model output also fails closed to refusal | PASS |
| Deterministic schema safety | Missing clinical schema now raises an error instead of silently skipping validation | PASS after review hardening |
| Adversarial/refusal categories | 10 refusal categories exist and are exercised by the compliance suite | PASS as compliance coverage; not an end-to-end retrieval-calibrated precision estimate |

## Important distinction for judging

The Day 3 citation validator proves that a citation points back to retrieved metadata. It does **not** by itself prove that every generated factual claim is semantically supported by that passage. That second, independent post-hoc check is intentionally added in Day 4 as unsupported-claim detection and faithfulness scoring.

The 10 refusal cases in `test_day3_compliance.py` use controlled mock LLM outputs to validate the refusal schema/invariants. They should be described as **10/10 refusal compliance categories covered**, not as a measured end-to-end “refusal precision” until Day 4 runs the real retriever on answerable and unsupported sets and calibrates the threshold.

## Changes made during this review

1. JSON Schema enforcement was changed from fail-open to **fail-closed** when `response_schema.json` is missing.
2. The grounding system prompt is now sent once as a system message instead of being duplicated inside the human prompt.
3. `jsonschema` and `transformers` are explicit project dependencies because the project imports them directly.
4. No Day 1/Day 2 source files, ground-truth labels, frozen configuration, PDFs, or the original `chroma_db` were overwritten by this review.

## Remaining Day 3 limitation handled by Day 4

- Claim-level support/faithfulness is not a Day 3 metadata-citation check. Day 4 adds `claim_validator.py`, strict citation accuracy, faithfulness, threshold calibration, and a live post-hoc guard.

## Frozen-index reproducibility check

The Day 3 logic is configured for BGE-small + Top-K 4, but the persisted `thyroid_section_aware` collection bundled in this ZIP contains 1,696 embeddings while the frozen Day 2 result declares 1,470. Therefore an exact end-to-end claim that Day 3 is running on the frozen 200-token index should only be made after building/auditing the separate Day 2 index.

The shared config now accepts non-breaking environment overrides. After building the separate index, Windows users can run Day 3 against it without changing source code:

```bat
set PERSIST_DIR=chroma_db_day2_frozen
set COLLECTION_NAME=thyroid_day2_frozen
set TOP_K=4
python day3\day3_grounded_generation.py
```

This preserves the original `chroma_db/` and makes the Day 2 → Day 3 handoff reproducible.
