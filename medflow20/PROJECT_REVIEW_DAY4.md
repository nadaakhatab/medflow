# MedFlow Project Review Before Day 4

## What was preserved

The Day 4 work was added as a new layer. It does not delete or overwrite:

- Day 1 source files
- Day 2 experiment source files
- the 16-item thyroid ground truth
- the 10 refusal cases
- source PDFs
- the original `chroma_db/`
- the frozen `results/best_retrieval_config.json`

A SHA-256 pre-change manifest is stored in `results/day4/pre_day4_manifest.json`.

## Day 3 review result

The project contains the required Day 3 building blocks:

- evidence-only grounding prompt,
- structured JSON answer contract,
- schema invariants,
- metadata-bound citation validation,
- empty/low-score pre-generation refusal,
- malformed-output safe fallback,
- fabricated-citation refusal behavior,
- 10 refusal compliance categories.

Small safety hardening was applied before Day 4:

1. missing JSON Schema now fails closed instead of bypassing validation;
2. the grounding prompt is no longer duplicated in both system and human messages;
3. dependencies imported directly by the project are explicit in `requirements.txt`.

See `day3/DAY3_REVIEW.md`.

## Day 3 claim that needed clarification

The existing 10-case refusal compliance test uses controlled mock model refusal outputs. It validates response behavior but is not a measured end-to-end retrieval **Refusal Precision** result. Day 4 adds the real answerable-versus-unsupported threshold calibration and refusal metrics.

## Day 4 implementation

Implemented:

- calibrated threshold sweep,
- unsafe-accept / false-refusal trade-off metrics,
- Precision@4,
- strict Day 4 Citation Accuracy,
- claim-level Faithfulness,
- unsupported numerical/dosage detection,
- post-hoc fail-closed safety wrapper,
- uncertainty language guidance,
- visible Responsible-AI disclaimer,
- full evaluation logger,
- audit and test suites,
- non-destructive exact-index builder.

## Reproducibility finding

`results/best_retrieval_config.json` declares **1,470** chunks for the frozen 200-token/0-overlap Day 2 index. A direct, read-only Chroma SQLite audit of the bundled `thyroid_section_aware` collection found **1,696** embeddings.

This means the current persisted DB is not byte/logically demonstrated to be the same index used for the frozen Day 2 benchmark. Day 4 therefore blocks unlabeled “frozen-comparable” evaluation on it by default. A separate exact index can be built in `chroma_db_day2_frozen/` without altering the original database.

## Full runtime verification note

The ZIP includes a Windows `.venv`. The review sandbox is Linux, so its compiled Windows wheels cannot be used as a real LangChain/Chroma/Torch runtime here. The project is therefore tested in two layers:

- syntax/unit/logic/compliance tests in the review sandbox;
- a provided command sequence for the real full-stack run inside the project's Windows environment.

No measured Day 4 score is invented when the full model/index runtime has not executed.


## Review test status

The final review run passed **38** source/unit/compliance tests (Day 1: 3, Day 2: 2, Day 3: 17, Day 4: 16), plus Python compilation and Day 3/Day 4 notebook validation. See `results/day4/SANDBOX_TEST_REPORT.md` for the environment caveat and exact runtime boundary.
