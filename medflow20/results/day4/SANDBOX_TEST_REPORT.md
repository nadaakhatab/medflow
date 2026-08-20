# Day 4 Integration Test Report

## Result

**38 source/unit/compliance tests passed** in the review environment, plus successful Python bytecode compilation, notebook schema validation, and a non-destructive Chroma index audit.

| Layer | Tests | Result | How it was verified here |
|---|---:|---|---|
| Day 1 | 3 | PASS | Real project PDFs parsed with `pypdf` through a minimal compatibility loader because the bundled environment is Windows-only |
| Day 2 | 2 | PASS | Native dependency-light tests; frozen config + 16-item ground truth validated |
| Day 3 | 17 | PASS | Real Pydantic/JSON Schema/corpus tests; only `langchain_core.Document` was compatibility-stubbed for unit execution |
| Day 4 | 16 | PASS | Native pure-Python metric, calibration, guardrail and compliance tests |
| Static compilation | — | PASS | `python -m compileall -q .` |
| Day 3 notebook | — | PASS | `nbformat.validate` |
| Day 4 notebook | — | PASS | `nbformat.validate` |
| Persisted index audit | — | PASS as audit | Read-only audit completed; mismatch found and reported |

## Index finding

Frozen Day 2 configuration declares **1,470** indexed chunks. The bundled persisted `thyroid_section_aware` collection contains **1,696** embeddings. The original database was not modified.

Because of that mismatch, the Day 4 evaluator correctly blocks an unlabeled full benchmark on this store. This prevents accidentally presenting Day 4 metrics as comparable to the frozen Day 2 benchmark when they are not based on the same index.

## Why the full LLM/vector runtime was not executed in this review environment

The uploaded project contains a Windows virtual environment. The review runtime is Linux and cannot safely execute its compiled Windows Chroma/Torch/LangChain wheels. No internet install was used to silently replace the user's environment.

Therefore, final real threshold/citation/faithfulness numbers are intentionally **not fabricated**. The project includes a non-destructive frozen-index builder and the exact full-run commands for the user's Windows environment.

## Required local final run

From the project root on the user's machine:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
python verify_project.py
python day4\frozen_index_builder.py
python day4\evaluate_day4.py --full --persist-dir chroma_db_day2_frozen --collection thyroid_day2_frozen
```

If the separate build produces a count different from 1,470, stop and investigate before presenting results.
