"""Explicit Day 4 input-risk classification for the live MedFlow wrapper.

The hackathon agenda asks for an input-risk stage before retrieval.  This module
keeps that stage transparent and deterministic with three labels:

- ALLOWED: in-scope, general thyroid-guideline question.
- NEEDS_CAUTION: in-scope but patient-specific / dosing / urgent wording; retrieval
  may continue, but the UI should visibly show caution and the normal evidence
  guardrails still apply.
- REFUSE_REDIRECT: prompt-injection or clearly out-of-scope request for this
  thyroid-only prototype.

This classifier is intentionally conservative and is not a medical triage model.
"""
from __future__ import annotations

import re
from typing import Dict

ALLOWED = "ALLOWED"
NEEDS_CAUTION = "NEEDS_CAUTION"
REFUSE_REDIRECT = "REFUSE_REDIRECT"

_THYROID_TERMS = {
    "thyroid", "hypothyroid", "hypothyroidism", "hyperthyroid", "hyperthyroidism",
    "hashimoto", "hashimoto's", "graves", "thyrotoxicosis", "tsh", "t3", "t4",
    "thyroxine", "levothyroxine", "thyroidectomy", "thyroglobulin", "nodule",
    "nodules", "goiter", "goitre", "orbitopathy", "ophthalmopathy", "fna",
}

_INJECTION_PATTERNS = (
    r"\bignore\b.{0,40}\b(previous|prior|system)\b.{0,30}\binstruction",
    r"\bno citations? needed\b",
    r"\bbypass\b.{0,30}\b(guard|safety|instruction|rule)",
    r"\breveal\b.{0,30}\b(system prompt|hidden prompt|instructions?)",
)

_PATIENT_SPECIFIC_PATTERNS = (
    r"\b(my|me|i|we|our|mother|father|grandmother|grandfather|wife|husband|child|patient)\b",
    r"\bwhat (dose|dosage)\b",
    r"\bhow much\b",
    r"\bshould (i|he|she|they|my)\b",
    r"\b(today|right now|immediately)\b",
    r"\bpregnan(?:t|cy)\b",
)


def _has_any(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE | re.DOTALL) for p in patterns)


def classify_input_risk(query: str) -> Dict[str, str]:
    """Return an agenda-aligned input-risk label plus a short UI-safe reason."""
    text = re.sub(r"\s+", " ", str(query or "")).strip().lower()
    if not text:
        return {
            "label": REFUSE_REDIRECT,
            "reason": "Empty query; no clinical question was provided.",
            "ui_guidance": "Refuse and ask for an in-scope thyroid-guideline question.",
        }

    if _has_any(_INJECTION_PATTERNS, text):
        return {
            "label": REFUSE_REDIRECT,
            "reason": "Prompt-injection / instruction-bypass pattern detected.",
            "ui_guidance": "Refuse; do not bypass citation or safety requirements.",
        }

    in_scope = any(term in text for term in _THYROID_TERMS)
    patient_specific = _has_any(_PATIENT_SPECIFIC_PATTERNS, text)

    if not in_scope:
        return {
            "label": REFUSE_REDIRECT,
            "reason": "Question is outside the thyroid-guideline scope of this hackathon prototype.",
            "ui_guidance": "Redirect to an appropriate clinician or evidence source; do not guess.",
        }

    if patient_specific:
        return {
            "label": NEEDS_CAUTION,
            "reason": "In-scope thyroid question contains patient-specific, dosing, pregnancy, or urgent wording.",
            "ui_guidance": "Continue only with guideline-grounded evidence and show a caution indicator.",
        }

    return {
        "label": ALLOWED,
        "reason": "General in-scope thyroid-guideline question.",
        "ui_guidance": "Proceed to retrieval and normal Day 4 evidence checks.",
    }
