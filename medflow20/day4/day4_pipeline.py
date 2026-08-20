"""Live Day 4 wrapper: retrieval -> Day 3 generation -> post-hoc safety guard."""
from __future__ import annotations

from typing import Any, Dict, Optional
import json

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from day4 import config
    from day4.safety_guardrails import apply_posthoc_guard
    from day4.threshold_calibration import threshold_for_query
    from day4.index_audit import audit_index
    from day4.risk_classifier import classify_input_risk, REFUSE_REDIRECT
else:
    from . import config
    from .safety_guardrails import apply_posthoc_guard
    from .threshold_calibration import threshold_for_query
    from .index_audit import audit_index
    from .risk_classifier import classify_input_risk, REFUSE_REDIRECT


def _saved_calibrated_threshold(query: str) -> float:
    """Use the latest Day 4 calibration when available, otherwise fall back safely."""
    path = config.RESULTS_DIR / "threshold_calibration.json"
    if not path.exists():
        return float(config.DEFAULT_CONFIDENCE_THRESHOLD)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stratified = payload.get("stratified_thresholds") or {
            "global_threshold": payload.get("selected_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD),
            "family_thresholds": {},
        }
        return float(threshold_for_query(query, stratified))
    except Exception:
        # A corrupt/missing calibration artifact must not break the wrapper. Keep the
        # conservative pre-calibration fallback explicit instead of guessing.
        return float(config.DEFAULT_CONFIDENCE_THRESHOLD)


def ask_safe_clinical_question(
    query: str,
    vector_store: Optional[Any] = None,
    top_k: int | None = None,
    llm: Optional[Any] = None,
    confidence_threshold: float | None = None,
    return_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Run MedFlow with the Day 4 live claim/citation safety net.

    The original Day 1-3 pipeline is left untouched. Day 4 wraps it instead of
    replacing it, which makes the change reversible and reduces regression risk.
    Diagnostics are returned by default so the evidence-strength language and visible
    Responsible-AI disclaimer can be shown in the live demo.
    """
    risk = classify_input_risk(query)
    if risk["label"] == REFUSE_REDIRECT:
        refusal = {
            "recommendation": "I can only answer questions that are supported by the indexed thyroid guidelines.",
            "evidence": "",
            "citations": [],
            "confidence": "insufficient",
        }
        result = {
            "answer": refusal,
            "risk": risk,
            "safety": {
                "safe_to_return_original": False,
                "guard_triggered": True,
                "repair_applied": False,
                "guard_reasons": ["input risk classification: REFUSE_REDIRECT"],
                "evidence_strength": "insufficient",
                "language_guidance": risk["ui_guidance"],
                "disclaimer": config.CLINICAL_DISCLAIMER,
            },
        }
        return result if return_diagnostics else refusal

    k = int(top_k or config.TOP_K)
    threshold = float(
        confidence_threshold
        if confidence_threshold is not None
        else _saved_calibrated_threshold(query)
    )
    if vector_store is None:
        # Day 4 must run on the audited frozen Day 2 retriever, not the older live
        # collection whose chunk count differs from the frozen benchmark. Audit first
        # so a missing/mismatched index fails closed before loading heavy ML modules.
        audit = audit_index(config.FROZEN_DAY4_PERSIST_DIR, config.FROZEN_DAY4_COLLECTION_NAME)
        if not audit.get("index_matches_frozen_day2"):
            raise RuntimeError(
                "Audited frozen Day 2 index is not ready. Run "
                "`python day4/frozen_index_builder.py` before the Day 4 live pipeline."
            )

    from rag_pipeline import Chroma, get_embeddings, semantic_retrieval
    from generator import generate_answer

    if vector_store is None:
        embeddings = get_embeddings(config.EMBEDDING_MODEL_NAME)
        vector_store = Chroma(
            collection_name=config.FROZEN_DAY4_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(config.FROZEN_DAY4_PERSIST_DIR),
        )

    retrieved = semantic_retrieval(vector_store, query, top_k=k)
    generated = generate_answer(query, retrieved, llm=llm, confidence_threshold=threshold)
    guarded = apply_posthoc_guard(generated, retrieved, confidence_threshold=threshold)
    guarded["risk"] = risk
    return guarded if return_diagnostics else guarded["answer"]
