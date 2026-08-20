"""Live Day 4 post-generation guardrails layered on top of Day 3.

The guard follows a repair-before-refuse policy:
- below-threshold retrieval still refuses deterministically;
- unsupported claims are removed instead of discarding a fully useful answer;
- citations are rebuilt from the retrieved chunks that actually supported the
  surviving claims, rather than trusting LLM-supplied page metadata;
- final faithfulness/citation metrics are recomputed on the repaired answer.
"""
from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

try:
    from .config import (
        CLAIM_SUPPORT_THRESHOLD,
        CLINICAL_DISCLAIMER,
        MIN_CITATION_ACCURACY,
        TARGET_FAITHFULNESS,
    )
    from .evaluation_metrics import citation_accuracy, faithfulness
    from .claim_validator import evaluate_claim
except ImportError:
    from config import CLAIM_SUPPORT_THRESHOLD, CLINICAL_DISCLAIMER, MIN_CITATION_ACCURACY, TARGET_FAITHFULNESS
    from evaluation_metrics import citation_accuracy, faithfulness
    from claim_validator import evaluate_claim

DEFAULT_REFUSAL_MESSAGE = "I couldn't find enough information in the indexed guideline to answer this confidently."


def uncertainty_language(top_score: float | None, threshold: float, faithfulness_score: float, citation_score: float) -> Dict[str, str]:
    if top_score is not None and top_score < threshold:
        return {
            "evidence_strength": "insufficient",
            "language_guidance": "Refuse — do not soften a below-threshold result into a guess.",
        }
    margin = (top_score - threshold) if top_score is not None else 0.0
    if faithfulness_score >= 0.95 and citation_score >= 1.0 and margin >= 0.10:
        return {
            "evidence_strength": "strong",
            "language_guidance": "The guideline recommends / reports…",
        }
    if faithfulness_score >= 0.90 and citation_score >= 0.80 and margin >= 0.03:
        return {
            "evidence_strength": "partial",
            "language_guidance": "The guideline suggests, though the retrieved evidence may not directly address every detail…",
        }
    return {
        "evidence_strength": "weak",
        "language_guidance": "Limited evidence found; consider consulting the full guideline before relying on this point.",
    }


def _top_score(retrieved_chunks: Sequence[Any]) -> float | None:
    scores = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            raw = chunk.get("similarity_score")
            if raw is None and isinstance(chunk.get("metadata"), dict):
                raw = chunk["metadata"].get("similarity_score")
        else:
            raw = getattr(chunk, "metadata", {}).get("similarity_score")
        try:
            if raw is not None:
                scores.append(float(raw))
        except (TypeError, ValueError):
            pass
    return max(scores) if scores else None


def _chunk_meta_content(chunk: Any) -> tuple[Dict[str, Any], str]:
    if isinstance(chunk, dict):
        meta = dict(chunk.get("metadata") or {})
        for key in (
            "document_name", "filename", "source", "page_number", "page",
            "section_title", "section", "chunk_id", "similarity_score",
        ):
            if key in chunk and key not in meta:
                meta[key] = chunk[key]
        content = str(
            chunk.get("retrieved_passage") or chunk.get("page_content")
            or chunk.get("text") or chunk.get("content") or ""
        )
    else:
        meta = dict(getattr(chunk, "metadata", {}) or {})
        content = str(getattr(chunk, "page_content", "") or "")
    return meta, content


def _page(meta: Mapping[str, Any]) -> int | None:
    if meta.get("page_number") is not None:
        raw = meta.get("page_number")
    elif meta.get("page") is not None:
        try:
            return int(meta.get("page")) + 1
        except (TypeError, ValueError):
            return None
    else:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _citation_from_chunk(chunk: Any) -> Dict[str, Any] | None:
    meta, _ = _chunk_meta_content(chunk)
    raw_doc = meta.get("document_name") or meta.get("filename") or meta.get("source")
    page = _page(meta)
    if not raw_doc or page is None:
        return None
    section = meta.get("section_title") or meta.get("section") or "General Content"
    return {
        "document": os.path.basename(str(raw_doc)),
        "section": str(section),
        "page": page,
    }


def _canonical_duplicate_index(chunks: Sequence[Any], index: int) -> int:
    """Prefer provenance-consistent metadata when duplicate PDF content exists.

    The frozen corpus can contain byte-identical PDFs under different filenames.
    We do not mutate or rebuild the frozen index here. When identical retrieved
    passage text is present more than once, prefer the filename whose year agrees
    with a year explicitly present in the passage (for example a 2025 executive
    summary accidentally stored under a 2016 filename).
    """
    if index < 1 or index > len(chunks):
        return index
    _, target_content = _chunk_meta_content(chunks[index - 1])
    normalized_target = re.sub(r"\s+", " ", target_content).strip()
    if not normalized_target:
        return index
    content_years = set(re.findall(r"\b(?:19|20)\d{2}\b", normalized_target))
    if not content_years:
        return index

    candidates = []
    for i, chunk in enumerate(chunks, start=1):
        meta, content = _chunk_meta_content(chunk)
        if re.sub(r"\s+", " ", content).strip() != normalized_target:
            continue
        raw_doc = meta.get("document_name") or meta.get("filename") or meta.get("source") or ""
        doc = os.path.basename(str(raw_doc))
        doc_years = set(re.findall(r"\b(?:19|20)\d{2}\b", doc))
        score = 10 * len(content_years & doc_years) - len(doc_years - content_years)
        candidates.append((score, -i, i))
    if not candidates:
        return index
    return max(candidates)[2]


def _dedupe_dicts(items: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = (item.get("document"), item.get("section"), item.get("page"))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _ensure_sentence(claim: str) -> str:
    claim = str(claim or "").strip()
    if not claim:
        return ""
    return claim if claim[-1:] in ".!?" else claim + "."


def _extractive_evidence(chunks: Sequence[Any], indexes: Sequence[int], max_chars_per_chunk: int = 1200) -> str:
    pieces = []
    seen = set()
    for idx in indexes:
        if idx in seen or idx < 1 or idx > len(chunks):
            continue
        seen.add(idx)
        _, content = _chunk_meta_content(chunks[idx - 1])
        text = re.sub(r"\s+", " ", content).strip()
        if not text:
            continue
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0].rstrip() + "…"
        pieces.append(text)
    return "\n\n".join(pieces)


def _empty_metric_payload(reason: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    citation = {
        "citation_accuracy": 1.0,
        "correct_citations": 0,
        "total_citations": 0,
        "claim_coverage": 1.0,
        "covered_claims": 0,
        "total_claims": 0,
        "details": [],
        "not_applicable_reason": reason,
    }
    faithful = {
        "faithfulness": 1.0,
        "supported_claims": 0,
        "total_claims": 0,
        "unsupported_claim_count": 0,
        "unsupported_claims": [],
        "claim_details": [],
        "not_applicable_reason": reason,
    }
    return citation, faithful


def _simplify_parenthetical_detail(claim: str) -> str:
    """Conservatively remove parenthetical detail for a second support check.

    LLMs sometimes add a non-essential identifier/qualifier such as ``(I-131)``
    or ``(<4 cm)`` to an otherwise supported statement. If that exact detail is
    absent from retrieved evidence, Day 4 should remove the detail rather than
    discard the whole clinically useful claim. Dosages outside parentheses are
    intentionally untouched and remain fail-closed.
    """
    simplified = re.sub(r"\([^()]*\)", "", str(claim or ""))
    simplified = re.sub(r"\s+", " ", simplified)
    simplified = re.sub(r"\s+([,.;:])", r"\1", simplified).strip(" ,;:")
    return simplified.strip()


def apply_posthoc_guard(
    answer: Mapping[str, Any],
    retrieved_chunks: Sequence[Any],
    confidence_threshold: float,
    target_faithfulness: float = TARGET_FAITHFULNESS,
    min_citation_accuracy: float = MIN_CITATION_ACCURACY,
    claim_support_threshold: float = CLAIM_SUPPORT_THRESHOLD,
) -> Dict[str, Any]:
    original = dict(answer)
    is_refusal = str(original.get("confidence", "")).lower() == "insufficient"
    score = _top_score(retrieved_chunks)

    # A deterministic Day 3 refusal is not a medical claim and must not be run
    # through the unsupported-claim checker.
    if is_refusal:
        citation, faithful = _empty_metric_payload("structured refusal")
        language = uncertainty_language(score, confidence_threshold, 1.0, 1.0)
        return {
            "answer": deepcopy(original),
            "safety": {
                "safe_to_return_original": True,
                "guard_triggered": False,
                "repair_applied": False,
                "guard_reasons": [],
                "top_similarity_score": score,
                "confidence_threshold": confidence_threshold,
                "raw_citation_accuracy": citation,
                "raw_faithfulness": faithful,
                "citation_accuracy": citation,
                "faithfulness": faithful,
                "unsupported_claims": [],
                "unsupported_numeric_claim_detected": False,
                **language,
                "disclaimer": CLINICAL_DISCLAIMER,
            },
        }

    raw_citation = citation_accuracy(original, retrieved_chunks, lexical_threshold=claim_support_threshold)
    raw_faithful = faithfulness(original, retrieved_chunks, lexical_threshold=claim_support_threshold)

    high_risk_unsupported_numeric = any(
        (d.get("missing_numbers") or d.get("missing_units")) and not d.get("supported")
        for d in raw_faithful.get("claim_details", [])
    )

    # Retrieval gating remains the first, non-negotiable safety decision.
    if score is not None and score < confidence_threshold:
        refusal = {
            "recommendation": DEFAULT_REFUSAL_MESSAGE,
            "evidence": "",
            "citations": [],
            "confidence": "insufficient",
        }
        language = uncertainty_language(score, confidence_threshold, raw_faithful["faithfulness"], raw_citation["citation_accuracy"])
        return {
            "answer": refusal,
            "safety": {
                "safe_to_return_original": False,
                "guard_triggered": True,
                "repair_applied": False,
                "guard_reasons": ["retrieval score below calibrated threshold"],
                "top_similarity_score": score,
                "confidence_threshold": confidence_threshold,
                "raw_citation_accuracy": raw_citation,
                "raw_faithfulness": raw_faithful,
                "citation_accuracy": raw_citation,
                "faithfulness": raw_faithful,
                "unsupported_claims": raw_faithful.get("unsupported_claims", []),
                "unsupported_numeric_claim_detected": high_risk_unsupported_numeric,
                **language,
                "disclaimer": CLINICAL_DISCLAIMER,
            },
        }

    claim_details = list(raw_faithful.get("claim_details", []))

    # Repair non-essential unsupported parenthetical detail before removing a
    # whole claim. The simplified statement must independently pass the same
    # claim validator against retrieved evidence.
    working_details = []
    simplified_claims = []
    for detail in claim_details:
        if detail.get("supported"):
            working_details.append(detail)
            continue
        original_claim = str(detail.get("claim") or "")
        simplified = _simplify_parenthetical_detail(original_claim)
        if simplified and simplified != original_claim:
            retry = evaluate_claim(simplified, retrieved_chunks, claim_support_threshold).to_dict()
            if retry.get("supported"):
                retry["original_claim"] = original_claim
                retry["simplification_applied"] = True
                working_details.append(retry)
                simplified_claims.append({"original": original_claim, "kept_as": simplified})
                continue
        working_details.append(detail)

    supported_details = [d for d in working_details if d.get("supported")]
    unsupported_details = [d for d in working_details if not d.get("supported")]

    # Nothing medically supported survived: fail closed.
    if not supported_details:
        refusal = {
            "recommendation": DEFAULT_REFUSAL_MESSAGE,
            "evidence": "",
            "citations": [],
            "confidence": "insufficient",
        }
        reasons = ["no supported generated claims remained after post-hoc verification"]
        if high_risk_unsupported_numeric:
            reasons.append("unsupported numerical/dosage claim detected")
        language = uncertainty_language(score, confidence_threshold, raw_faithful["faithfulness"], raw_citation["citation_accuracy"])
        return {
            "answer": refusal,
            "safety": {
                "safe_to_return_original": False,
                "guard_triggered": True,
                "repair_applied": False,
                "guard_reasons": reasons,
                "top_similarity_score": score,
                "confidence_threshold": confidence_threshold,
                "raw_citation_accuracy": raw_citation,
                "raw_faithfulness": raw_faithful,
                "citation_accuracy": raw_citation,
                "faithfulness": raw_faithful,
                "unsupported_claims": raw_faithful.get("unsupported_claims", []),
                "unsupported_numeric_claim_detected": high_risk_unsupported_numeric,
                **language,
                "disclaimer": CLINICAL_DISCLAIMER,
            },
        }

    supporting_indexes = []
    for detail in supported_details:
        indexes = detail.get("supporting_evidence_indexes") or []
        if indexes:
            supporting_indexes.extend(int(i) for i in indexes if i is not None)
        elif detail.get("best_evidence_index") is not None:
            supporting_indexes.append(int(detail["best_evidence_index"]))
    supporting_indexes = [
        _canonical_duplicate_index(retrieved_chunks, i) for i in supporting_indexes
    ]
    supporting_indexes = list(dict.fromkeys(supporting_indexes))
    citations = _dedupe_dicts([
        c for c in (_citation_from_chunk(retrieved_chunks[i - 1]) for i in supporting_indexes if 1 <= i <= len(retrieved_chunks))
        if c is not None
    ])
    repaired_recommendation = " ".join(_ensure_sentence(d.get("claim", "")) for d in supported_details).strip()
    repaired_evidence = _extractive_evidence(retrieved_chunks, supporting_indexes)

    repaired = {
        "recommendation": repaired_recommendation,
        "evidence": repaired_evidence,
        "citations": citations,
        "confidence": original.get("confidence", "medium") if not unsupported_details else "medium",
    }

    final_citation = citation_accuracy(repaired, retrieved_chunks, lexical_threshold=claim_support_threshold)
    final_faithful = faithfulness(repaired, retrieved_chunks, lexical_threshold=claim_support_threshold)
    final_coverage = float(final_citation.get("claim_coverage", 0.0))

    final_reasons = []
    if final_faithful["faithfulness"] < target_faithfulness:
        final_reasons.append(
            f"final faithfulness {final_faithful['faithfulness']:.3f} below target {target_faithfulness:.3f}"
        )
    if final_citation["citation_accuracy"] < min_citation_accuracy:
        final_reasons.append(
            f"final citation accuracy {final_citation['citation_accuracy']:.3f} below minimum {min_citation_accuracy:.3f}"
        )
    if final_coverage < 1.0:
        final_reasons.append(f"final citation claim coverage {final_coverage:.3f} below 1.000")
    if not citations:
        final_reasons.append("no deterministic supporting citation could be resolved")

    repair_applied = bool(
        unsupported_details
        or raw_citation.get("citation_accuracy") != 1.0
        or raw_citation.get("claim_coverage", 0.0) != 1.0
        or original.get("citations") != repaired.get("citations")
    )

    if final_reasons:
        final_answer = {
            "recommendation": DEFAULT_REFUSAL_MESSAGE,
            "evidence": "",
            "citations": [],
            "confidence": "insufficient",
        }
        safe_to_return_original = False
        guard_triggered = True
        guard_reasons = final_reasons
    else:
        final_answer = repaired
        safe_to_return_original = not repair_applied
        guard_triggered = repair_applied
        guard_reasons = []
        if unsupported_details:
            guard_reasons.append(f"removed {len(unsupported_details)} unsupported claim(s) and kept supported content")
        if simplified_claims:
            guard_reasons.append(f"removed unsupported parenthetical detail from {len(simplified_claims)} claim(s)")
        if raw_citation.get("citation_accuracy") != 1.0 or raw_citation.get("claim_coverage", 0.0) != 1.0:
            guard_reasons.append("rebuilt citations deterministically from supporting retrieved chunks")

    language = uncertainty_language(
        top_score=score,
        threshold=confidence_threshold,
        faithfulness_score=final_faithful["faithfulness"],
        citation_score=final_citation["citation_accuracy"],
    )

    # Calibrate wording only after the final answer has passed grounding checks.
    if final_answer.get("confidence") != "insufficient" and final_answer.get("recommendation"):
        strength = language["evidence_strength"]
        rec = str(final_answer["recommendation"]).strip()
        if strength == "partial":
            final_answer["recommendation"] = (
                "The retrieved guideline evidence suggests, though it may not directly address every detail: " + rec
            )
        elif strength == "weak":
            final_answer["recommendation"] = (
                "Limited evidence found. Based only on the retrieved passage: " + rec
            )

    return {
        "answer": final_answer,
        "safety": {
            "safe_to_return_original": safe_to_return_original,
            "guard_triggered": guard_triggered,
            "repair_applied": repair_applied and final_answer.get("confidence") != "insufficient",
            "guard_reasons": guard_reasons,
            "top_similarity_score": score,
            "confidence_threshold": confidence_threshold,
            "raw_citation_accuracy": raw_citation,
            "raw_faithfulness": raw_faithful,
            "citation_accuracy": final_citation,
            "faithfulness": final_faithful,
            "unsupported_claims": raw_faithful.get("unsupported_claims", []),
            "removed_claims": [d.get("claim") for d in unsupported_details],
            "simplified_claims": simplified_claims,
            "unsupported_numeric_claim_detected": high_risk_unsupported_numeric,
            **language,
            "disclaimer": CLINICAL_DISCLAIMER,
        },
    }
