"""Post-hoc unsupported-claim detection for Day 4.

The checker is intentionally transparent and dependency-light. It uses lexical
support plus conservative numeric/unit checks. Negation handling is *local* to
matched concepts so an unrelated word such as "not" elsewhere in a 200-token
chunk cannot invalidate an otherwise supported claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Sequence

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "for", "from", "has", "have", "had", "in", "into", "is", "it", "its",
    "may", "of", "on", "or", "should", "that", "the", "their", "there", "these",
    "this", "those", "to", "was", "were", "will", "with", "would", "guideline",
    "patient", "patients", "clinical", "evidence", "recommend", "recommends",
}
NEGATIONS = {"no", "not", "never", "without", "contraindicated", "avoid"}
UNIT_PATTERN = re.compile(
    r"\b(?:mg|mcg|µg|ug|g|kg|ml|l|mmol|meq|units?|iu|%|cm|mm|hours?|days?|weeks?|months?|years?)\b",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?(?!\w)")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9µ]+(?:[-'][A-Za-z0-9µ]+)?")


@dataclass
class ClaimSupport:
    claim: str
    supported: bool
    support_score: float
    best_evidence_index: int | None
    matched_terms: List[str]
    missing_numbers: List[str]
    missing_units: List[str]
    negation_conflict: bool
    reason: str
    supporting_evidence_indexes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    return (
        str(text or "")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _content_from_chunk(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(
            chunk.get("retrieved_passage")
            or chunk.get("page_content")
            or chunk.get("text")
            or chunk.get("content")
            or ""
        )
    return str(getattr(chunk, "page_content", "") or "")


def split_claims(text: str) -> List[str]:
    """Split a generated recommendation into auditable sentence/bullet claims."""
    if not text or not text.strip():
        return []
    normalized = re.sub(r"\r\n?", "\n", _normalize(text).strip())
    normalized = re.sub(r"^[\s\-•*]+", "", normalized)
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s*[;]\s*", normalized)
    claims = []
    for part in parts:
        part = re.sub(r"^[\s\-•*\d.)]+", "", part).strip()
        if len(part) >= 3 and re.search(r"[A-Za-z]", part):
            claims.append(part)
    return claims


def content_tokens(text: str) -> List[str]:
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(_normalize(text))]
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _all_tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(_normalize(text))]


def _numbers(text: str) -> List[str]:
    normalized = _normalize(text)
    values = [re.sub(r"\s+", "", x) for x in NUMBER_PATTERN.findall(normalized)]
    # Isotope identifiers are often written either I-131 or 131I. Treat these
    # spellings as the same factual number without classifying them as a dose.
    for pattern in (
        r"\b(?:i|iodine)[-\s]?(\d{2,3})\b",
        r"\b(\d{2,3})\s*i\b",
    ):
        values.extend(re.findall(pattern, normalized, flags=re.IGNORECASE))
    return list(dict.fromkeys(values))


def _units(text: str) -> List[str]:
    return [x.lower() for x in UNIT_PATTERN.findall(_normalize(text))]


def _negated_scope_terms(text: str, window: int = 4) -> set[str]:
    """Return content terms located close to an explicit negator.

    Scope never crosses a sentence/clause boundary. This prevents an unrelated
    phrase such as "does not cause obesity" from negating "mild weight gain" in
    the preceding sentence merely because both occur in the same 200-token chunk.
    """
    scoped: set[str] = set()
    segments = re.split(r"[.!?;\n]+", _normalize(text))
    for segment in segments:
        tokens = _all_tokens(segment)
        for i, token in enumerate(tokens):
            if token not in NEGATIONS:
                continue
            lo = max(0, i - window)
            hi = min(len(tokens), i + window + 1)
            for nearby in tokens[lo:hi]:
                if nearby not in NEGATIONS and len(nearby) > 1 and nearby not in STOPWORDS:
                    scoped.add(nearby)
    return scoped


def _local_negation_conflict(claim: str, evidence: str, matched_terms: Sequence[str]) -> bool:
    salient = set(matched_terms)
    if not salient:
        return False
    claim_negated = _negated_scope_terms(claim) & salient
    evidence_negated = _negated_scope_terms(evidence) & salient
    # Compare whether the proposition has local negative polarity, not whether
    # every token inside the negation window is identical. This preserves true
    # conflicts ("FNA is indicated" vs "FNA is not indicated") while avoiding
    # false conflicts caused by slightly different nearby wording.
    return bool(claim_negated) != bool(evidence_negated)


def _evidence_segments(text: str) -> List[str]:
    normalized = _normalize(text)
    # Clinical guideline prose frequently contains semicolon-separated rules.
    # Keep each local proposition separate so a negative exception elsewhere in
    # a 200-token chunk cannot negate a positive recommendation.
    parts = re.split(r"(?<=[.!?])\s+|[;,\n]+", normalized)
    return [re.sub(r"\s+", " ", x).strip() for x in parts if x and x.strip()]


def score_claim_against_evidence(claim: str, evidence: str) -> ClaimSupport:
    claim_clean = " ".join(_normalize(claim).split())
    evidence_clean = " ".join(_normalize(evidence).split())
    if not claim_clean or not evidence_clean:
        return ClaimSupport(claim, False, 0.0, None, [], _numbers(claim), _units(claim), False, "empty evidence")

    claim_terms = set(content_tokens(claim_clean))

    # Score against polarity-compatible local propositions rather than the
    # entire chunk. This is crucial for guideline passages that contain both a
    # recommendation and a nearby exception/negative statement. Compatible
    # segments may be combined because one generated claim can legitimately
    # summarize two sentences from the same retrieved passage.
    compatible_segments: List[str] = []
    for segment in _evidence_segments(evidence_clean):
        seg_terms = set(content_tokens(segment))
        matched_here = sorted(claim_terms & seg_terms)
        if not matched_here:
            continue
        if not _local_negation_conflict(claim_clean, segment, matched_here):
            compatible_segments.append(segment)

    effective_evidence = " ".join(compatible_segments) if compatible_segments else evidence_clean
    evidence_terms = set(content_tokens(effective_evidence))
    matched = sorted(claim_terms & evidence_terms)
    score = (len(matched) / len(claim_terms)) if claim_terms else 0.0

    claim_numbers = _numbers(claim_clean)
    evidence_numbers = set(_numbers(effective_evidence))
    missing_numbers = [n for n in claim_numbers if n not in evidence_numbers]

    claim_units = _units(claim_clean)
    evidence_units = set(_units(effective_evidence))
    missing_units = [u for u in claim_units if u not in evidence_units]

    # If no compatible proposition could be found, preserve the strict local
    # polarity check on the original evidence. Otherwise the selected evidence
    # is already polarity-compatible by construction.
    negation_conflict = False if compatible_segments else _local_negation_conflict(claim_clean, evidence_clean, matched)

    exact = claim_clean.lower() in effective_evidence.lower()
    if exact and not missing_numbers and not missing_units and not negation_conflict:
        return ClaimSupport(
            claim=claim,
            supported=True,
            support_score=1.0,
            best_evidence_index=None,
            matched_terms=matched or sorted(claim_terms),
            missing_numbers=[],
            missing_units=[],
            negation_conflict=False,
            reason="exact text match",
        )

    reason_bits = [f"lexical_support={score:.3f}"]
    if missing_numbers:
        reason_bits.append("missing numeric value(s): " + ", ".join(missing_numbers))
    if missing_units:
        reason_bits.append("missing unit(s): " + ", ".join(missing_units))
    if negation_conflict:
        reason_bits.append("local negation polarity differs")

    return ClaimSupport(
        claim=claim,
        supported=False,
        support_score=round(score, 4),
        best_evidence_index=None,
        matched_terms=matched,
        missing_numbers=missing_numbers,
        missing_units=missing_units,
        negation_conflict=negation_conflict,
        reason="; ".join(reason_bits),
    )


def evaluate_claim(
    claim: str,
    retrieved_chunks: Sequence[Any],
    lexical_threshold: float = 0.35,
) -> ClaimSupport:
    best: ClaimSupport | None = None
    best_index: int | None = None
    per_chunk: List[tuple[int, ClaimSupport, str]] = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        content = _content_from_chunk(chunk)
        candidate = score_claim_against_evidence(claim, content)
        per_chunk.append((idx, candidate, content))
        if best is None or candidate.support_score > best.support_score:
            best = candidate
            best_index = idx
        if (
            candidate.support_score >= lexical_threshold
            and not candidate.missing_numbers
            and not candidate.missing_units
            and not candidate.negation_conflict
        ):
            candidate.supported = True
            candidate.best_evidence_index = idx
            candidate.supporting_evidence_indexes = [idx]
            candidate.reason = f"supported by evidence {idx}; lexical_support={candidate.support_score:.3f}"
            return candidate

    # A single generated claim may summarize evidence distributed across more
    # than one of the top-k chunks (for example initial risk features in one
    # chunk and response-to-therapy refinement in another). Evaluate the union
    # only after no individual chunk was sufficient.
    # Greedily build the smallest high-signal subset of chunks whose union
    # supports the claim. This preserves traceable multi-source provenance and
    # avoids citing every top-k chunk just because it shares a few words.
    ranked = sorted(per_chunk, key=lambda row: row[1].support_score, reverse=True)
    selected: List[tuple[int, ClaimSupport, str]] = []
    for row in ranked:
        if not row[2] or row[1].support_score <= 0.0:
            continue
        selected.append(row)
        combined_text = "\n".join(content for _, _, content in selected)
        combined = score_claim_against_evidence(claim, combined_text)
        if (
            combined.support_score >= lexical_threshold
            and not combined.missing_numbers
            and not combined.missing_units
            and not combined.negation_conflict
        ):
            contributing = [idx for idx, _, _ in selected]
            combined.supported = True
            combined.best_evidence_index = contributing[0] if contributing else best_index
            combined.supporting_evidence_indexes = contributing
            combined.reason = (
                "supported by combined retrieved evidence "
                + ",".join(str(i) for i in contributing)
                + f"; lexical_support={combined.support_score:.3f}"
            )
            return combined

    if best is None:
        best = ClaimSupport(claim, False, 0.0, None, [], _numbers(claim), _units(claim), False, "no retrieved evidence")
    best.best_evidence_index = best_index
    best.supporting_evidence_indexes = [best_index] if best_index else []
    best.supported = False
    if best.support_score < lexical_threshold:
        best.reason = f"best lexical support {best.support_score:.3f} below threshold {lexical_threshold:.3f}" + (
            "; " + best.reason if best.reason else ""
        )
    return best


def evaluate_faithfulness(
    answer_text: str,
    retrieved_chunks: Sequence[Any],
    lexical_threshold: float = 0.35,
) -> Dict[str, Any]:
    claims = split_claims(answer_text)
    details = [evaluate_claim(c, retrieved_chunks, lexical_threshold) for c in claims]
    supported = sum(1 for d in details if d.supported)
    total = len(details)
    score = 1.0 if total == 0 else supported / total
    return {
        "faithfulness": round(score, 4),
        "supported_claims": supported,
        "total_claims": total,
        "unsupported_claim_count": total - supported,
        "unsupported_claims": [d.claim for d in details if not d.supported],
        "claim_details": [d.to_dict() for d in details],
    }
