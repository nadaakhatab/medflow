"""Day 4 evaluation metrics: Precision@k, citation accuracy, faithfulness and refusal metrics."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Sequence

try:
    from .claim_validator import evaluate_faithfulness, score_claim_against_evidence, split_claims
except ImportError:  # direct script execution
    from claim_validator import evaluate_faithfulness, score_claim_against_evidence, split_claims


def _meta_content(chunk: Any) -> tuple[Dict[str, Any], str]:
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


def _doc_name(meta: Mapping[str, Any]) -> str:
    value = meta.get("document_name") or meta.get("filename") or meta.get("source") or ""
    return os.path.basename(str(value)).strip().lower()


def _page_number(meta: Mapping[str, Any]) -> int:
    if meta.get("page_number") is not None:
        raw = meta.get("page_number")
    elif meta.get("page") is not None:
        try:
            return int(meta.get("page")) + 1
        except (TypeError, ValueError):
            return -1
    else:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def is_chunk_relevant(meta: Mapping[str, Any], gt_item: Mapping[str, Any]) -> bool:
    """Use the frozen Day 2 relevance rule so Day 4 Precision@4 is comparable."""
    doc_name = _doc_name(meta)
    page_num = _page_number(meta)
    expected_doc = str(gt_item.get("expected_document", "")).lower()
    alternatives = [str(x).lower() for x in gt_item.get("acceptable_alternative_sources", [])]
    pages = gt_item.get("expected_page_range", [1, 999])
    min_page, max_page = int(pages[0]), int(pages[1])
    page_matches = (min_page - 1) <= page_num <= (max_page + 1)

    if doc_name == expected_doc and page_matches:
        return True
    if doc_name in alternatives and page_matches:
        return True
    if doc_name == expected_doc and (min_page - 2) <= page_num <= (max_page + 2):
        return True
    return False


def precision_at_k(retrieved_chunks: Sequence[Any], gt_item: Mapping[str, Any], k: int) -> Dict[str, Any]:
    considered = list(retrieved_chunks[:k])
    flags: List[bool] = []
    for chunk in considered:
        meta, _ = _meta_content(chunk)
        flags.append(is_chunk_relevant(meta, gt_item))
    hits = sum(flags)
    denominator = k if k > 0 else 1
    return {
        "precision_at_k": round(hits / denominator, 4),
        "k": k,
        "relevant_chunks": hits,
        "retrieved_count": len(considered),
        "relevance_flags": flags,
    }


def _norm_section(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _citation_is_refusal(answer: Mapping[str, Any]) -> bool:
    return str(answer.get("confidence", "")).strip().lower() == "insufficient"


def _section_compatible(cited: str, source: str) -> bool:
    if not cited or cited in {"general content", "n/a", "not available"}:
        return True
    if not source or source in {"general content", "n/a", "not available"}:
        return False
    return cited == source or cited in source or source in cited


def _claim_supports_content(claim: str, content: str, lexical_threshold: float) -> tuple[bool, float]:
    support = score_claim_against_evidence(claim, content)
    ok = bool(
        support.support_score >= lexical_threshold
        and not support.missing_numbers
        and not support.missing_units
        and not support.negation_conflict
    )
    return ok, support.support_score


def citation_accuracy(answer: Mapping[str, Any], retrieved_chunks: Sequence[Any], lexical_threshold: float = 0.35) -> Dict[str, Any]:
    """Strict citation accuracy with claim-level source support.

    A citation is correct only when its document/page resolves to at least one
    retrieved chunk on that page, its section is compatible, and that cited page
    supports at least one recommendation claim. All retrieved chunks on the same
    cited page are considered; choosing only the first chunk can create false
    failures when a page was split into several 200-token chunks.

    The return payload also reports *claim_coverage*: the fraction of generated
    claims supported by at least one correct citation. Runtime guardrails require
    both accurate citations and complete claim coverage.
    """
    citations = list(answer.get("citations") or [])
    if not citations:
        return {
            "citation_accuracy": 1.0 if _citation_is_refusal(answer) else 0.0,
            "correct_citations": 0,
            "total_citations": 0,
            "claim_coverage": 1.0 if _citation_is_refusal(answer) else 0.0,
            "covered_claims": 0,
            "total_claims": 0 if _citation_is_refusal(answer) else len(split_claims(str(answer.get("recommendation") or ""))),
            "details": [],
        }

    claims = split_claims(str(answer.get("recommendation") or ""))
    chunk_records = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        meta, content = _meta_content(chunk)
        chunk_records.append((idx, meta, content))

    details: List[Dict[str, Any]] = []
    correct = 0
    covered_claim_indexes: set[int] = set()

    for cit in citations:
        cit_doc = os.path.basename(str(cit.get("document", ""))).strip().lower()
        try:
            cit_page = int(cit.get("page"))
        except (TypeError, ValueError):
            cit_page = -1
        cit_section = _norm_section(cit.get("section"))

        page_candidates = [
            (idx, meta, content)
            for idx, meta, content in chunk_records
            if _doc_name(meta) == cit_doc and _page_number(meta) == cit_page
        ]

        supported_union: set[int] = set()
        partial_union: set[int] = set()
        best_score = 0.0
        supporting_chunk_indexes: List[int] = []
        any_section_ok = False
        compatible_contents: List[str] = []
        contribution_threshold = max(0.10, lexical_threshold * 0.25)
        for chunk_idx, meta, content in page_candidates:
            src_section = _norm_section(meta.get("section_title") or meta.get("section"))
            section_ok = _section_compatible(cit_section, src_section)
            any_section_ok = any_section_ok or section_ok
            if not section_ok:
                continue
            compatible_contents.append(content)
            chunk_supported = []
            for claim_idx, claim in enumerate(claims):
                support = score_claim_against_evidence(claim, content)
                score = support.support_score
                best_score = max(best_score, score)
                if score >= contribution_threshold:
                    partial_union.add(claim_idx)
                ok = bool(
                    score >= lexical_threshold
                    and not support.missing_numbers
                    and not support.missing_units
                    and not support.negation_conflict
                )
                if ok:
                    supported_union.add(claim_idx)
                    chunk_supported.append(claim_idx)
            if chunk_supported:
                supporting_chunk_indexes.append(chunk_idx)

        document_match = bool(page_candidates)
        page_match = bool(page_candidates)
        evidence_support = bool(supported_union)
        best = {
            "document_match": document_match,
            "page_match": page_match,
            "section_match": any_section_ok,
            "evidence_support": evidence_support,
            "evidence_support_score": round(best_score, 4),
            "supporting_chunk_index": supporting_chunk_indexes[0] if supporting_chunk_indexes else None,
            "supporting_chunk_indexes": supporting_chunk_indexes,
            "supported_claim_indexes": sorted(supported_union),
            "partial_claim_indexes": sorted(partial_union),
            "joint_evidence_support": False,
            "_compatible_contents": compatible_contents,
        }

        is_correct = bool(document_match and page_match and any_section_ok and evidence_support)
        if is_correct:
            covered_claim_indexes.update(supported_union)
        best["citation"] = dict(cit)
        best["correct"] = is_correct
        details.append(best)

    # Some recommendation sentences legitimately synthesize two retrieved
    # sources. For claims not fully supported by any single cited page, verify
    # the union of the cited passages. Only citations that materially contribute
    # to a jointly supported claim receive credit.
    for claim_idx, claim in enumerate(claims):
        if claim_idx in covered_claim_indexes:
            continue
        contributors = [d for d in details if claim_idx in d.get("partial_claim_indexes", [])]
        joint_text = "\n".join(
            content
            for d in contributors
            for content in d.get("_compatible_contents", [])
            if content
        )
        if not joint_text:
            continue
        ok, _ = _claim_supports_content(claim, joint_text, lexical_threshold)
        if not ok:
            continue
        covered_claim_indexes.add(claim_idx)
        for d in contributors:
            if d["document_match"] and d["page_match"] and d["section_match"]:
                d["joint_evidence_support"] = True
                d["evidence_support"] = True
                d["correct"] = True
                if claim_idx not in d["supported_claim_indexes"]:
                    d["supported_claim_indexes"].append(claim_idx)
                    d["supported_claim_indexes"].sort()

    correct = sum(1 for d in details if d.get("correct"))
    for d in details:
        d.pop("_compatible_contents", None)

    total = len(citations)
    total_claims = len(claims)
    claim_coverage = (len(covered_claim_indexes) / total_claims) if total_claims else 1.0
    return {
        "citation_accuracy": round(correct / total, 4),
        "correct_citations": correct,
        "total_citations": total,
        "claim_coverage": round(claim_coverage, 4),
        "covered_claims": len(covered_claim_indexes),
        "total_claims": total_claims,
        "details": details,
    }


def faithfulness(answer: Mapping[str, Any], retrieved_chunks: Sequence[Any], lexical_threshold: float = 0.35) -> Dict[str, Any]:
    if _citation_is_refusal(answer):
        return {
            "faithfulness": 1.0,
            "supported_claims": 0,
            "total_claims": 0,
            "unsupported_claim_count": 0,
            "unsupported_claims": [],
            "claim_details": [],
            "not_applicable_reason": "structured refusal",
        }
    return evaluate_faithfulness(str(answer.get("recommendation") or ""), retrieved_chunks, lexical_threshold)


def refusal_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Metrics where expected_answerable=True means the system should answer."""
    tp = tn = fp = fn = 0
    for row in records:
        expected_answerable = bool(row.get("expected_answerable"))
        answered = bool(row.get("answered"))
        if expected_answerable and answered:
            tp += 1
        elif (not expected_answerable) and (not answered):
            tn += 1
        elif (not expected_answerable) and answered:
            fp += 1  # unsafe accept
        else:
            fn += 1  # false refusal

    total = tp + tn + fp + fn
    answer_precision = tp / (tp + fp) if (tp + fp) else 1.0
    answer_recall = tp / (tp + fn) if (tp + fn) else 1.0
    refusal_precision = tn / (tn + fn) if (tn + fn) else 1.0
    refusal_recall = tn / (tn + fp) if (tn + fp) else 1.0
    return {
        "total": total,
        "tp_answered_supported": tp,
        "tn_refused_unsupported": tn,
        "fp_unsafe_accept": fp,
        "fn_false_refusal": fn,
        "answerability_accuracy": round((tp + tn) / total, 4) if total else 1.0,
        "answer_precision": round(answer_precision, 4),
        "answer_recall": round(answer_recall, 4),
        "refusal_precision": round(refusal_precision, 4),
        "refusal_recall": round(refusal_recall, 4),
        "unsafe_accept_rate": round(fp / max(1, fp + tn), 4),
        "false_refusal_rate": round(fn / max(1, fn + tp), 4),
    }
