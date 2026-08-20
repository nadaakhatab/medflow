"""Scientific threshold sweep for answerable vs unsupported queries."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from .config import MAX_UNSAFE_ACCEPT_RATE
except ImportError:
    from config import MAX_UNSAFE_ACCEPT_RATE


def _metrics(samples: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for sample in samples:
        expected = bool(sample["expected_answerable"])
        predicted = float(sample["top_score"]) >= threshold
        if expected and predicted:
            tp += 1
        elif (not expected) and (not predicted):
            tn += 1
        elif (not expected) and predicted:
            fp += 1
        else:
            fn += 1

    pos = tp + fn
    neg = tn + fp
    total = pos + neg
    answer_precision = tp / (tp + fp) if (tp + fp) else 1.0
    answer_recall = tp / pos if pos else 1.0
    answer_f1 = 2 * answer_precision * answer_recall / (answer_precision + answer_recall) if (answer_precision + answer_recall) else 0.0
    refusal_precision = tn / (tn + fn) if (tn + fn) else 1.0
    refusal_recall = tn / neg if neg else 1.0
    specificity = tn / neg if neg else 1.0
    balanced = 0.5 * (answer_recall + specificity)
    return {
        "threshold": round(float(threshold), 6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 1.0,
        "balanced_accuracy": round(balanced, 4),
        "answer_precision": round(answer_precision, 4),
        "answer_recall": round(answer_recall, 4),
        "answer_f1": round(answer_f1, 4),
        "refusal_precision": round(refusal_precision, 4),
        "refusal_recall": round(refusal_recall, 4),
        "unsafe_accept_rate": round(fp / neg, 4) if neg else 0.0,
        "false_refusal_rate": round(fn / pos, 4) if pos else 0.0,
    }


def candidate_thresholds(samples: Sequence[Mapping[str, Any]]) -> List[float]:
    scores = sorted({round(float(s["top_score"]), 6) for s in samples})
    candidates = {0.0, 1.0}
    candidates.update(scores)
    for a, b in zip(scores, scores[1:]):
        candidates.add(round((a + b) / 2.0, 6))
    # Also expose the range used as a teaching starting point in the Day 4 deck,
    # without privileging it over the team's measured data.
    candidates.update(round(x / 100, 2) for x in range(40, 86))
    return sorted(candidates)


def calibrate_threshold(
    samples: Sequence[Mapping[str, Any]],
    max_unsafe_accept_rate: float = MAX_UNSAFE_ACCEPT_RATE,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Threshold calibration requires at least one labeled sample.")
    if not any(bool(s["expected_answerable"]) for s in samples):
        raise ValueError("Calibration requires answerable samples.")
    if not any(not bool(s["expected_answerable"]) for s in samples):
        raise ValueError("Calibration requires unsupported/refusal samples.")

    rows = [_metrics(samples, t) for t in candidate_thresholds(samples)]
    safe = [r for r in rows if r["unsafe_accept_rate"] <= max_unsafe_accept_rate]
    if safe:
        # Clinical priority: respect the unsafe-accept ceiling, then maximize balanced
        # correctness, then answer coverage, and finally choose the lower threshold
        # to avoid needless refusals when tied.
        selected = max(
            safe,
            key=lambda r: (r["balanced_accuracy"], r["answer_recall"], r["answer_f1"], -r["threshold"]),
        )
        selection_mode = "safety_constrained_balanced_accuracy"
    else:
        selected = min(
            rows,
            key=lambda r: (r["unsafe_accept_rate"], -r["balanced_accuracy"], -r["answer_recall"], r["threshold"]),
        )
        selection_mode = "minimum_unsafe_accept_fallback"

    answerable_scores = [float(s["top_score"]) for s in samples if bool(s["expected_answerable"])]
    unsupported_scores = [float(s["top_score"]) for s in samples if not bool(s["expected_answerable"])]
    separation = min(answerable_scores) - max(unsupported_scores)

    return {
        "selected_threshold": selected["threshold"],
        "selection_mode": selection_mode,
        "safety_constraint_max_unsafe_accept_rate": max_unsafe_accept_rate,
        "selected_metrics": selected,
        "answerable_score_min": round(min(answerable_scores), 4),
        "answerable_score_max": round(max(answerable_scores), 4),
        "unsupported_score_min": round(min(unsupported_scores), 4),
        "unsupported_score_max": round(max(unsupported_scores), 4),
        "score_separation_gap": round(separation, 4),
        "clean_separation": separation > 0,
        "sample_count": len(samples),
        "sweep": rows,
    }


def save_calibration(result: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    sweep = list(result.get("sweep") or [])
    if sweep:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            writer.writeheader()
            writer.writerows(sweep)


def infer_query_family(query: str) -> str:
    """Small transparent family classifier used only for optional threshold stratification."""
    q = (query or "").lower()
    if any(x in q for x in ("dose", "dosing", "mg/kg", "milligram", "microgram", "mcg/kg")):
        return "dosage"
    if any(x in q for x in ("diagnos", "screening", "screen ", "test for", "tests are used")):
        return "diagnosis_evaluation"
    if any(x in q for x in ("treat", "therapy", "medication", "medicine", "antihypertensive", "management")):
        return "treatment_management"
    if any(x in q for x in ("surgery", "surgical", "procedure", "biopsy", "thyroidectomy", "fna")):
        return "procedure"
    if any(x in q for x in ("symptom", "signs", "manifestation", "orbitopathy")):
        return "symptoms"
    return "other"


def calibrate_stratified_thresholds(
    samples: Sequence[Mapping[str, Any]],
    global_result: Mapping[str, Any] | None = None,
    min_positive: int = 2,
    min_negative: int = 2,
    max_unsafe_accept_rate: float = MAX_UNSAFE_ACCEPT_RATE,
) -> Dict[str, Any]:
    """Calibrate per-query-family thresholds only when labels are sufficient.

    Day 4 warns against blindly forcing one threshold onto every query type. We
    therefore permit family-specific operating points, but fall back to the global
    threshold when a family lacks enough positive/negative examples. This avoids
    pretending that a tiny subgroup is scientifically calibrated.
    """
    global_result = dict(global_result or calibrate_threshold(samples, max_unsafe_accept_rate))
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for s in samples:
        family = str(s.get("query_family") or infer_query_family(str(s.get("query", ""))))
        groups.setdefault(family, []).append(s)

    family_results: Dict[str, Any] = {}
    for family, rows in sorted(groups.items()):
        pos = sum(1 for r in rows if bool(r["expected_answerable"]))
        neg = len(rows) - pos
        if pos >= min_positive and neg >= min_negative:
            result = calibrate_threshold(rows, max_unsafe_accept_rate)
            family_results[family] = {
                "mode": "family_calibrated",
                "sample_count": len(rows),
                "positive_count": pos,
                "negative_count": neg,
                "selected_threshold": result["selected_threshold"],
                "selected_metrics": result["selected_metrics"],
            }
        else:
            family_results[family] = {
                "mode": "global_fallback_insufficient_family_labels",
                "sample_count": len(rows),
                "positive_count": pos,
                "negative_count": neg,
                "selected_threshold": global_result["selected_threshold"],
            }
    return {
        "global_threshold": global_result["selected_threshold"],
        "family_thresholds": family_results,
        "minimum_labels_per_class": {"positive": min_positive, "negative": min_negative},
    }


def threshold_for_query(query: str, stratified: Mapping[str, Any]) -> float:
    family = infer_query_family(query)
    family_row = (stratified.get("family_thresholds") or {}).get(family) or {}
    return float(family_row.get("selected_threshold", stratified.get("global_threshold", 0.50)))
