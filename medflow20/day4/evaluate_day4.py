"""Day 4 internal evaluation runner for MedFlow.

Default behavior is conservative: it audits the persisted index first and refuses to
present frozen-Day-2-comparable metrics when the live collection does not match the
frozen chunk count. Use --allow-index-mismatch only for an explicitly labeled live
index diagnostic run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from day4 import config
    from day4.evaluation_metrics import citation_accuracy, faithfulness, precision_at_k, refusal_metrics
    from day4.index_audit import audit_index, save_audit
    from day4.safety_guardrails import apply_posthoc_guard
    from day4.threshold_calibration import calibrate_threshold, calibrate_stratified_thresholds, infer_query_family, threshold_for_query, save_calibration
else:
    from . import config
    from .evaluation_metrics import citation_accuracy, faithfulness, precision_at_k, refusal_metrics
    from .index_audit import audit_index, save_audit
    from .safety_guardrails import apply_posthoc_guard
    from .threshold_calibration import calibrate_threshold, calibrate_stratified_thresholds, infer_query_family, threshold_for_query, save_calibration


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _runtime_store(persist_dir: Path, collection: str):
    from langchain_chroma import Chroma
    from rag_pipeline import get_embeddings
    embeddings = get_embeddings(config.EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def _retrieve(store, query: str, k: int):
    from rag_pipeline import semantic_retrieval
    return semantic_retrieval(store, query, top_k=k)


def _collect_calibration_samples(store, k: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for item in _load_json(config.GROUND_TRUTH_PATH):
        chunks = _retrieve(store, item["question"], k)
        top = max((float(c.get("similarity_score", 0.0)) for c in chunks), default=0.0)
        samples.append({
            "id": item["query_id"], "query": item["question"], "category": item.get("condition", ""),
            "expected_answerable": True, "top_score": top, "query_family": infer_query_family(item["question"]),
        })
    for item in _load_json(config.REFUSAL_CASES_PATH):
        chunks = _retrieve(store, item["query"], k)
        top = max((float(c.get("similarity_score", 0.0)) for c in chunks), default=0.0)
        samples.append({
            "id": item["test_id"], "query": item["query"], "category": item.get("category", ""),
            "expected_answerable": False, "top_score": top, "query_family": infer_query_family(item["query"]),
        })
    return samples


def _write_rows(rows: List[Dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    flat_rows = []
    for r in rows:
        flat_rows.append({
            "id": r.get("id"), "query": r.get("query"), "expected_answerable": r.get("expected_answerable"),
            "top_score": r.get("top_score"), "threshold": r.get("threshold"), "answered": r.get("answered"),
            "precision_at_4": r.get("precision_at_4"), "citation_accuracy": r.get("citation_accuracy"),
            "faithfulness": r.get("faithfulness"), "raw_citation_accuracy": r.get("raw_citation_accuracy"),
            "raw_faithfulness": r.get("raw_faithfulness"), "unsupported_claim_count": r.get("unsupported_claim_count"),
            "guard_triggered": r.get("guard_triggered"), "repair_applied": r.get("repair_applied"),
            "final_confidence": r.get("final_confidence"),
        })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader(); writer.writerows(flat_rows)


def run_full_evaluation(store, calibration: Dict[str, Any], k: int, index_audit_result: Dict[str, Any]) -> Dict[str, Any]:
    from generator import generate_answer
    rows: List[Dict[str, Any]] = []
    refusal_records: List[Dict[str, Any]] = []

    stratified = calibration.get("stratified_thresholds") or {
        "global_threshold": calibration["selected_threshold"],
        "family_thresholds": {},
    }

    for item in _load_json(config.GROUND_TRUTH_PATH):
        query = item["question"]
        threshold = threshold_for_query(query, stratified)
        chunks = _retrieve(store, query, k)
        top = max((float(c.get("similarity_score", 0.0)) for c in chunks), default=0.0)
        p = precision_at_k(chunks, item, k)
        raw = generate_answer(query, chunks, confidence_threshold=threshold)
        raw_was_refusal = str(raw.get("confidence", "")).lower() == "insufficient"
        guarded = apply_posthoc_guard(raw, chunks, confidence_threshold=threshold)
        final_answer = guarded["answer"]
        safety = guarded["safety"]
        answered = str(final_answer.get("confidence", "")).lower() != "insufficient"
        refusal_records.append({"expected_answerable": True, "answered": answered})

        # A pre-generation/refusal-path answer has no generated claims or citations to score.
        # Mark those metrics N/A instead of awarding an artificial 1.0. If generation
        # occurred and the post-hoc guard later refused it, retain the original safety
        # scores because they are exactly what triggered the guard.
        citation_score = None if raw_was_refusal else safety["citation_accuracy"]["citation_accuracy"]
        faithfulness_score = None if raw_was_refusal else safety["faithfulness"]["faithfulness"]
        raw_citation_score = None if raw_was_refusal else safety.get("raw_citation_accuracy", safety["citation_accuracy"])["citation_accuracy"]
        raw_faithfulness_score = None if raw_was_refusal else safety.get("raw_faithfulness", safety["faithfulness"])["faithfulness"]
        unsupported_count = None if raw_was_refusal else safety.get("raw_faithfulness", safety["faithfulness"])["unsupported_claim_count"]

        rows.append({
            "id": item["query_id"], "query": query, "expected_answerable": True,
            "top_score": top, "threshold": threshold, "answered": answered,
            "precision_at_4": p["precision_at_k"],
            "citation_accuracy": citation_score,
            "faithfulness": faithfulness_score,
            "raw_citation_accuracy": raw_citation_score,
            "raw_faithfulness": raw_faithfulness_score,
            "unsupported_claim_count": unsupported_count,
            "guard_triggered": safety["guard_triggered"],
            "repair_applied": bool(safety.get("repair_applied", False)),
            "final_confidence": final_answer.get("confidence"),
            "details": {"precision": p, "safety": safety, "raw_answer": raw, "answer": final_answer},
        })

    for item in _load_json(config.REFUSAL_CASES_PATH):
        query = item["query"]
        threshold = threshold_for_query(query, stratified)
        chunks = _retrieve(store, query, k)
        top = max((float(c.get("similarity_score", 0.0)) for c in chunks), default=0.0)
        raw = generate_answer(query, chunks, confidence_threshold=threshold)
        guarded = apply_posthoc_guard(raw, chunks, confidence_threshold=threshold)
        final_answer = guarded["answer"]
        answered = str(final_answer.get("confidence", "")).lower() != "insufficient"
        refusal_records.append({"expected_answerable": False, "answered": answered})
        rows.append({
            "id": item["test_id"], "query": query, "expected_answerable": False,
            "top_score": top, "threshold": threshold, "answered": answered,
            "precision_at_4": None, "citation_accuracy": None, "faithfulness": None,
            "raw_citation_accuracy": None, "raw_faithfulness": None,
            "unsupported_claim_count": None, "guard_triggered": guarded["safety"]["guard_triggered"],
            "repair_applied": bool(guarded["safety"].get("repair_applied", False)),
            "final_confidence": final_answer.get("confidence"),
            "details": {"safety": guarded["safety"], "raw_answer": raw, "answer": final_answer},
        })

    answerable_rows = [r for r in rows if r["expected_answerable"]]
    grounding_rows = [
        r for r in answerable_rows
        if r["citation_accuracy"] is not None and r["faithfulness"] is not None
    ]
    mean_citation = (
        round(sum(r["citation_accuracy"] for r in grounding_rows) / len(grounding_rows), 4)
        if grounding_rows else None
    )
    mean_faithfulness = (
        round(sum(r["faithfulness"] for r in grounding_rows) / len(grounding_rows), 4)
        if grounding_rows else None
    )
    raw_grounding_rows = [
        r for r in answerable_rows
        if r.get("raw_citation_accuracy") is not None and r.get("raw_faithfulness") is not None
    ]
    mean_raw_citation = (
        round(sum(r["raw_citation_accuracy"] for r in raw_grounding_rows) / len(raw_grounding_rows), 4)
        if raw_grounding_rows else None
    )
    mean_raw_faithfulness = (
        round(sum(r["raw_faithfulness"] for r in raw_grounding_rows) / len(raw_grounding_rows), 4)
        if raw_grounding_rows else None
    )
    summary = {
        "index_audit": index_audit_result,
        "calibrated_threshold": calibration["selected_threshold"],
        "stratified_thresholds": calibration["stratified_thresholds"],
        "top_k": k,
        "answerable_queries": len(answerable_rows),
        "refusal_queries": len(rows) - len(answerable_rows),
        "mean_precision_at_4": round(sum(r["precision_at_4"] for r in answerable_rows) / len(answerable_rows), 4) if answerable_rows else None,
        "grounding_metrics_scored_queries": len(grounding_rows),
        "mean_citation_accuracy": mean_citation,
        "mean_faithfulness": mean_faithfulness,
        "mean_raw_citation_accuracy": mean_raw_citation,
        "mean_raw_faithfulness": mean_raw_faithfulness,
        "repaired_answer_count": sum(1 for r in answerable_rows if r.get("repair_applied")),
        "citation_accuracy_target": config.MIN_CITATION_ACCURACY,
        "citation_accuracy_target_met": (mean_citation >= config.MIN_CITATION_ACCURACY) if mean_citation is not None else None,
        "faithfulness_target": config.TARGET_FAITHFULNESS,
        "faithfulness_target_met": (mean_faithfulness >= config.TARGET_FAITHFULNESS) if mean_faithfulness is not None else None,
        "refusal_metrics": refusal_metrics(refusal_records),
        "metrics_are_frozen_day2_comparable": bool(index_audit_result.get("index_matches_frozen_day2")),
    }

    _write_rows(
        rows,
        config.RESULTS_DIR / "day4_evaluation_log.json",
        config.RESULTS_DIR / "day4_evaluation_log.csv",
    )
    (config.RESULTS_DIR / "day4_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedFlow Day 4 safety evaluation")
    parser.add_argument("--audit-index", action="store_true", help="Audit the persisted index only")
    parser.add_argument("--calibrate-only", action="store_true", help="Calibrate threshold and stop")
    parser.add_argument("--full", action="store_true", help="Run threshold calibration plus full evaluation")
    parser.add_argument("--allow-index-mismatch", action="store_true", help="Allow explicitly labeled diagnostics on a live index that differs from frozen Day 2")
    parser.add_argument("--persist-dir", type=Path, default=config.LIVE_PERSIST_DIR)
    parser.add_argument("--collection", default=config.LIVE_COLLECTION_NAME)
    args = parser.parse_args(argv)

    audit = audit_index(args.persist_dir, args.collection)
    save_audit(audit)
    print(json.dumps(audit, indent=2))
    if args.audit_index and not args.calibrate_only and not args.full:
        return 0

    if config.STRICT_INDEX_MATCH and not audit["index_matches_frozen_day2"] and not args.allow_index_mismatch:
        print(
            "\nSTOP: the selected persisted collection does not match the frozen Day 2 index.\n"
            "This evaluator will not silently mix a different live index with frozen benchmark claims.\n"
            "Build the separate token-aware index with `python day4/frozen_index_builder.py`, then run:\n"
            "  python day4/evaluate_day4.py --full --persist-dir chroma_db_day2_frozen --collection thyroid_day2_frozen\n"
            "or use --allow-index-mismatch for a clearly labeled diagnostic run.",
            file=sys.stderr,
        )
        return 2

    store = _runtime_store(args.persist_dir, args.collection)
    samples = _collect_calibration_samples(store, config.TOP_K)
    calibration = calibrate_threshold(samples)
    calibration["stratified_thresholds"] = calibrate_stratified_thresholds(samples, global_result=calibration)
    calibration["samples"] = samples
    calibration["index_audit"] = audit
    save_calibration(
        calibration,
        config.RESULTS_DIR / "threshold_calibration.json",
        config.RESULTS_DIR / "threshold_calibration_sweep.csv",
    )
    print(f"\nSelected threshold: {calibration['selected_threshold']:.4f}")
    print("Calibration operating point:")
    print(json.dumps({
        "selection_mode": calibration.get("selection_mode"),
        "selected_metrics": calibration.get("selected_metrics"),
        "answerable_score_min": calibration.get("answerable_score_min"),
        "answerable_score_max": calibration.get("answerable_score_max"),
        "unsupported_score_min": calibration.get("unsupported_score_min"),
        "unsupported_score_max": calibration.get("unsupported_score_max"),
        "score_separation_gap": calibration.get("score_separation_gap"),
        "clean_separation": calibration.get("clean_separation"),
    }, indent=2))
    if args.calibrate_only and not args.full:
        return 0

    if args.full:
        summary = run_full_evaluation(store, calibration, config.TOP_K, audit)
        print("\nDAY 4 SUMMARY")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
