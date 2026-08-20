"""Print the judge-ready Day 2 retrieval scorecard from saved experiment artifacts.

No embeddings are rebuilt and no benchmark is rerun.  The report consolidates the
already-measured K=3/4/5 trade-off so the agenda's Precision@K requirement is
shown explicitly while preserving the frozen K=4 configuration.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPK_PATH = ROOT / "evaluation" / "top_k_validation_results.json"


def load_scorecard():
    data = json.loads(TOPK_PATH.read_text(encoding="utf-8"))
    k3, k4, k5 = data["3"], data["4"], data["5"]
    return {
        "precision_at_3": k3["precision_at_k"],
        "precision_at_4": k4["precision_at_k"],
        "precision_at_5": k5["precision_at_k"],
        "hit_at_3": k3["hit_at_k"],
        "hit_at_4": k4["hit_at_k"],
        "hit_at_5": k5["hit_at_k"],
        "mrr_at_selected_k": k4["mrr"],
        "selected_k": 4,
        "selected_noise_ratio_pct": k4["noise_ratio_pct"],
        "selection_rationale": (
            "K=4 improves Hit@K over K=3 (0.8750 vs 0.8125); K=5 adds no hit-rate "
            "gain while lowering precision and increasing noise."
        ),
    }


def main():
    report = load_scorecard()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
