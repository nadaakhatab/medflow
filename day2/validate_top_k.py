import os
import sys
import time
import json
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db_embeddings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "emb_BAAI_bge_small_en_v1.5"
GROUND_TRUTH_FILE = os.path.join("evaluation", "thyroid_ground_truth.json")


def is_chunk_relevant(doc_metadata: Dict[str, Any], gt_item: Dict[str, Any]) -> bool:
    doc_name = doc_metadata.get("document_name") or doc_metadata.get("filename") or os.path.basename(doc_metadata.get("source", ""))
    page_num = doc_metadata.get("page_number") or (doc_metadata.get("page", 0) + 1)
    try:
        page_num = int(page_num)
    except (ValueError, TypeError):
        page_num = -1

    expected_doc = gt_item.get("expected_document", "")
    alt_sources = gt_item.get("acceptable_alternative_sources", [])
    expected_pages = gt_item.get("expected_page_range", [1, 999])
    
    min_page, max_page = expected_pages[0], expected_pages[1]
    page_matches = (min_page - 1) <= page_num <= (max_page + 1)

    if doc_name.lower() == expected_doc.lower() and page_matches:
        return True

    for alt in alt_sources:
        if doc_name.lower() == alt.lower() and page_matches:
            return True
            
    if doc_name.lower() == expected_doc.lower() and (min_page - 2) <= page_num <= (max_page + 2):
        return True

    return False


def validate_top_k_options():
    print(f"\n=======================================================")
    print(f"VALIDATING TOP-K (K=3 vs K=4 vs K=5) on BGE-Small (200 tokens)")
    print(f"=======================================================\n")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    k_values = [3, 4, 5]
    results_by_k = {}
    gt15_analysis = {}

    for k in k_values:
        latencies = []
        hits = []
        precisions = []
        rrs = []
        total_retrieved = 0
        total_relevant = 0
        total_noise = 0

        for item in ground_truth:
            q_text = item["question"]
            search_query = f"Represent this sentence for searching relevant passages: {q_text}"

            t0 = time.time()
            results = vector_store.similarity_search_with_score(search_query, k=k)
            latencies.append((time.time() - t0) * 1000.0)

            first_rel = None
            rel_count = 0

            for rank, (doc, dist) in enumerate(results, 1):
                is_rel = is_chunk_relevant(doc.metadata, item)
                if is_rel:
                    rel_count += 1
                    if first_rel is None:
                        first_rel = rank

                # Log GT_15 specifically
                if item["query_id"] == "GT_15":
                    if k not in gt15_analysis:
                        gt15_analysis[k] = []
                    gt15_analysis[k].append({
                        "rank": rank,
                        "doc": doc.metadata.get("filename", ""),
                        "page": doc.metadata.get("page_number", ""),
                        "is_relevant": is_rel,
                        "snippet": doc.page_content[:150].replace("\n", " ")
                    })

            total_retrieved += k
            total_relevant += rel_count
            total_noise += (k - rel_count)

            hits.append(1 if (first_rel is not None) else 0)
            precisions.append(rel_count / float(k))
            rrs.append(1.0 / first_rel if first_rel else 0.0)

        total_q = len(ground_truth)
        results_by_k[k] = {
            "k": k,
            "hit_at_k": round(sum(hits) / total_q, 4),
            "precision_at_k": round(sum(precisions) / total_q, 4),
            "mrr": round(sum(rrs) / total_q, 4),
            "mean_latency_ms": round(sum(latencies) / total_q, 2),
            "total_retrieved_chunks": total_retrieved,
            "total_relevant_chunks": total_relevant,
            "total_noise_chunks": total_noise,
            "noise_ratio_pct": round((total_noise / total_retrieved) * 100, 1)
        }

    print("\n================ TOP-K COMPARISON SUMMARY ================")
    print(f"| K   | Hit@K  | Precision@K | MRR    | Latency (ms) | Rel Chunks | Noise Chunks | Noise Ratio |")
    print(f"|-----+--------+-------------+--------+--------------+------------+--------------+-------------|")
    for k, res in results_by_k.items():
        print(f"| K={k} | {res['hit_at_k']:<6} | {res['precision_at_k']:<11} | {res['mrr']:<6} | {res['mean_latency_ms']:<12} | {res['total_relevant_chunks']:<10} | {res['total_noise_chunks']:<12} | {res['noise_ratio_pct']}%{' '*7} |")
    print("==========================================================\n")

    print("\n--- GT_15 (DTC Surveillance / Thyroglobulin) Analysis by K ---")
    for k, rows in gt15_analysis.items():
        print(f"\n[K={k} for GT_15]")
        for r in rows:
            print(f"  Rank {r['rank']}: Doc={r['doc']} (p.{r['page']}) | Relevant={r['is_relevant']} | Snippet: {r['snippet'][:90]}...")

    with open(os.path.join("evaluation", "top_k_validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_k, f, indent=2)

    return results_by_k


if __name__ == "__main__":
    validate_top_k_options()
