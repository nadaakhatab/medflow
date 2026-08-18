import os
import sys
import time
import json
import csv
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROUND_TRUTH_FILE = os.path.join("evaluation", "thyroid_ground_truth.json")


def is_chunk_relevant(doc_metadata: Dict[str, Any], gt_item: Dict[str, Any]) -> bool:
    """Checks if a retrieved chunk matches the expected document/page/acceptable sources."""
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
    # Allow +/- 1 page tolerance for PDF page offset variations
    page_matches = (min_page - 1) <= page_num <= (max_page + 1)

    # Primary exact match
    if doc_name.lower() == expected_doc.lower() and page_matches:
        return True

    # Acceptable alternative sources match
    for alt in alt_sources:
        if doc_name.lower() == alt.lower() and page_matches:
            return True
            
    # If primary document matches regardless of minor page shift
    if doc_name.lower() == expected_doc.lower() and (min_page - 2) <= page_num <= (max_page + 2):
        return True

    return False


def run_evaluation(
    collection_name: str = "thyroid_section_aware",
    model_name: str = BASELINE_MODEL,
    persist_dir: str = PERSIST_DIR,
    top_k: int = 5,
    output_prefix: str = "baseline"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"RUNNING RETRIEVAL EVALUATION: {output_prefix.upper()}")
    print(f"Collection: '{collection_name}' | Model: '{model_name}' | Top-K: {top_k}")
    print(f"=======================================================\n")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(ground_truth)} Ground Truth evaluation questions.")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    query_results = []
    latencies = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    precisions_at_3 = []
    precisions_at_5 = []
    reciprocal_ranks = []

    csv_rows = []

    for item in ground_truth:
        q_id = item["query_id"]
        q_text = item["question"]

        start_time = time.time()
        search_results = vector_store.similarity_search_with_score(q_text, k=top_k)
        latency_ms = (time.time() - start_time) * 1000.0
        latencies.append(latency_ms)

        retrieved_list = []
        first_relevant_rank = None
        relevant_in_top_3 = 0
        relevant_in_top_5 = 0

        for rank, (doc, dist) in enumerate(search_results, 1):
            sim = round(max(0.0, 1.0 - (dist / 2.0)), 4)
            meta = doc.metadata
            doc_name = meta.get("document_name") or meta.get("filename") or os.path.basename(meta.get("source", "Unknown"))
            page_num = meta.get("page_number") or (meta.get("page", 0) + 1)
            sec_title = meta.get("section_title", "N/A")
            chunk_id = meta.get("chunk_id", "N/A")

            is_rel = is_chunk_relevant(meta, item)
            if is_rel and first_relevant_rank is None:
                first_relevant_rank = rank

            if rank <= 3 and is_rel:
                relevant_in_top_3 += 1
            if rank <= 5 and is_rel:
                relevant_in_top_5 += 1

            retrieved_info = {
                "rank": rank,
                "document": doc_name,
                "page": page_num,
                "section": sec_title,
                "chunk_id": chunk_id,
                "similarity": sim,
                "is_relevant": is_rel,
                "snippet": doc.page_content[:200].replace("\n", " ")
            }
            retrieved_list.append(retrieved_info)

            csv_rows.append({
                "query_id": q_id,
                "condition": item["condition"],
                "intent": item["intent"],
                "question": q_text,
                "rank": rank,
                "document": doc_name,
                "page": page_num,
                "section": sec_title,
                "chunk_id": chunk_id,
                "similarity": sim,
                "is_relevant": "YES" if is_rel else "NO"
            })

        hit_1 = 1 if (first_relevant_rank == 1) else 0
        hit_3 = 1 if (first_relevant_rank is not None and first_relevant_rank <= 3) else 0
        hit_5 = 1 if (first_relevant_rank is not None and first_relevant_rank <= 5) else 0
        p_3 = relevant_in_top_3 / 3.0
        p_5 = relevant_in_top_5 / 5.0
        rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        hits_at_1.append(hit_1)
        hits_at_3.append(hit_3)
        hits_at_5.append(hit_5)
        precisions_at_3.append(p_3)
        precisions_at_5.append(p_5)
        reciprocal_ranks.append(rr)

        query_results.append({
            "query_id": q_id,
            "condition": item["condition"],
            "intent": item["intent"],
            "question": q_text,
            "latency_ms": round(latency_ms, 2),
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "hit_at_5": hit_5,
            "precision_at_3": round(p_3, 4),
            "precision_at_5": round(p_5, 4),
            "reciprocal_rank": round(rr, 4),
            "first_relevant_rank": first_relevant_rank,
            "retrieved": retrieved_list
        })

    # Summary Metrics
    total_q = len(ground_truth)
    summary = {
        "evaluation_name": output_prefix,
        "collection_name": collection_name,
        "model_name": model_name,
        "total_queries": total_q,
        "hit_at_1": round(sum(hits_at_1) / total_q, 4),
        "hit_at_3": round(sum(hits_at_3) / total_q, 4),
        "hit_at_5": round(sum(hits_at_5) / total_q, 4),
        "precision_at_3": round(sum(precisions_at_3) / total_q, 4),
        "precision_at_5": round(sum(precisions_at_5) / total_q, 4),
        "mrr": round(sum(reciprocal_ranks) / total_q, 4),
        "mean_latency_ms": round(sum(latencies) / total_q, 2),
        "detailed_queries": query_results
    }

    # Save to evaluation directory
    os.makedirs("evaluation", exist_ok=True)
    json_path = os.path.join("evaluation", f"{output_prefix}_results.json")
    csv_path = os.path.join("evaluation", f"{output_prefix}_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_id", "condition", "intent", "question", "rank", "document", "page", "section", "chunk_id", "similarity", "is_relevant"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n---------------- EVALUATION SUMMARY ----------------")
    print(f"Total Evaluated Queries: {total_q}")
    print(f"Hit@1:         {summary['hit_at_1']:.4f} ({summary['hit_at_1']*100:.1f}%)")
    print(f"Hit@3:         {summary['hit_at_3']:.4f} ({summary['hit_at_3']*100:.1f}%)")
    print(f"Hit@5:         {summary['hit_at_5']:.4f} ({summary['hit_at_5']*100:.1f}%)")
    print(f"Precision@3:   {summary['precision_at_3']:.4f}")
    print(f"Precision@5:   {summary['precision_at_5']:.4f}")
    print(f"MRR:           {summary['mrr']:.4f}")
    print(f"Mean Latency:  {summary['mean_latency_ms']} ms")
    print(f"Saved: {json_path} and {csv_path}")
    print("----------------------------------------------------\n")

    return summary


if __name__ == "__main__":
    run_evaluation(
        collection_name="thyroid_section_aware",
        model_name=BASELINE_MODEL,
        persist_dir=PERSIST_DIR,
        top_k=5,
        output_prefix="baseline"
    )
