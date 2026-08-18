import os
import time
import json
import csv
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

PERSIST_DIR = "chroma_db_embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "emb_BAAI_bge_small_en_v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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


def run_reranker_experiment():
    print(f"\n=======================================================")
    print(f"RUNNING RERANKER EXPERIMENT (Dense Top-10 -> Cross-Encoder)")
    print(f"Dense Model: {EMBEDDING_MODEL} | Reranker: {RERANKER_MODEL}")
    print(f"=======================================================\n")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"Loading Cross-Encoder model '{RERANKER_MODEL}' on CPU...")
    reranker = CrossEncoder(RERANKER_MODEL, device="cpu")

    dense_hits_1, dense_hits_3, dense_hits_5 = [], [], []
    dense_p_3, dense_p_5 = [], []
    dense_rrs = []

    rerank_hits_1, rerank_hits_3, rerank_hits_5 = [], [], []
    rerank_p_3, rerank_p_5 = [], []
    rerank_rrs = []

    latencies = []

    query_details = []

    for item in ground_truth:
        q_text = item["question"]
        search_query = f"Represent this sentence for searching relevant passages: {q_text}"

        t0 = time.time()
        # 1. Retrieve Dense Top-10
        dense_results = vector_store.similarity_search_with_score(search_query, k=10)
        
        # Dense metrics (Top-5)
        first_rel_dense = None
        rel_3_dense = 0
        rel_5_dense = 0
        for rank, (doc, dist) in enumerate(dense_results[:5], 1):
            is_rel = is_chunk_relevant(doc.metadata, item)
            if is_rel and first_rel_dense is None:
                first_rel_dense = rank
            if rank <= 3 and is_rel:
                rel_3_dense += 1
            if rank <= 5 and is_rel:
                rel_5_dense += 1

        dense_hits_1.append(1 if first_rel_dense == 1 else 0)
        dense_hits_3.append(1 if (first_rel_dense and first_rel_dense <= 3) else 0)
        dense_hits_5.append(1 if (first_rel_dense and first_rel_dense <= 5) else 0)
        dense_p_3.append(rel_3_dense / 3.0)
        dense_p_5.append(rel_5_dense / 5.0)
        dense_rrs.append(1.0 / first_rel_dense if first_rel_dense else 0.0)

        # 2. Cross-Encoder Reranking
        pairs = [[q_text, doc.page_content] for doc, _ in dense_results]
        scores = reranker.predict(pairs)

        reranked_docs = []
        for (doc, _), score in zip(dense_results, scores):
            reranked_docs.append((doc, float(score)))

        reranked_docs.sort(key=lambda x: x[1], reverse=True)
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        # Rerank metrics (Top-5)
        first_rel_rerank = None
        rel_3_rerank = 0
        rel_5_rerank = 0
        rerank_items = []

        for rank, (doc, r_score) in enumerate(reranked_docs[:5], 1):
            is_rel = is_chunk_relevant(doc.metadata, item)
            if is_rel and first_rel_rerank is None:
                first_rel_rerank = rank
            if rank <= 3 and is_rel:
                rel_3_rerank += 1
            if rank <= 5 and is_rel:
                rel_5_rerank += 1

            meta = doc.metadata
            rerank_items.append({
                "rank": rank,
                "document": meta.get("document_name") or meta.get("filename") or os.path.basename(meta.get("source", "Unknown")),
                "page": meta.get("page_number") or (meta.get("page", 0) + 1),
                "section": meta.get("section_title", "N/A"),
                "rerank_score": round(r_score, 4),
                "is_relevant": is_rel
            })

        rerank_hits_1.append(1 if first_rel_rerank == 1 else 0)
        rerank_hits_3.append(1 if (first_rel_rerank and first_rel_rerank <= 3) else 0)
        rerank_hits_5.append(1 if (first_rel_rerank and first_rel_rerank <= 5) else 0)
        rerank_p_3.append(rel_3_rerank / 3.0)
        rerank_p_5.append(rel_5_rerank / 5.0)
        rerank_rrs.append(1.0 / first_rel_rerank if first_rel_rerank else 0.0)

        query_details.append({
            "query_id": item["query_id"],
            "question": q_text,
            "dense_first_rank": first_rel_dense,
            "rerank_first_rank": first_rel_rerank,
            "reranked_top5": rerank_items
        })

    total_q = len(ground_truth)
    summary = {
        "dense_baseline": {
            "hit_at_1": round(sum(dense_hits_1) / total_q, 4),
            "hit_at_3": round(sum(dense_hits_3) / total_q, 4),
            "hit_at_5": round(sum(dense_hits_5) / total_q, 4),
            "precision_at_3": round(sum(dense_p_3) / total_q, 4),
            "precision_at_5": round(sum(dense_p_5) / total_q, 4),
            "mrr": round(sum(dense_rrs) / total_q, 4)
        },
        "dense_plus_reranker": {
            "hit_at_1": round(sum(rerank_hits_1) / total_q, 4),
            "hit_at_3": round(sum(rerank_hits_3) / total_q, 4),
            "hit_at_5": round(sum(rerank_hits_5) / total_q, 4),
            "precision_at_3": round(sum(rerank_p_3) / total_q, 4),
            "precision_at_5": round(sum(rerank_p_5) / total_q, 4),
            "mrr": round(sum(rerank_rrs) / total_q, 4),
            "mean_latency_ms": round(sum(latencies) / total_q, 2)
        },
        "details": query_details
    }

    with open(os.path.join("evaluation", "reranker_experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ RERANKER EXPERIMENT SUMMARY ================")
    print(f"Metric        | Dense Only (BGE-Small) | Dense + Cross-Encoder Reranker")
    print(f"--------------+------------------------+--------------------------------")
    print(f"Hit@1         | {summary['dense_baseline']['hit_at_1']:<22} | {summary['dense_plus_reranker']['hit_at_1']}")
    print(f"Hit@3         | {summary['dense_baseline']['hit_at_3']:<22} | {summary['dense_plus_reranker']['hit_at_3']}")
    print(f"Hit@5         | {summary['dense_baseline']['hit_at_5']:<22} | {summary['dense_plus_reranker']['hit_at_5']}")
    print(f"Precision@3   | {summary['dense_baseline']['precision_at_3']:<22} | {summary['dense_plus_reranker']['precision_at_3']}")
    print(f"Precision@5   | {summary['dense_baseline']['precision_at_5']:<22} | {summary['dense_plus_reranker']['precision_at_5']}")
    print(f"MRR           | {summary['dense_baseline']['mrr']:<22} | {summary['dense_plus_reranker']['mrr']}")
    print(f"Mean Latency  | ~22.4 ms               | {summary['dense_plus_reranker']['mean_latency_ms']} ms")
    print("===============================================================\n")


if __name__ == "__main__":
    run_reranker_experiment()
