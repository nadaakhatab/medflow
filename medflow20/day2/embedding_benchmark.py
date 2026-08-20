import os
import sys
import time
import json
import csv
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

DATA_DIR = "data"
PERSIST_DIR = "chroma_db_embeddings"
GROUND_TRUTH_FILE = os.path.join("evaluation", "thyroid_ground_truth.json")


def load_raw_documents() -> List[Document]:
    loader = PyPDFDirectoryLoader(DATA_DIR)
    raw_docs = loader.load()
    docs = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "")
        filename = os.path.basename(source)
        page_number = int(doc.metadata.get("page", 0)) + 1
        meta = dict(doc.metadata)
        meta["filename"] = filename
        meta["document_name"] = filename
        meta["page_number"] = page_number
        docs.append(Document(page_content=doc.page_content, metadata=meta))
    docs.sort(key=lambda d: (d.metadata.get("filename", ""), d.metadata.get("page_number", 0)))
    return docs


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


def benchmark_embedding_model(
    model_name: str,
    dim: int,
    chunks: List[Document],
    ground_truth: List[Dict[str, Any]],
    query_prefix: str = ""
) -> Dict[str, Any]:
    clean_name = model_name.replace("/", "_").replace("-", "_")
    print(f"\n=======================================================")
    print(f"BENCHMARKING EMBEDDING MODEL: {model_name} (Dim: {dim})")
    print(f"=======================================================")

    t_init_start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    init_time_s = round(time.time() - t_init_start, 2)

    t_idx_start = time.time()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"emb_{clean_name}",
        persist_directory=PERSIST_DIR
    )
    indexing_time_s = round(time.time() - t_idx_start, 2)
    print(f"Indexing completed in {indexing_time_s}s for {len(chunks)} chunks.")

    latencies = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    p_at_3 = []
    p_at_5 = []
    rrs = []

    for item in ground_truth:
        q_text = item["question"]
        if query_prefix:
            search_query = f"{query_prefix}{q_text}"
        else:
            search_query = q_text

        t0 = time.time()
        results = store.similarity_search_with_score(search_query, k=5)
        latencies.append((time.time() - t0) * 1000.0)

        first_rel = None
        rel_3 = 0
        rel_5 = 0

        for rank, (doc, dist) in enumerate(results, 1):
            is_rel = is_chunk_relevant(doc.metadata, item)
            if is_rel and first_rel is None:
                first_rel = rank
            if rank <= 3 and is_rel:
                rel_3 += 1
            if rank <= 5 and is_rel:
                rel_5 += 1

        hits_at_1.append(1 if first_rel == 1 else 0)
        hits_at_3.append(1 if (first_rel and first_rel <= 3) else 0)
        hits_at_5.append(1 if (first_rel and first_rel <= 5) else 0)
        p_at_3.append(rel_3 / 3.0)
        p_at_5.append(rel_5 / 5.0)
        rrs.append(1.0 / first_rel if first_rel else 0.0)

    total_q = len(ground_truth)
    metrics = {
        "model_name": model_name,
        "dimension": dim,
        "hit_at_1": round(sum(hits_at_1) / total_q, 4),
        "hit_at_3": round(sum(hits_at_3) / total_q, 4),
        "hit_at_5": round(sum(hits_at_5) / total_q, 4),
        "p_at_3": round(sum(p_at_3) / total_q, 4),
        "p_at_5": round(sum(p_at_5) / total_q, 4),
        "mrr": round(sum(rrs) / total_q, 4),
        "query_latency_ms": round(sum(latencies) / total_q, 2),
        "indexing_time_s": indexing_time_s
    }

    print(f"Metrics for {model_name}: Hit@1={metrics['hit_at_1']} | Hit@3={metrics['hit_at_3']} | P@3={metrics['p_at_3']} | P@5={metrics['p_at_5']} | MRR={metrics['mrr']} | Latency={metrics['query_latency_ms']}ms")
    return metrics


def run_embedding_benchmarks():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    docs = load_raw_documents()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=200, chunk_overlap=0
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"C{i:05d}"

    models = [
        ("sentence-transformers/all-MiniLM-L6-v2", 384, ""),
        ("sentence-transformers/all-mpnet-base-v2", 768, ""),
        ("BAAI/bge-small-en-v1.5", 384, "Represent this sentence for searching relevant passages: ")
    ]

    all_metrics = []
    for model_name, dim, prefix in models:
        m = benchmark_embedding_model(model_name, dim, chunks, ground_truth, prefix)
        all_metrics.append(m)

    os.makedirs("evaluation", exist_ok=True)
    with open(os.path.join("evaluation", "embedding_benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join("evaluation", "embedding_benchmark_results.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model_name", "dimension", "hit_at_1", "hit_at_3", "hit_at_5", "p_at_3", "p_at_5", "mrr", "query_latency_ms", "indexing_time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print("\n================ EMBEDDING BENCHMARK SUMMARY ================")
    for m in all_metrics:
        print(f"| {m['model_name']:<40} | Dim: {m['dimension']:<4} | Hit@1: {m['hit_at_1']:<6} | Hit@3: {m['hit_at_3']:<6} | P@3: {m['p_at_3']:<6} | MRR: {m['mrr']:<6} | Latency: {m['query_latency_ms']:<5}ms |")
    print("=============================================================\n")


if __name__ == "__main__":
    run_embedding_benchmarks()
