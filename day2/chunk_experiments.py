import os
import sys
import time
import json
import csv
from collections import defaultdict
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
PERSIST_DIR = "chroma_db_experiments"
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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


def build_and_evaluate_chunk_config(
    docs: List[Document],
    config_name: str,
    splitter,
    embeddings,
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    print(f"\n---> Building and indexing chunk config: '{config_name}'...")
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{config_name}_{i:05d}"

    print(f"Created {len(chunks)} chunks for {config_name}. Indexing in ChromaDB...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config_name,
        persist_directory=PERSIST_DIR
    )

    latencies = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    p_at_3 = []
    p_at_5 = []
    rrs = []

    for item in ground_truth:
        q_text = item["question"]
        t0 = time.time()
        results = store.similarity_search_with_score(q_text, k=5)
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
        "config_name": config_name,
        "num_chunks": len(chunks),
        "hit_at_1": round(sum(hits_at_1) / total_q, 4),
        "hit_at_3": round(sum(hits_at_3) / total_q, 4),
        "hit_at_5": round(sum(hits_at_5) / total_q, 4),
        "p_at_3": round(sum(p_at_3) / total_q, 4),
        "p_at_5": round(sum(p_at_5) / total_q, 4),
        "mrr": round(sum(rrs) / total_q, 4),
        "latency_ms": round(sum(latencies) / total_q, 2)
    }

    print(f"Results for {config_name}: Chunks={len(chunks)} | Hit@3={metrics['hit_at_3']} | P@3={metrics['p_at_3']} | P@5={metrics['p_at_5']} | MRR={metrics['mrr']} | Latency={metrics['latency_ms']}ms")
    return metrics


def run_chunk_experiments():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    docs = load_raw_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name=BASELINE_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    hf_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)

    configs = [
        ("tokens_200_ov0", RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            hf_tokenizer, chunk_size=200, chunk_overlap=0
        )),
        ("tokens_400_ov50", RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            hf_tokenizer, chunk_size=400, chunk_overlap=50
        )),
        ("tokens_600_ov100", RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            hf_tokenizer, chunk_size=600, chunk_overlap=100
        )),
        ("chars_500_ov50_naive", RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        ))
    ]

    all_results = []
    for cfg_name, splitter in configs:
        res = build_and_evaluate_chunk_config(docs, cfg_name, splitter, embeddings, ground_truth)
        all_results.append(res)

    # Save to evaluation directory
    with open(os.path.join("evaluation", "chunking_experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join("evaluation", "chunking_experiment_results.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = ["config_name", "num_chunks", "hit_at_1", "hit_at_3", "hit_at_5", "p_at_3", "p_at_5", "mrr", "latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n================ CHUNKING EXPERIMENTS COMPLETE ================")
    for r in all_results:
        print(f"| {r['config_name']:<22} | Chunks: {r['num_chunks']:<5} | P@3: {r['p_at_3']:<6} | P@5: {r['p_at_5']:<6} | Hit@3: {r['hit_at_3']:<6} | MRR: {r['mrr']:<6} | Latency: {r['latency_ms']}ms |")
    print("===============================================================\n")


if __name__ == "__main__":
    run_chunk_experiments()
