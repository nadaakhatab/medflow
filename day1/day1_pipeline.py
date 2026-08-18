import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from day1.ingest import load_pdfs, naive_chunk_documents, section_aware_chunk_documents, build_index
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERIES = [
    "How is hypothyroidism diagnosed?",
    "How is Hashimoto's disease diagnosed?",
    "How is hyperthyroidism diagnosed?",
    "What are the treatment options for Graves disease?",
    "How should a thyroid nodule be evaluated?",
    "How is differentiated thyroid cancer managed?"
]


def run_day1_baseline():
    print("=======================================================")
    print("RUNNING DAY 1 DOCUMENT INGESTION & BASELINE RETRIEVAL")
    print("=======================================================")
    
    docs = load_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} pages.")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        collection_name="thyroid_section_aware",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    for i, query in enumerate(QUERIES, 1):
        print(f"\n--- QUERY {i}: {query} ---")
        results = vectorstore.similarity_search_with_score(query, k=3)
        for rank, (doc, dist) in enumerate(results, 1):
            sim = max(0.0, min(1.0, 1.0 - dist))
            doc_name = doc.metadata.get("document_name", "Unknown")
            page_num = doc.metadata.get("page_number", "Unknown")
            sec = doc.metadata.get("section_title", "General Content")
            print(f"[{rank}] {doc_name} (Page {page_num}) | Section: {sec} | Sim: {sim:.4f}")
            print(f"    Passage: {doc.page_content[:200].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    run_day1_baseline()
