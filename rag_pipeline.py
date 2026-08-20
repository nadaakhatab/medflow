import os
import re
import shutil
from collections import defaultdict
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

# High-performance embedding model optimized for dense retrieval & high cosine similarity
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def clean_medical_text(text: str) -> str:
    """Cleans extracted PDF text from noise, headers, and encoding artifacts."""
    # Replace non-breaking spaces and null bytes
    text = text.replace("\xa0", " ").replace("\x00", " ")
    # Remove copyright footers and URL lines
    text = re.sub(r"This page and its contents are Copyright.*?(?:\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.thyroid\.org", "", text, flags=re.IGNORECASE)
    text = re.sub(r"American Thyroid Association", "", text, flags=re.IGNORECASE)
    # Remove excessive blank lines and whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_medical_documents(directory_path: str = DATA_DIR) -> List[Document]:
    """Loads all PDF documents from directory, sorting by filename and page number."""
    print(f"--> Loading PDFs from '{directory_path}'...", flush=True)
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    loader = PyPDFDirectoryLoader(directory_path)
    raw_docs = loader.load()
    processed_docs = []

    for doc in raw_docs:
        source_path = doc.metadata.get("source", "")
        filename = os.path.basename(source_path)
        page_num = int(doc.metadata.get("page", 0)) + 1
        cleaned_content = clean_medical_text(doc.page_content)

        if not cleaned_content:
            continue

        metadata = dict(doc.metadata)
        metadata["filename"] = filename
        metadata["document_name"] = filename
        metadata["page_number"] = page_num
        processed_docs.append(Document(page_content=cleaned_content, metadata=metadata))

    processed_docs.sort(
        key=lambda d: (d.metadata.get("filename", ""), d.metadata.get("page_number", 0))
    )
    print(f"--> Successfully loaded {len(processed_docs)} document pages.", flush=True)
    return processed_docs


def naive_fixed_size_chunking(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """Naive fixed-size recursive character chunking."""
    print("--> Performing Naive Fixed-Size Chunking...", flush=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"N{i:05d}"
        chunk.metadata["chunking_strategy"] = "naive_fixed_size"
    print(f"--> Naive Fixed-Size Chunking created {len(chunks)} chunks.", flush=True)
    return chunks


SECTION_PATTERNS = [
    ("Diagnosis & Tests", [
        r"\bhow\s+is\b.{0,80}\bdiagnos(?:ed|is)\b",
        r"\bdiagnosis\s+of\b",
        r"\bdiagnostic\s+(?:evaluation|testing|criteria|procedures)\b",
        r"\bbiochemical\s+evaluation\b",
        r"\bthyroid\s+function\s+tests\b",
        r"\btsh\b.{0,30}\b(?:t3|t4|free)\b",
        r"\bdetermination\s+of\s+etiology\b",
    ]),
    ("Treatment & Management", [
        r"\bhow\s+is\b.{0,80}\btreated\b",
        r"\btreatment\s+options?\b",
        r"\btreatment\s+of\b",
        r"\bmedical\s+therapy\b",
        r"\bantithyroid\s+drugs\b",
        r"\bradioactive\s+iodine\b",
        r"\blevothyroxine\b",
        r"\bmanagement\s+of\b",
        r"\bhow\s+should\b.{0,80}\bmanaged\b",
    ]),
    ("Symptoms & Clinical Presentation", [
        r"\bwhat\s+are\s+the\s+(?:clinical\s+)?symptoms\b",
        r"\bsigns\s+and\s+symptoms\b",
        r"\bsymptoms\s+of\b",
        r"\bclinical\s+presentation\b",
        r"\bmanifestations\b",
    ]),
    ("Evaluation & Workup", [
        r"\binitial\s+evaluation\b",
        r"\bevaluation\s+of\b",
        r"\bultrasound\s+(?:features|evaluation)\b",
        r"\bfine\s+needle\s+aspiration\b",
        r"\bfna\b",
    ]),
    ("Surgery", [
        r"\bthyroidectomy\b",
        r"\bsurgical\s+(?:management|treatment|procedure|options)\b",
        r"\bextent\s+of\s+(?:initial\s+)?surgery\b",
    ]),
    ("Surveillance & Follow-Up", [
        r"\bfollow[- ]?up\b",
        r"\bsurveillance\b",
        r"\blong[- ]term\s+monitoring\b",
    ]),
    ("Risk Stratification & Complications", [
        r"\brisk\s+stratification\b",
        r"\brecurrence\s+risk\b",
        r"\bcomplications?\b",
        r"\badverse\s+effects?\b",
    ]),
]


def detect_section(text: str, previous_section: str = "General Content") -> str:
    """Identifies the clinical section header based on normalized regex matching."""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    candidates = []
    for section_name, patterns in SECTION_PATTERNS:
        best_pos = None
        for pattern in patterns:
            m = re.search(pattern, normalized, flags=re.IGNORECASE)
            if m:
                pos = m.start()
                if best_pos is None or pos < best_pos:
                    best_pos = pos
        if best_pos is not None:
            candidates.append((best_pos, section_name))

    if not candidates:
        return previous_section

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def trim_reference_tail(text: str, page_number: int, total_pages: int):
    """Trims trailing bibliography/reference sections near the end of documents."""
    if total_pages <= 0 or page_number < max(3, int(total_pages * 0.55)):
        return text, False

    lines = text.splitlines()
    for i, raw in enumerate(lines):
        heading = re.sub(r"[^A-Za-z]", "", raw).upper()
        if heading in {"REFERENCES", "BIBLIOGRAPHY"}:
            kept = "\n".join(lines[:i]).strip()
            return kept, True

    return text, False


def section_aware_chunking(
    documents: List[Document],
    chunk_size: int = 550,
    chunk_overlap: int = 70
) -> List[Document]:
    """Section-aware chunking enriched with document and section metadata headers."""
    print("--> Performing Improved Section-Aware Chunking...", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )

    total_pages_by_doc = defaultdict(int)
    for doc in documents:
        filename = doc.metadata.get("filename", "")
        total_pages_by_doc[filename] = max(
            total_pages_by_doc[filename],
            int(doc.metadata.get("page_number", 0))
        )

    chunks = []
    chunk_counter = 0
    last_section = defaultdict(lambda: "General Content")
    references_started = set()

    for doc in documents:
        filename = doc.metadata.get("filename", "")
        page_number = int(doc.metadata.get("page_number", 0))
        total_pages = total_pages_by_doc[filename]

        if filename in references_started:
            continue

        page_text, refs_start_here = trim_reference_tail(
            doc.page_content,
            page_number,
            total_pages
        )

        if not page_text.strip():
            if refs_start_here:
                references_started.add(filename)
            continue

        page_chunks = splitter.split_text(page_text)

        for text in page_chunks:
            text = text.strip()
            if len(text) < 60:
                continue

            section = detect_section(text, last_section[filename])
            last_section[filename] = section

            metadata = dict(doc.metadata)
            metadata["section_title"] = section
            metadata["chunk_id"] = f"S{chunk_counter:05d}"
            metadata["chunking_strategy"] = "section_aware"

            # Context enrichment: Prepend section title and document topic to maximize retrieval similarity
            clean_doc_title = filename.replace(".pdf", "").replace("_", " ")
            if section != "General Content":
                enriched_text = f"[{clean_doc_title} | {section}]\n{text}"
            else:
                enriched_text = f"[{clean_doc_title}]\n{text}"

            chunks.append(Document(
                page_content=enriched_text,
                metadata=metadata
            ))
            chunk_counter += 1

        if refs_start_here:
            references_started.add(filename)

    print(f"--> Section-Aware Chunking created {len(chunks)} enriched chunks.", flush=True)
    return chunks


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Initializes HuggingFace embeddings with L2 normalization enabled."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(
    chunks: List[Document],
    collection_name: str,
    persist_directory: str = PERSIST_DIR,
    model_name: str = EMBEDDING_MODEL_NAME
) -> Chroma:
    """Builds and persists a ChromaDB vector store with cosine similarity space."""
    print(f"--> Initializing embedding model '{model_name}'...", flush=True)
    embeddings = get_embeddings(model_name)

    print(f"--> Storing {len(chunks)} vectors in ChromaDB collection '{collection_name}' (cosine space)...", flush=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"--> Successfully indexed vectors into '{collection_name}'.", flush=True)
    return vector_store


def semantic_retrieval(
    vector_store: Chroma,
    query: str,
    top_k: int = 3,
    similarity_threshold: Optional[float] = None,
    is_bge_model: bool = True
) -> List[Dict[str, Any]]:
    """Retrieves top_k relevant documents with cosine similarity calculation and optional filtering."""
    search_query = (BGE_QUERY_PREFIX + query) if (is_bge_model and "bge" in EMBEDDING_MODEL_NAME) else query
    results_with_scores = vector_store.similarity_search_with_score(search_query, k=top_k)
    formatted_results = []

    for doc, distance in results_with_scores:
        # For cosine space: distance = 1 - cosine_similarity => similarity = 1 - distance
        # For L2 space (fallback): similarity = 1 - (distance / 2.0)
        similarity_score = round(max(0.0, min(1.0, 1.0 - distance)), 4)

        if similarity_threshold is not None and similarity_score < similarity_threshold:
            continue

        doc_name = (
            doc.metadata.get("filename")
            or os.path.basename(doc.metadata.get("source", "Unknown"))
        )
        page_num = (
            doc.metadata.get("page_number")
            or (doc.metadata.get("page", 0) + 1)
        )
        formatted_results.append({
            "retrieved_passage": doc.page_content,
            "document_name": doc_name,
            "page_number": page_num,
            "similarity_score": similarity_score,
            "section_title": doc.metadata.get("section_title", "N/A"),
            "chunking_strategy": doc.metadata.get("chunking_strategy", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "metadata": doc.metadata
        })

    return formatted_results


def run_pipeline(
    query: str = "How is hyperthyroidism diagnosed?",
    rebuild_index: bool = False
):
    """Full execution pipeline for loading, chunking, indexing, and querying."""
    db_exists = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0
    embeddings = get_embeddings(EMBEDDING_MODEL_NAME)

    if rebuild_index or not db_exists:
        if os.path.exists(PERSIST_DIR):
            print(f"--> Refreshing ChromaDB directory at '{PERSIST_DIR}'...", flush=True)
            shutil.rmtree(PERSIST_DIR)

        docs = load_medical_documents()
        naive_chunks = naive_fixed_size_chunking(docs)
        section_chunks = section_aware_chunking(docs)

        naive_store = create_vector_store(naive_chunks, "thyroid_naive")
        section_store = create_vector_store(section_chunks, "thyroid_section_aware")
    else:
        print(f"--> Loading existing ChromaDB vector stores from '{PERSIST_DIR}'...", flush=True)
        naive_store = Chroma(
            collection_name="thyroid_naive",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )
        section_store = Chroma(
            collection_name="thyroid_section_aware",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )

    print("\n" + "=" * 90, flush=True)
    print(f"QUERY: {query}", flush=True)
    print(f"EMBEDDING MODEL: {EMBEDDING_MODEL_NAME}", flush=True)
    print("=" * 90 + "\n", flush=True)

    print("--- 1. NAIVE FIXED-SIZE RETRIEVAL ---", flush=True)
    naive_results = semantic_retrieval(naive_store, query, top_k=3)
    for idx, r in enumerate(naive_results, 1):
        print(f"\n[Result {idx}] | Similarity: {r['similarity_score']:.4f} | Chunk ID: {r['chunk_id']}")
        print(f"Document: {r['document_name']} (Page {r['page_number']})")
        print(f"Passage:  {r['retrieved_passage'][:280].replace(chr(10), ' ')}...")

    print("\n" + "-" * 90, flush=True)
    print("--- 2. ENHANCED SECTION-AWARE RETRIEVAL ---", flush=True)
    section_results = semantic_retrieval(section_store, query, top_k=3)
    for idx, r in enumerate(section_results, 1):
        print(f"\n[Result {idx}] | Similarity: {r['similarity_score']:.4f} | Section: {r['section_title']} | Chunk ID: {r['chunk_id']}")
        print(f"Document: {r['document_name']} (Page {r['page_number']})")
        print(f"Passage:  {r['retrieved_passage'][:280].replace(chr(10), ' ')}...")

    return naive_results, section_results


def ask_clinical_question(
    query: str,
    vector_store: Optional[Chroma] = None,
    top_k: int = 4,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """End-to-end clinical question answering: retrieves relevant evidence chunks and generates a citation-grounded response."""
    try:
        from day3.generator import generate_answer
    except ImportError:
        from generator import generate_answer

    if vector_store is None:
        embeddings = get_embeddings(EMBEDDING_MODEL_NAME)
        vector_store = Chroma(
            collection_name="thyroid_section_aware",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )

    retrieved = semantic_retrieval(vector_store, query, top_k=top_k)
    answer = generate_answer(query, retrieved, llm=llm)
    return answer


if __name__ == "__main__":
    run_pipeline()

