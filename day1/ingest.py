import os
import re
import sys
from collections import defaultdict
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import settings


def clean_medical_text(text: str) -> str:
    """Cleans extracted PDF text from noise, headers, and encoding artifacts."""
    text = text.replace("\xa0", " ").replace("\x00", " ")
    text = re.sub(r"This page and its contents are Copyright.*?(?:\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.thyroid\.org", "", text, flags=re.IGNORECASE)
    text = re.sub(r"American Thyroid Association", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_pdfs(directory_path: str = "data") -> List[Document]:
    """Loads all PDF documents from directory, sorting by filename and page number."""
    if not os.path.isabs(directory_path) and not os.path.exists(directory_path):
        parent_data = os.path.join(os.path.dirname(__file__), "..", directory_path)
        if os.path.exists(parent_data):
            directory_path = parent_data

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


def naive_chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """Naive fixed-size recursive character chunking."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"N{i:05d}"
        chunk.metadata["chunking_strategy"] = "naive_fixed_size"
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


def section_aware_chunk_documents(
    documents: List[Document],
    chunk_size: int = 550,
    chunk_overlap: int = 70
) -> List[Document]:
    """Section-aware chunking enriched with document and section metadata headers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )

    chunks = []
    chunk_counter = 0
    last_section = defaultdict(lambda: "General Content")

    for doc in documents:
        filename = doc.metadata.get("filename", "")
        page_chunks = splitter.split_text(doc.page_content)

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

    return chunks


def build_index(
    chunks: List[Document],
    collection_name: str = "thyroid_section_aware",
    persist_dir: str = "chroma_db",
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> Chroma:
    """Builds and persists a ChromaDB vector store."""
    if not os.path.isabs(persist_dir) and not os.path.exists(persist_dir):
        parent_persist = os.path.join(os.path.dirname(__file__), "..", persist_dir)
        persist_dir = parent_persist

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store
