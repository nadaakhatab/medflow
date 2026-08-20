"""Optional NON-DESTRUCTIVE builder for the exact frozen Day 2 token-aware index.

It writes to chroma_db_day2_frozen/ and never deletes or overwrites the original
chroma_db/. This file is only needed when the live persisted index fails the Day 4
audit.
"""
from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

if __package__ in (None, ""):
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from day4 import config
else:
    from . import config


def build_frozen_index(rebuild: bool = False):
    # Heavy dependencies are deliberately imported lazily so project audits/tests
    # remain runnable even on machines that have not activated the project venv.
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from transformers import AutoTokenizer

    target = config.FROZEN_DAY4_PERSIST_DIR
    if target.exists() and any(target.iterdir()) and not rebuild:
        raise FileExistsError(
            f"{target} already exists. Pass rebuild=True only if you intentionally want to replace the separate Day 4 frozen index."
        )
    if rebuild and target.exists():
        shutil.rmtree(target)

    loader = PyPDFDirectoryLoader(str(config.PROJECT_ROOT / "data"))
    raw_pages = loader.load()
    pages = []
    for d in raw_pages:
        m = dict(d.metadata or {})
        filename = os.path.basename(m.get("source", ""))
        m["filename"] = filename
        m["document_name"] = filename
        m["page_number"] = int(m.get("page", 0)) + 1
        pages.append(Document(page_content=d.page_content, metadata=m))

    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"C{i:05d}"
        chunk.metadata["chunk_tokens_target"] = config.CHUNK_SIZE_TOKENS
        chunk.metadata["overlap_tokens_target"] = config.CHUNK_OVERLAP_TOKENS

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.FROZEN_DAY4_COLLECTION_NAME,
        persist_directory=str(target),
        collection_metadata={"hnsw:space": "cosine"},
    )
    manifest = {
        "purpose": "non-destructive exact Day 2 frozen-index rebuild for Day 4 evaluation",
        "collection_name": config.FROZEN_DAY4_COLLECTION_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "normalize_embeddings": True,
        "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "indexed_chunks": len(chunks),
        "expected_indexed_chunks": config.EXPECTED_INDEXED_CHUNKS,
        "top_k": config.TOP_K,
        "source_pdf_count": len(list((config.PROJECT_ROOT / "data").glob("*.pdf"))),
    }
    (target / "frozen_index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store, len(chunks)


if __name__ == "__main__":
    _, count = build_frozen_index(rebuild=False)
    print(f"Built separate frozen Day 2 index with {count} chunks.")
    print(f"Expected from frozen config: {config.EXPECTED_INDEXED_CHUNKS}")
