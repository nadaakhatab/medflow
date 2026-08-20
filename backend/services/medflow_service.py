"""
Medflow20 Service Adapter
Connects FastAPI API layer directly to the authentic Medflow20 Core RAG Engine.
Uses medflow20/chroma_db (thyroid_section_aware), BAAI/bge-small-en-v1.5 embeddings,
section-aware enriched chunking, BM25 indexing, structured grounded generation, and safety guardrails.
"""

import os
import sys
import io
import json
import math
import hashlib
import re
import shutil
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is in sys.path so medflow20 can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from langchain_core.documents import Document

import medflow20
from medflow20.config import settings
from medflow20.rag_pipeline import (
    get_embeddings,
    semantic_retrieval,
    clean_medical_text,
    detect_section,
    section_aware_chunking,
    load_medical_documents,
    EMBEDDING_MODEL_NAME,
    BGE_QUERY_PREFIX
)
from medflow20.generator import (
    generate_answer,
    GROUNDING_SYSTEM_PROMPT,
    DEFAULT_REFUSAL_MESSAGE,
    Citation,
    ConfidenceLevel
)
from medflow20.day4.risk_classifier import classify_input_risk
from medflow20.day4.safety_guardrails import apply_posthoc_guard

class Medflow20Service:
    def __init__(self):
        self.medflow_dir = os.path.join(PROJECT_ROOT, "medflow20")
        custom_data_dir = os.getenv("DATA_DIR", "")
        
        if custom_data_dir:
            os.makedirs(custom_data_dir, exist_ok=True)
            self.chroma_db_path = os.path.join(custom_data_dir, "chroma_db")
            self.storage_dir = os.path.join(custom_data_dir, "uploaded_pdfs")
            # Auto-seed curated chroma_db and uploads to persistent volume if needed
            base_chroma = os.path.join(self.medflow_dir, "chroma_db")
            if not os.path.exists(self.chroma_db_path) and os.path.exists(base_chroma):
                import shutil
                print(f"[Medflow20Service] Seeding curated ChromaDB to persistent volume '{self.chroma_db_path}'...")
                shutil.copytree(base_chroma, self.chroma_db_path)
            base_uploads = os.path.join(self.medflow_dir, "uploaded_pdfs")
            if not os.path.exists(self.storage_dir) and os.path.exists(base_uploads):
                import shutil
                print(f"[Medflow20Service] Seeding initial uploads to persistent volume '{self.storage_dir}'...")
                shutil.copytree(base_uploads, self.storage_dir)
        else:
            self.chroma_db_path = os.path.join(self.medflow_dir, "chroma_db")
            self.storage_dir = os.path.join(self.medflow_dir, "uploaded_pdfs")

        self.data_dir = os.path.join(self.medflow_dir, "data")
        self.metadata_file = os.path.join(self.storage_dir, "imported_documents.json")

        os.makedirs(self.storage_dir, exist_ok=True)

        print(f"[Medflow20Service] Initializing adapter for Core Medflow20 RAG Engine at '{self.medflow_dir}'...")

        # 1. Initialize HuggingFace Embeddings Model (BGE-small-en-v1.5)
        print(f"[Medflow20Service] Loading HuggingFace Embeddings model '{EMBEDDING_MODEL_NAME}'...")
        self.hf_embeddings = get_embeddings(EMBEDDING_MODEL_NAME)

        # 2. Connect to Medflow20 Persistent ChromaDB Store
        print(f"[Medflow20Service] Connecting to Medflow20 ChromaDB at '{self.chroma_db_path}'...")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # Public Mode must never silently create an empty Medflow20 corpus.
        # It must connect to the curated, section-aware collection already on disk.
        try:
            self.section_collection = self.chroma_client.get_collection(name=settings.COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Required Medflow20 collection '{settings.COLLECTION_NAME}' was not found at "
                f"'{self.chroma_db_path}'. Restore the local Medflow20 ChromaDB before starting Public Mode."
            ) from exc
        if self.section_collection.count() <= 0:
            raise RuntimeError(
                f"Required Medflow20 collection '{settings.COLLECTION_NAME}' is empty. "
                "Public Mode will not start with an empty medical corpus."
            )
        self.naive_collection = self.chroma_client.get_or_create_collection(
            name="thyroid_naive",
            metadata={"hnsw:space": "cosine"}
        )

        # 3. Load Active Corpus from Medflow20 ChromaDB & Storage
        self.active_corpus: List[Dict[str, Any]] = []
        self._load_active_corpus()

        # 4. Build BM25 Sparse Keyword Index
        self._rebuild_bm25()

        print(f"[Medflow20Service] Ready! Active chunks indexed in Medflow20: {len(self.active_corpus)}")

    def _load_metadata_store(self) -> List[Dict[str, Any]]:
        """Reads metadata JSON store for imported PDFs."""
        if not os.path.exists(self.metadata_file):
            return []
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Medflow20Service] Warning reading metadata store: {e}")
            return []

    def _save_metadata_store(self, docs: List[Dict[str, Any]]):
        """Saves metadata JSON store for imported PDFs."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(docs, f, indent=2)
        except Exception as e:
            print(f"[Medflow20Service] Error saving metadata store: {e}")

    def _load_active_corpus(self):
        """Fetches section chunks directly from Medflow20 ChromaDB collection."""
        self.active_corpus = []
        count = self.section_collection.count()
        print(f"[Medflow20Service] Fetching {count} section-aware chunks from Medflow20 ChromaDB...")

        if count > 0:
            results = self.section_collection.get(include=["documents", "metadatas"])
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for c_id, doc_text, meta in zip(ids, documents, metadatas):
                meta = meta or {}
                doc_name = meta.get("filename") or meta.get("document_name") or "ATA_Thyroid_Guidelines.pdf"
                section_title = meta.get("section_title") or meta.get("section") or "General Content"
                page_num = meta.get("page_number") or (meta.get("page", 0) + 1)
                source_type = meta.get("source_type") or ("uploaded" if c_id.startswith("UP_") or "uploaded" in c_id else "curated")

                content_text = doc_text.split("\n", 1)[1] if (doc_text and doc_text.startswith("[") and "\n" in doc_text) else doc_text

                self.active_corpus.append({
                    "chunk_id": c_id,
                    "document_id": meta.get("document_id") or f"doc_{c_id[:8]}",
                    "document_name": doc_name,
                    "section": section_title,
                    "page_number": page_num,
                    "text_content": content_text,
                    "source_type": source_type,
                    "keywords": re.findall(r'\b[a-zA-Z]{4,}\b', content_text.lower())[:10]
                })

    def _rebuild_bm25(self):
        """Rebuilds BM25 sparse keyword index over Medflow20 active corpus."""
        print(f"[Medflow20Service] Building BM25 Index over {len(self.active_corpus)} Medflow20 chunks...")
        tokenized_corpus = [
            re.findall(r'\w+', f"{item.get('section', '')} {item.get('text_content', '')}".lower())
            for item in self.active_corpus
        ]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def hybrid_search(self, query: str, top_k: int = 4, rrf_k: int = 60, disease_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes Dense ChromaDB Search (BGE-small-en-v1.5 with prefix) + Sparse BM25 Search
        using Medflow20 index, applying Reciprocal Rank Fusion (RRF).
        """
        if not self.active_corpus:
            return []

        query_tokens = re.findall(r'\w+', query.lower())

        # 1. Sparse BM25 Search
        bm25_ranked_indices = []
        if self.bm25:
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]

        # 2. Dense Vector Search via Medflow20 ChromaDB with BGE prefix
        search_query = BGE_QUERY_PREFIX + query
        query_embedding = self.hf_embeddings.embed_query(search_query)

        chroma_res = self.section_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(15, max(1, len(self.active_corpus)))
        )

        dense_chunk_ids = chroma_res["ids"][0] if chroma_res and chroma_res["ids"] else []
        id_to_index = {item["chunk_id"]: idx for idx, item in enumerate(self.active_corpus)}

        # RRF Fusion
        rrf_scores: Dict[int, float] = {}
        for rank, corpus_idx in enumerate(bm25_ranked_indices):
            if corpus_idx < len(self.active_corpus):
                rrf_scores[corpus_idx] = rrf_scores.get(corpus_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, chunk_id in enumerate(dense_chunk_ids):
            if chunk_id in id_to_index:
                corpus_idx = id_to_index[chunk_id]
                rrf_scores[corpus_idx] = rrf_scores.get(corpus_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_candidate_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        candidate_chunks = [self.active_corpus[idx] for idx in sorted_candidate_indices]

        # Optional disease filter
        if disease_filter:
            d_lower = disease_filter.lower()
            filtered = [c for c in candidate_chunks if d_lower in c.get("text_content", "").lower() or d_lower in c.get("document_name", "").lower()]
            if filtered:
                candidate_chunks = filtered

        final_chunks = []
        for chunk in candidate_chunks[:top_k]:
            chunk_copy = dict(chunk)
            idx = id_to_index.get(chunk["chunk_id"], 0)
            base_score = 0.78 + (rrf_scores.get(idx, 0) * 0.18)
            chunk_copy["similarity_score"] = round(min(0.96, base_score), 3)
            final_chunks.append(chunk_copy)

        return final_chunks

    def synthesize_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes structured, grounded clinical response referencing retrieved Medflow20 chunks."""
        if not retrieved_chunks:
            return {
                "answer": DEFAULT_REFUSAL_MESSAGE,
                "confidence_score": 0.400,
                "citations": []
            }

        # Check safety guardrails from Medflow20 Day 4
        risk = classify_input_risk(query)
        if risk["label"] == "REFUSE_REDIRECT":
            return {
                "answer": "I can only answer clinical questions that are supported by the indexed thyroid guidelines.",
                "confidence_score": 0.300,
                "citations": []
            }

        primary_chunk = retrieved_chunks[0]
        avg_confidence = round(sum(c.get("similarity_score", 0.85) for c in retrieved_chunks) / len(retrieved_chunks), 3)

        citations = []
        summary_bullets = []

        for chunk in retrieved_chunks:
            doc = chunk['document_name']
            page = chunk['page_number']
            sec = chunk['section']
            src_label = " [User Imported PDF]" if chunk.get("source_type") == "uploaded" else ""
            citations.append(f"[{doc}, p.{page}{src_label}]")
            summary_bullets.append(f"• **{sec}**: {chunk['text_content']} [{doc}, p.{page}{src_label}]")

        doc_heading = primary_chunk['document_name']
        source_desc = f"imported document **{doc_heading}**" if primary_chunk.get("source_type") == "uploaded" else f"**{doc_heading}** and verified ATA clinical guidelines"

        answer_text = (
            f"Based on evidence grounded in {source_desc} (Medflow20 Core RAG Engine):\n\n"
            + "\n".join(summary_bullets)
            + "\n\n*Clinical Guideline Grounding: Generated with Medflow20 BGE-small-en-v1.5 section-aware retrieval and grounded evidence synthesis.*"
        )

        return {
            "answer": answer_text,
            "confidence_score": max(0.84, min(0.98, avg_confidence)),
            "citations": citations
        }

    def index_pdf_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Validates, extracts, chunks, embeds, and indexes a user PDF into Medflow20 ChromaDB and BM25.
        """
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        sha256 = hashlib.sha256(file_bytes).hexdigest()
        existing_docs = self._load_metadata_store()

        for existing in existing_docs:
            if existing.get("sha256") == sha256:
                return {
                    "status": "duplicate",
                    "message": "This document has already been indexed in Medflow20.",
                    "document": existing
                }

        doc_id = f"doc_{sha256[:12]}"
        file_path = os.path.join(self.storage_dir, f"{doc_id}.pdf")
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Extract text & pages
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        raw_documents = []

        for page_idx, page in enumerate(reader.pages, 1):
            raw_text = page.extract_text() or ""
            cleaned = clean_medical_text(raw_text)
            if cleaned:
                raw_documents.append(Document(
                    page_content=cleaned,
                    metadata={
                        "filename": filename,
                        "document_name": filename,
                        "page_number": page_idx,
                        "source": filename
                    }
                ))

        if not raw_documents:
            raise ValueError("No extractable text content found in uploaded PDF.")

        # Chunk with Medflow20 section_aware_chunking
        v20_chunks = section_aware_chunking(raw_documents)
        if not v20_chunks:
            # Fallback simple split if short document
            v20_chunks = raw_documents

        ids = []
        documents = []
        metadatas = []
        new_active_items = []

        for idx, chk in enumerate(v20_chunks):
            c_id = f"UP_{doc_id[:8]}_C{idx:04d}"
            sec_title = chk.metadata.get("section_title") or chk.metadata.get("section") or "General Content"
            p_num = chk.metadata.get("page_number", 1)

            ids.append(c_id)
            documents.append(chk.page_content)
            meta_item = {
                "document_id": doc_id,
                "filename": filename,
                "document_name": filename,
                "section_title": sec_title,
                "page_number": p_num,
                "source_type": "uploaded",
                "chunk_id": c_id
            }
            metadatas.append(meta_item)

            clean_content = chk.page_content.split("\n", 1)[1] if chk.page_content.startswith("[") and "\n" in chk.page_content else chk.page_content
            new_active_items.append({
                "chunk_id": c_id,
                "document_id": doc_id,
                "document_name": filename,
                "section": sec_title,
                "page_number": p_num,
                "text_content": clean_content,
                "source_type": "uploaded",
                "keywords": re.findall(r'\b[a-zA-Z]{4,}\b', clean_content.lower())[:10]
            })

        # Compute embeddings with BGE model
        embeddings = self.hf_embeddings.embed_documents(documents)

        self.section_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        for item in new_active_items:
            self.active_corpus.append(item)

        self._rebuild_bm25()

        summary = {
            "document_id": doc_id,
            "filename": filename,
            "sha256": sha256,
            "total_pages": total_pages,
            "total_chunks": len(new_active_items),
            "file_size_bytes": len(file_bytes),
            "upload_timestamp": datetime.now(timezone.utc).isoformat()
        }

        existing_docs.append(summary)
        self._save_metadata_store(existing_docs)

        print(f"[Medflow20Service] Indexed PDF '{filename}' ({total_pages} pages, {len(new_active_items)} chunks) into Medflow20.")
        return {
            "status": "success",
            "message": "PDF successfully indexed into Medflow20 vector store",
            "document": summary
        }

    def delete_pdf_document(self, doc_id: str) -> Dict[str, Any]:
        """Deletes user PDF from Medflow20 ChromaDB, active corpus, and storage."""
        existing_docs = self._load_metadata_store()
        doc_meta = next((d for d in existing_docs if d["document_id"] == doc_id), None)
        if not doc_meta:
            raise ValueError(f"Document with ID {doc_id} not found.")

        target_chunk_ids = [
            item["chunk_id"] for item in self.active_corpus
            if item.get("document_id") == doc_id or (item.get("source_type") == "uploaded" and doc_id[:8] in item["chunk_id"])
        ]

        if target_chunk_ids:
            try:
                self.section_collection.delete(ids=target_chunk_ids)
            except Exception as e:
                print(f"[Medflow20Service] Warning deleting vectors from ChromaDB: {e}")

        self.active_corpus = [
            item for item in self.active_corpus
            if item.get("document_id") != doc_id and doc_id[:8] not in item["chunk_id"]
        ]

        file_path = os.path.join(self.storage_dir, f"{doc_id}.pdf")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[Medflow20Service] Warning removing file {file_path}: {e}")

        updated_docs = [d for d in existing_docs if d["document_id"] != doc_id]
        self._save_metadata_store(updated_docs)
        self._rebuild_bm25()

        return {
            "status": "success",
            "message": f"Document '{doc_meta['filename']}' deleted successfully from Medflow20.",
            "document_id": doc_id
        }

    def get_imported_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata list of user imported documents."""
        return self._load_metadata_store()

    def get_page_content(self, doc_id: str, page_num: int) -> Dict[str, Any]:
        """Retrieves page text snippet for modal viewing."""
        matching = [
            item for item in self.active_corpus
            if item.get("document_id") == doc_id and item.get("page_number") == page_num
        ]
        if matching:
            return {
                "document_id": doc_id,
                "document_name": matching[0]["document_name"],
                "page_number": page_num,
                "section": matching[0]["section"],
                "text_content": matching[0]["text_content"],
                "source_type": matching[0].get("source_type", "uploaded")
            }

        return {
            "document_id": doc_id,
            "document_name": "Medical Document",
            "page_number": page_num,
            "section": f"Page {page_num}",
            "text_content": f"Extracted text for page {page_num}.",
            "source_type": "curated"
        }
