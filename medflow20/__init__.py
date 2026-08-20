"""
Medflow20 - Validated Evidence-Grounded Medical RAG for Thyroid Diseases
Day 1-4 Complete Production Architecture
"""

from .config import settings
from .rag_pipeline import (
    get_embeddings,
    semantic_retrieval,
    clean_medical_text,
    detect_section,
    section_aware_chunking,
    load_medical_documents,
    create_vector_store,
    ask_clinical_question,
    EMBEDDING_MODEL_NAME,
    BGE_QUERY_PREFIX
)
from .generator import generate_answer, ConfidenceLevel, Citation

__version__ = "2.0.0"
__all__ = [
    "settings",
    "get_embeddings",
    "semantic_retrieval",
    "clean_medical_text",
    "detect_section",
    "section_aware_chunking",
    "load_medical_documents",
    "create_vector_store",
    "ask_clinical_question",
    "generate_answer",
    "ConfidenceLevel",
    "Citation",
    "EMBEDDING_MODEL_NAME",
    "BGE_QUERY_PREFIX",
]
