import os
from typing import Optional
from dotenv import load_dotenv

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))

load_dotenv(os.path.join(_CURR_DIR, ".env"))
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

class Day2Settings:
    """Day 2 Retrieval Optimization & Benchmarking Configuration."""
    
    DATA_DIR: str = os.path.join(_CURR_DIR, "data") if os.path.exists(os.path.join(_CURR_DIR, "data")) else os.path.join(_ROOT_DIR, "data")
    PERSIST_DIR: str = os.path.join(_CURR_DIR, "chroma_db") if os.path.exists(os.path.join(_CURR_DIR, "chroma_db")) else os.path.join(_ROOT_DIR, "chroma_db")
    GROUND_TRUTH_FILE: str = os.path.join(_CURR_DIR, "evaluation", "thyroid_ground_truth.json") if os.path.exists(os.path.join(_CURR_DIR, "evaluation", "thyroid_ground_truth.json")) else os.path.join(_ROOT_DIR, "evaluation", "thyroid_ground_truth.json")
    
    COLLECTION_NAME: str = "thyroid_section_aware"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
    
    TOP_K: int = 4
    CHUNK_SIZE_TOKENS: int = 200
    RERANKER_ENABLED: bool = False

settings = Day2Settings()
