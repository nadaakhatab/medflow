import os
from typing import Optional
from dotenv import load_dotenv

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))

load_dotenv(os.path.join(_CURR_DIR, ".env"))
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

class Day3Settings:
    """Day 3 Grounded Generation & Citation Configuration."""
    
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    TEMPERATURE: float = 0.0
    
    SCHEMA_PATH: str = os.path.join(_CURR_DIR, "schema", "response_schema.json") if os.path.exists(os.path.join(_CURR_DIR, "schema", "response_schema.json")) else os.path.join(_ROOT_DIR, "schema", "response_schema.json")
    REFUSAL_BENCHMARK_PATH: str = os.path.join(_CURR_DIR, "evaluation", "day3_refusal_test_cases.json") if os.path.exists(os.path.join(_CURR_DIR, "evaluation", "day3_refusal_test_cases.json")) else os.path.join(_ROOT_DIR, "evaluation", "day3_refusal_test_cases.json")
    
    PERSIST_DIR: str = os.path.join(_CURR_DIR, "chroma_db") if os.path.exists(os.path.join(_CURR_DIR, "chroma_db")) else os.path.join(_ROOT_DIR, "chroma_db")
    COLLECTION_NAME: str = "thyroid_section_aware"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
    TOP_K: int = 4
    
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

settings = Day3Settings()
