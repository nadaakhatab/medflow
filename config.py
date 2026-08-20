import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class Settings:
    """MedFlow Application Configuration Settings."""
    
    # LLM Configuration
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    TEMPERATURE: float = 0.0
    
    # Retrieval Configuration (Day 2 Frozen Final Config)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
    PERSIST_DIR: str = "chroma_db"
    COLLECTION_NAME: str = "thyroid_section_aware"
    TOP_K: int = 4

    # Day 3 Refusal & Grounding Configuration
    # Note: CONFIDENCE_THRESHOLD is an illustrative baseline gating threshold for Day 3 refusal triggers.
    # Scientific calibration of this threshold using precision@k will be conducted in Day 4.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

settings = Settings()
