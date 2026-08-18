import os
import re
import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from json_repair import loads as json_repair_loads  # type: ignore
except ImportError:
    def json_repair_loads(s: str):
        # Fallback basic JSON extractor
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(s)

from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)

# Schema path for JSON Schema validation
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(_CURR_DIR, "schema", "response_schema.json")
if not os.path.exists(SCHEMA_PATH):
    SCHEMA_PATH = os.path.join(os.path.dirname(_CURR_DIR), "schema", "response_schema.json")

DEFAULT_REFUSAL_MESSAGE = "I couldn't find enough information in the indexed guideline to answer this confidently."

GROUNDING_SYSTEM_PROMPT = """You are a citation-bound clinical evidence assistant.

RULES - follow every one exactly:

1. Answer ONLY using the context passages provided below.
   Never use outside medical knowledge.

2. Every claim in your "recommendation" must be directly supported
   by the "evidence" you cite.

3. You MUST return your answer as JSON matching exactly this structure:

{
    "recommendation": "...",
    "evidence": "...",
    "citations": [
        {
            "document": "...",
            "section": "...",
            "page": N
        }
    ],
    "confidence": "high" | "medium" | "low" | "insufficient"
}

4. If the context does not contain enough information to answer
   confidently:

   - Set confidence to "insufficient"
   - Leave evidence empty
   - Leave citations empty
   - Write a plain refusal in recommendation
   - Never guess or supplement the answer from outside knowledge

5. Never invent a citation.

6. Never soften a refusal into a partial guess.

7. Return ONLY the JSON object.
   Do not return Markdown.
   Do not return explanations outside the JSON."""


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class Citation(BaseModel):
    document: str = Field(description="Name of the source document")
    section: str = Field(default="", description="Section header or title")
    page: int = Field(default=1, description="Page number in the document")

    @field_validator("document", mode="before")
    @classmethod
    def clean_document(cls, v: Any) -> str:
        if v is None:
            return "Unknown"
        s = str(v).strip()
        return s if s and s.lower() != "none" else "Unknown"

    @field_validator("section", mode="before")
    @classmethod
    def clean_section(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        return s if s.lower() != "none" else ""

    @field_validator("page", mode="before")
    @classmethod
    def clean_page(cls, v: Any) -> int:
        if v is None:
            return 1
        try:
            val = int(v)
            return val if val > 0 else 1
        except (ValueError, TypeError):
            return 1


class ClinicalAnswer(BaseModel):
    recommendation: str = Field(description="Grounded clinical recommendation or refusal")
    evidence: str = Field(default="", description="Verbatim or summarized supporting evidence from context")
    citations: List[Citation] = Field(default_factory=list, description="List of validated citations")
    confidence: ConfidenceLevel = Field(description="Confidence rating: high, medium, low, or insufficient")

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: Any) -> ConfidenceLevel:
        if isinstance(v, ConfidenceLevel):
            return v
        if isinstance(v, str):
            v_lower = v.strip().lower()
            for member in ConfidenceLevel:
                if member.value == v_lower:
                    return member
        return ConfidenceLevel.INSUFFICIENT

    @model_validator(mode="after")
    def enforce_schema_invariants(self) -> "ClinicalAnswer":
        if self.confidence == ConfidenceLevel.INSUFFICIENT:
            # If insufficient, evidence and citations must be empty
            self.evidence = ""
            self.citations = []
            if not self.recommendation or not self.recommendation.strip():
                self.recommendation = DEFAULT_REFUSAL_MESSAGE
        else:
            # Non-insufficient answers MUST have supporting evidence and at least one citation
            if not self.evidence or not self.evidence.strip():
                raise ValueError("Non-insufficient answer must have supporting evidence.")
            if not self.citations or len(self.citations) == 0:
                raise ValueError("Non-insufficient answer must have at least one citation.")
        return self


def validate_with_json_schema(data: Dict[str, Any]) -> bool:
    """Validates response dictionary against schema/response_schema.json using jsonschema."""
    if os.path.exists(SCHEMA_PATH):
        try:
            import jsonschema
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema = json.load(f)
            jsonschema.validate(instance=data, schema=schema)
            return True
        except Exception as e:
            logger.error(f"JSON Schema validation error: {e}")
            raise
    return True


def extract_chunk_metadata(chunk: Union[Document, Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts normalized metadata and content from either a LangChain Document or dictionary."""
    if isinstance(chunk, Document):
        meta = dict(chunk.metadata or {})
        content = chunk.page_content or ""
    elif isinstance(chunk, dict):
        meta = dict(chunk.get("metadata") or {})
        # Merge top-level fields into meta if not present
        for key in ["document_name", "filename", "source", "page_number", "page", "section_title", "section", "Header 1", "Header 2", "Header 3", "similarity_score"]:
            if key in chunk and key not in meta:
                meta[key] = chunk[key]
        content = chunk.get("retrieved_passage") or chunk.get("page_content") or chunk.get("text") or chunk.get("content") or ""
    else:
        meta = {}
        content = str(chunk)

    # Document Name
    doc_name = meta.get("document_name") or meta.get("filename") or meta.get("source") or ""
    if doc_name and ("/" in str(doc_name) or "\\" in str(doc_name)):
        doc_name = os.path.basename(str(doc_name))
    doc_name = str(doc_name).strip() if doc_name else "Unknown"
    if doc_name.lower() == "none":
        doc_name = "Unknown"

    # Page Number
    raw_page = meta.get("page_number")
    if raw_page is None:
        raw_page = meta.get("page")
        if raw_page is not None:
            try:
                raw_page = int(raw_page) + 1  # 0-indexed fallback
            except (ValueError, TypeError):
                raw_page = 1
    try:
        page_num = int(raw_page) if raw_page is not None else 1
        page_num = max(1, page_num)
    except (ValueError, TypeError):
        page_num = 1

    # Section / Headers
    header_parts = []
    for h in ["Header 1", "Header 2", "Header 3"]:
        val = meta.get(h)
        if val and str(val).strip() and str(val).strip().lower() != "none":
            header_parts.append(str(val).strip())
    
    if header_parts:
        section_info = " > ".join(header_parts)
    else:
        sec = meta.get("section_title") or meta.get("section") or ""
        sec = str(sec).strip()
        section_info = sec if sec and sec.lower() not in ("none", "general content", "n/a") else "General Content"

    sim_score = meta.get("similarity_score")
    try:
        sim_score = float(sim_score) if sim_score is not None else None
    except (ValueError, TypeError):
        sim_score = None

    return {
        "document_name": doc_name,
        "page_number": page_num,
        "section_info": section_info,
        "content": content.strip(),
        "similarity_score": sim_score
    }


def build_prompt(question: str, retrieved_chunks: List[Union[Document, Dict[str, Any]]]) -> str:
    """Builds the grounded prompt containing structured context and user question."""
    context_blocks = []
    for chunk in retrieved_chunks:
        extracted = extract_chunk_metadata(chunk)
        if not extracted["content"]:
            continue
        
        doc_line = f"[Document: {extracted['document_name']}]"
        page_line = f"[Page: {extracted['page_number']}]"
        section_line = f"[Section: {extracted['section_info']}]" if extracted["section_info"] else "[Section: General Content]"
        
        block = f"{doc_line}\n{page_line}\n{section_line}\n\n{extracted['content']}"
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""{GROUNDING_SYSTEM_PROMPT}

Context:
{context_str}

Question:
{question}

Respond with the JSON object described above, nothing else."""


def validate_citations(
    citations: List[Citation],
    retrieved_chunks: List[Union[Document, Dict[str, Any]]]
) -> List[Citation]:
    """Validates citations against the metadata of retrieved chunks to ensure strict citation integrity."""
    if not retrieved_chunks:
        return []

    available_sources = []
    for chunk in retrieved_chunks:
        meta = extract_chunk_metadata(chunk)
        available_sources.append(meta)

    validated = []
    for cit in citations:
        cit_doc = cit.document.lower().replace("_", " ").replace(".pdf", "")
        cit_page = cit.page

        # Verify if citation matches at least one retrieved chunk
        matched = False
        for src in available_sources:
            src_doc = src["document_name"].lower().replace("_", " ").replace(".pdf", "")
            src_page = src["page_number"]
            
            doc_matches = (cit_doc in src_doc) or (src_doc in cit_doc) or (cit.document.lower() == src["document_name"].lower())
            # Page match allows exact or +/- 1 tolerance for index alignment
            page_matches = abs(cit_page - src_page) <= 1

            if doc_matches and page_matches:
                matched = True
                # Normalize citation document name to the official retrieved chunk document name
                validated.append(Citation(
                    document=src["document_name"],
                    section=cit.section or src["section_info"],
                    page=src["page_number"]
                ))
                break

        if not matched:
            logger.warning(f"Citation dropped due to failing grounding check: {cit.model_dump()}")

    return validated


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None
):
    """Initializes and returns the configured LLM client (Groq by default)."""
    chosen_model = model or settings.GROQ_MODEL
    key = api_key or settings.GROQ_API_KEY

    try:
        from langchain_groq import ChatGroq  # type: ignore
        return ChatGroq(
            model=chosen_model,
            temperature=temperature,
            api_key=key if key else "dummy_key_for_testing"
        )
    except Exception as e:
        logger.warning(f"Could not initialize ChatGroq: {e}. Falling back to generic LLM wrapper.")
        raise


def parse_and_validate_llm_response(
    raw_content: str,
    retrieved_chunks: List[Union[Document, Dict[str, Any]]]
) -> ClinicalAnswer:
    """Parses raw LLM string, repairs JSON if needed, validates schema with Pydantic and checks citation integrity."""
    # 1. Clean markdown code fences if present
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # 2. Parse JSON using json_repair
    try:
        parsed_dict = json_repair_loads(cleaned)
        if isinstance(parsed_dict, str):
            # Sometimes json_repair returns a string if parsing was tricky
            parsed_dict = json.loads(parsed_dict)
    except Exception as e:
        logger.error(f"Failed to parse LLM JSON response: {e}. Raw content: {raw_content[:200]}")
        return ClinicalAnswer(
            recommendation=DEFAULT_REFUSAL_MESSAGE,
            evidence="",
            citations=[],
            confidence=ConfidenceLevel.INSUFFICIENT
        )

    if not isinstance(parsed_dict, dict):
        return ClinicalAnswer(
            recommendation=DEFAULT_REFUSAL_MESSAGE,
            evidence="",
            citations=[],
            confidence=ConfidenceLevel.INSUFFICIENT
        )

    # 3. Validate with Pydantic model
    try:
        answer = ClinicalAnswer.model_validate(parsed_dict)
    except Exception as e:
        logger.warning(f"Pydantic schema validation failed on LLM output: {e}. Falling back to insufficient refusal.")
        return ClinicalAnswer(
            recommendation=DEFAULT_REFUSAL_MESSAGE,
            evidence="",
            citations=[],
            confidence=ConfidenceLevel.INSUFFICIENT
        )

    # 4. Citation integrity check against retrieved chunks
    if answer.confidence != ConfidenceLevel.INSUFFICIENT and answer.citations:
        valid_citations = validate_citations(answer.citations, retrieved_chunks)
        if not valid_citations and answer.citations:
            # All citations fabricated or ungrounded -> enforce safety refusal
            logger.warning("All citations failed grounding validation against retrieved evidence.")
            return ClinicalAnswer(
                recommendation=DEFAULT_REFUSAL_MESSAGE,
                evidence="",
                citations=[],
                confidence=ConfidenceLevel.INSUFFICIENT
            )
        answer.citations = valid_citations

        # Re-check schema invariant after citation filtering
        if not answer.citations:
            return ClinicalAnswer(
                recommendation=DEFAULT_REFUSAL_MESSAGE,
                evidence="",
                citations=[],
                confidence=ConfidenceLevel.INSUFFICIENT
            )

    return answer


def generate_answer(
    query: str,
    retrieved_chunks: List[Union[Document, Dict[str, Any]]],
    llm: Optional[Any] = None,
    confidence_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Generates a citation-grounded clinical answer strictly from retrieved evidence.
    
    Includes explicit retrieval score threshold gating for Day 3 refusal mechanism.
    
    Returns a dictionary matching the schema:
    {
      "recommendation": "string",
      "evidence": "string",
      "citations": [{"document": "string", "section": "string", "page": 1}],
      "confidence": "high" | "medium" | "low" | "insufficient"
    }
    """
    threshold = confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD

    # 1. Deterministic check for empty or missing chunks
    if not retrieved_chunks:
        ans = ClinicalAnswer(
            recommendation=DEFAULT_REFUSAL_MESSAGE,
            evidence="",
            citations=[],
            confidence=ConfidenceLevel.INSUFFICIENT
        ).model_dump()
        validate_with_json_schema(ans)
        return ans

    # 2. Check retrieval similarity score threshold if available
    scores = []
    for c in retrieved_chunks:
        meta = extract_chunk_metadata(c)
        if meta.get("similarity_score") is not None:
            scores.append(meta["similarity_score"])

    if scores:
        top_score = max(scores)
        if top_score < threshold:
            logger.info(f"Top retrieval similarity score ({top_score:.4f}) below threshold ({threshold:.4f}). Triggering refusal path.")
            ans = ClinicalAnswer(
                recommendation=DEFAULT_REFUSAL_MESSAGE,
                evidence="",
                citations=[],
                confidence=ConfidenceLevel.INSUFFICIENT
            ).model_dump()
            validate_with_json_schema(ans)
            return ans

    # Check if all chunks have empty content
    has_valid_content = any(
        bool(extract_chunk_metadata(c)["content"]) for c in retrieved_chunks
    )
    if not has_valid_content:
        ans = ClinicalAnswer(
            recommendation=DEFAULT_REFUSAL_MESSAGE,
            evidence="",
            citations=[],
            confidence=ConfidenceLevel.INSUFFICIENT
        ).model_dump()
        validate_with_json_schema(ans)
        return ans

    # 3. Build grounded prompt
    prompt = build_prompt(question=query, retrieved_chunks=retrieved_chunks)

    # 4. Initialize or use provided LLM
    if llm is None:
        try:
            llm = get_llm()
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            ans = ClinicalAnswer(
                recommendation=DEFAULT_REFUSAL_MESSAGE,
                evidence="",
                citations=[],
                confidence=ConfidenceLevel.INSUFFICIENT
            ).model_dump()
            validate_with_json_schema(ans)
            return ans

    # 5. Invoke LLM safely
    try:
        messages = [
            ("system", GROUNDING_SYSTEM_PROMPT),
            ("human", prompt)
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        ans = ClinicalAnswer(
            recommendation=DEFAULT_REFUSAL_MESSAGE,
            evidence="",
            citations=[],
            confidence=ConfidenceLevel.INSUFFICIENT
        ).model_dump()
        validate_with_json_schema(ans)
        return ans

    # 6. Parse, repair, validate schema and citations
    answer = parse_and_validate_llm_response(content, retrieved_chunks)
    result = answer.model_dump()
    validate_with_json_schema(result)
    return result
