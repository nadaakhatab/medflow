"""
MedFlow Medical AI Assistant - FastAPI Backend Service
University AI Competition Production Backend Architecture
Supports dynamic PDF uploading, vector database indexing, document deletion,
and grounded RAG querying over built-in & user-imported guidelines.
Extends lab interpretation to handle Post-Thyroidectomy / Athyreotic & Congenital Hypothyroidism.
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Path, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import sys
import asyncio

# Ensure project root and medflow20 are in sys.path so medflow20 can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MEDFLOW20_DIR = os.path.join(PROJECT_ROOT, "medflow20")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MEDFLOW20_DIR not in sys.path:
    sys.path.insert(0, MEDFLOW20_DIR)

from services.medflow_service import Medflow20Service
import models
from database import engine, SessionLocal, get_db
from sqlalchemy.orm import Session
from sqlalchemy import text
import auth
import analytics

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MedFlow Medical Assistant API",
    description="Specialized Medical RAG API wrapping Medflow20 Core Engine",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
        "null"
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(analytics.router)

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        # DB Migration: Ensure profession column exists
        try:
            db.execute(text("ALTER TABLE users ADD COLUMN profession VARCHAR DEFAULT 'Other'"))
            db.commit()
        except Exception:
            pass # Column already exists
            
        # Seed Admin User
        admin_email = os.getenv("INITIAL_ADMIN_EMAIL", "medflow@gmail.com")
        admin_password = os.getenv("INITIAL_ADMIN_PASSWORD", "medflow@2026")
        
        admin_user = db.query(models.User).filter(models.User.email == admin_email.lower()).first()
        if not admin_user:
            hashed_pw = auth.get_password_hash(admin_password)
            new_admin = models.User(
                full_name="Medflow Admin",
                email=admin_email.lower(),
                password_hash=hashed_pw,
                role="admin",
                profession="Admin",
                is_active=True
            )
            db.add(new_admin)
            db.commit()
            print(f"[FastAPI] Created initial admin account: {admin_email}")
        else:
            # Ensure the existing user has admin role and updated password hash
            admin_user.role = "admin"
            admin_user.password_hash = auth.get_password_hash(admin_password)
            db.commit()
            print(f"[FastAPI] Verified existing admin user {admin_email}.")
    finally:
        db.close()

# Instantiate live Medflow20 Core RAG Engine Adapter
print("[FastAPI] Initializing Medflow20 Core RAG Engine Service...")
rag_engine = Medflow20Service()
print("[FastAPI] Medflow20 Core RAG Engine Ready!")

# --- Pydantic Schemas ---

class RAGQueryRequest(BaseModel):
    query: str = Field(..., example="What are TSH targets for subclinical hypothyroidism in pregnancy?")
    top_k: int = Field(default=4, ge=1, le=10)
    disease_filter: Optional[str] = Field(default=None, example="Hypothyroidism")

class DocumentChunk(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    text_content: str
    similarity_score: float
    section: Optional[str] = None
    source_type: Optional[str] = "curated"

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    confidence_score: float
    retrieved_chunks: List[DocumentChunk]
    disclaimer: str

class LabInterpreterRequest(BaseModel):
    tsh: float = Field(..., description="mIU/L")
    free_t4: float = Field(..., description="ng/dL")
    free_t3: Optional[float] = Field(None, description="pg/mL")
    thyroid_status: Optional[str] = Field("functioning", description="functioning | removed_ablated | congenital | unknown")
    thyroid_removal_reason: Optional[str] = Field("unknown", description="cancer | graves | goiter_nodules | benign | rai | other | unknown")
    congenital_condition: Optional[str] = Field("unknown", description="agenesis | dysgenesis | ectopic | dyshormonogenesis | unknown | other")
    patient_age: Optional[float] = Field(None, description="Patient age number")
    patient_age_unit: Optional[str] = Field("years", description="days | weeks | months | years")
    current_replacement_therapy: Optional[bool] = Field(False, description="True if patient takes levothyroxine")
    levothyroxine_dose_mcg: Optional[float] = Field(None, description="Current prescribed dose in mcg")
    pregnancy_context: Optional[bool] = Field(False, description="True if patient is pregnant")
    calcium: Optional[float] = Field(None, description="mg/dL")
    pth: Optional[float] = Field(None, description="pg/mL")
    thyroglobulin: Optional[float] = Field(None, description="ng/mL")
    anti_tg: Optional[float] = Field(None, description="IU/mL")

class LabInterpretationResponse(BaseModel):
    pattern: str
    summary: str
    guideline_citation: str
    risk_level: str
    rag_guidance_chunk: Optional[str] = None


@app.get("/")
@app.get("/index.html")
async def root_endpoint(request: Request):
    """
    Unified entry point:
    - Serves the Single-Page Application (index.html) for web browser navigation.
    - Returns JSON status when explicitly queried with application/json.
    """
    accept = request.headers.get("accept", "")
    # If standard browser navigation or HTML requested, serve index.html
    if "text/html" in accept or request.url.path.endswith("index.html"):
        index_path = os.path.join(PROJECT_ROOT, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    
    # Otherwise return JSON health status
    return get_system_health()

@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    """Liveness & Readiness probe for cloud and container orchestrators"""
    return get_system_health()

def get_system_health():
    imported_docs = rag_engine.get_imported_documents()
    total_active_chunks = len(rag_engine.active_corpus)
    return {
        "status": "healthy",
        "ready": True,
        "service": "MedFlow Medical RAG API (Medflow20 Core Engine)",
        "vector_store": f"Medflow20 ChromaDB Persistent ({total_active_chunks} Active Section-Aware Chunks Indexed)",
        "imported_documents_count": len(imported_docs),
        "models": {
            "embeddings": "BAAI/bge-small-en-v1.5",
            "search_type": "Medflow20 Hybrid Dense (ChromaDB BGE) + Sparse (BM25) + RRF + Grounded Citations"
        }
    }

@app.post("/api/v1/query", response_model=RAGQueryResponse)
async def query_rag_engine(payload: RAGQueryRequest, current_user: models.User = Depends(auth.get_current_user)):
    """
    Live Hybrid RAG retrieval and synthesis endpoint combining ChromaDB + BM25 + Reranker.
    Searches both built-in guidelines and user-uploaded PDFs.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    retrieved = rag_engine.hybrid_search(payload.query, top_k=payload.top_k)
    synthesis = rag_engine.synthesize_answer(payload.query, retrieved)
    
    formatted_chunks = [
        DocumentChunk(
            chunk_id=c.get("chunk_id", "chk_unknown"),
            document_name=c.get("document_name", "ATA_Thyroid_Guidelines_2023.pdf"),
            page_number=c.get("page_number", 1),
            text_content=c.get("text_content", ""),
            similarity_score=c.get("similarity_score", 0.85),
            section=c.get("section", ""),
            source_type=c.get("source_type", "curated")
        )
        for c in retrieved
    ]

    return RAGQueryResponse(
        query=payload.query,
        answer=synthesis["answer"],
        confidence_score=synthesis["confidence_score"],
        retrieved_chunks=formatted_chunks,
        disclaimer="Medical RAG Assistant output for educational research and decision support demonstration only. Consult a licensed clinician."
    )

@app.post("/api/v1/query/stream")
async def stream_rag_engine(
    payload: RAGQueryRequest, 
    x_session_id: Optional[str] = Header(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time typewriter AI response.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    # Server-side event logging
    if x_session_id:
        topic = analytics.classify_thyroid_topic(payload.query)
        analytics.log_event(db, current_user.id, x_session_id, "rag_query", "RAG Chat", json.dumps({"query": payload.query, "topic": topic}))
        
    retrieved = rag_engine.hybrid_search(payload.query, top_k=payload.top_k)
    synthesis = rag_engine.synthesize_answer(payload.query, retrieved)
    
    formatted_chunks = [
        DocumentChunk(
            chunk_id=c.get("chunk_id", "chk_unknown"),
            document_name=c.get("document_name", "ATA_Thyroid_Guidelines_2023.pdf"),
            page_number=c.get("page_number", 1),
            text_content=c.get("text_content", ""),
            similarity_score=c.get("similarity_score", 0.85),
            section=c.get("section", ""),
            source_type=c.get("source_type", "curated")
        ).dict()
        for c in retrieved
    ]

    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'content': 'Executing BM25 + ChromaDB Hybrid Vector Search...'})}\n\n"
        await asyncio.sleep(0.15)
        
        yield f"data: {json.dumps({'type': 'chunks', 'content': formatted_chunks})}\n\n"
        await asyncio.sleep(0.15)

        yield f"data: {json.dumps({'type': 'status', 'content': 'Applying Cross-Encoder Re-ranking & Synthesis...'})}\n\n"
        await asyncio.sleep(0.15)

        tokens = synthesis["answer"].split(" ")
        for token in tokens:
            yield f"data: {json.dumps({'type': 'token', 'content': token + ' '})}\n\n"
            await asyncio.sleep(0.015)

        yield f"data: {json.dumps({'type': 'done', 'confidence': synthesis['confidence_score']})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/v1/interpret-labs", response_model=LabInterpretationResponse)
async def interpret_labs(
    payload: LabInterpreterRequest, 
    x_session_id: Optional[str] = Header(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """
    Context-aware Lab Test Interpreter endpoint evaluating patient thyroid status
    (functioning, post-thyroidectomy/ablated, congenital hypothyroidism, or unknown).
    """
    # Server-side event logging
    if x_session_id:
        analytics.log_event(db, current_user.id, x_session_id, "lab_interpretation", "Lab Interpreter", json.dumps({"status": payload.thyroid_status}))

    status = (payload.thyroid_status or "functioning").lower().strip()

    # --- PATHWAY A: POST-THYROIDECTOMY / ABLATED THYROID ---
    if status in ["removed_ablated", "no_removed", "ablated", "post_thyroidectomy"]:
        reason = (payload.thyroid_removal_reason or "unknown").lower()
        
        # Retrieve RAG guidance for post-thyroidectomy
        rag_query = f"post thyroidectomy hormone replacement TSH {payload.tsh} free T4 {payload.free_t4} {reason}"
        rag_results = rag_engine.hybrid_search(rag_query, top_k=1)
        rag_chunk_text = rag_results[0]["text_content"] if rag_results else None

        if reason == "cancer":
            pattern = "Differentiated Thyroid Cancer TSH Suppression Monitoring"
            if payload.tsh > 2.0:
                summary = (
                    f"In a post-thyroidectomy cancer patient, serum TSH ({payload.tsh} mIU/L) is above suppression target. "
                    "Oncology guidelines recommend maintaining TSH < 0.1 mIU/L for high-risk or 0.5–2.0 mIU/L for low-risk disease. "
                    "Thyroglobulin (Tg) and anti-Tg antibody titers should be reviewed."
                )
                risk = "Moderate to High - Oncology Review Needed"
            else:
                summary = (
                    f"Serum TSH ({payload.tsh} mIU/L) meets post-operative suppression target boundaries. "
                    "Serum Tg and anti-Tg antibody monitoring recommended."
                )
                risk = "Target Range - Routine Cancer Follow-Up"
            citation = "ATA Differentiated Thyroid Cancer Guidelines 2024 Section 11.3 (p. 95)"

        else:
            pattern = "Post-Thyroidectomy Hormone Replacement Monitoring"
            if payload.tsh > 4.5:
                summary = (
                    f"TSH ({payload.tsh} mIU/L) is above target while Free T4 ({payload.free_t4} ng/dL) is low in an athyreotic patient. "
                    "This pattern indicates inadequate exogenous levothyroxine substitution or reduced bioavailability. "
                    "Ensure levothyroxine is taken on an empty stomach and spaced at least 4 hours apart from calcium or iron supplements."
                )
                risk = "Moderate - Exogenous Dose Review Indicated"
            elif payload.tsh < 0.45:
                summary = (
                    f"TSH ({payload.tsh} mIU/L) is suppressed in an athyreotic patient. "
                    "May indicate levothyroxine over-substitution. Clinical evaluation is recommended."
                )
                risk = "Moderate - Exogenous Over-Substitution Risk"
            else:
                summary = (
                    f"Serum TSH ({payload.tsh} mIU/L) and Free T4 ({payload.free_t4} ng/dL) fall within physiological replacement target limits for athyreotic monitoring."
                )
                risk = "Normal Target Range"
            citation = "ATA Surgical Guidelines 2024 Section 9.1 (p. 64)"

        return LabInterpretationResponse(
            pattern=pattern,
            summary=summary,
            guideline_citation=citation,
            risk_level=risk,
            rag_guidance_chunk=rag_chunk_text
        )

    # --- PATHWAY B: CONGENITAL HYPOTHYROIDISM / BORN WITHOUT FUNCTIONING THYROID ---
    elif status in ["congenital", "no_congenital", "born_without"]:
        is_pediatric = False
        age_desc = "unspecified age"
        
        if payload.patient_age is not None:
            unit = (payload.patient_age_unit or "years").lower()
            age_val = int(payload.patient_age) if payload.patient_age == int(payload.patient_age) else payload.patient_age
            age_desc = f"{age_val} {unit}"
            if unit in ["days", "weeks", "months"] or (unit == "years" and payload.patient_age < 18):
                is_pediatric = True

        rag_query = f"congenital hypothyroidism pediatric pediatric reference range age {age_desc} TSH {payload.tsh}"
        rag_results = rag_engine.hybrid_search(rag_query, top_k=1)
        rag_chunk_text = rag_results[0]["text_content"] if rag_results else None

        pattern = "Congenital Hypothyroidism Follow-Up Assessment"

        if is_pediatric:
            summary = (
                f"Patient is a pediatric case ({age_desc}) with congenital hypothyroidism ({payload.congenital_condition or 'known subtype'}). "
                "CRITICAL SAFETY NOTICE: Standard adult reference ranges (0.45 – 4.5 mIU/L) MUST NOT be applied to pediatric patients. "
                f"Current TSH is {payload.tsh} mIU/L and Free T4 is {payload.free_t4} ng/dL. "
                "Age-specific pediatric reference standards and frequent follow-up intervals apply. Consult pediatric endocrinology guidance."
            )
            risk = "Pediatric Endocrine Follow-Up Required"
            citation = "ATA/LWPES Pediatric Guidelines Section 2.1 (p. 12)"
        else:
            summary = (
                f"Adult patient ({age_desc}) with history of congenital hypothyroidism. "
                f"Serum TSH ({payload.tsh} mIU/L) and Free T4 ({payload.free_t4} ng/dL) evaluated in long-term substitution maintenance context."
            )
            risk = "Routine Maintenance Monitoring"
            citation = "ATA/LWPES Pediatric Guidelines Section 3.4 (p. 24)"

        return LabInterpretationResponse(
            pattern=pattern,
            summary=summary,
            guideline_citation=citation,
            risk_level=risk,
            rag_guidance_chunk=rag_chunk_text
        )

    # --- PATHWAY C: UNKNOWN THYROID STATUS ---
    elif status in ["unknown", "unspecified"]:
        rag_results = rag_engine.hybrid_search(f"TSH {payload.tsh} free T4 {payload.free_t4}", top_k=1)
        rag_chunk_text = rag_results[0]["text_content"] if rag_results else None

        pattern = "Uncertain Context (Unknown Thyroid Status)"
        summary = (
            f"Serum TSH ({payload.tsh} mIU/L) and Free T4 ({payload.free_t4} ng/dL) evaluated against standard reference boundaries. "
            "CAUTION: Patient thyroid status is unknown. Interpretation differs significantly if the thyroid gland was removed or if congenital hypothyroidism exists. "
            "Select specific Thyroid Status for contextual clinical assessment."
        )
        citation = "Mayo Clinic Endocrine Manual (p. 88)"
        risk = "Context Caution - Verify Patient Status"

        return LabInterpretationResponse(
            pattern=pattern,
            summary=summary,
            guideline_citation=citation,
            risk_level=risk,
            rag_guidance_chunk=rag_chunk_text
        )

    # --- PATHWAY D: STANDARD FUNCTIONING THYROID PATH (EXISTING BEHAVIOR) ---
    else:
        rag_results = rag_engine.hybrid_search(f"TSH {payload.tsh} free T4 {payload.free_t4} hypothyroidism hyperthyroidism", top_k=1)
        rag_chunk_text = rag_results[0]["text_content"] if rag_results else None

        if payload.tsh > 4.5 and payload.free_t4 < 0.82:
            pattern = "Primary Hypothyroidism Pattern"
            summary = f"Elevated TSH ({payload.tsh} mIU/L) combined with low Free T4 ({payload.free_t4} ng/dL) confirms primary thyroid failure."
            citation = "ATA Guidelines 2023 Section 4.2 (p. 14)"
            risk = "High - Levothyroxine Substitution Recommended"
        elif payload.tsh < 0.45 and payload.free_t4 > 1.77:
            pattern = "Overt Hyperthyroidism Pattern"
            summary = f"Suppressed TSH ({payload.tsh} mIU/L) with elevated Free T4 ({payload.free_t4} ng/dL) indicates autonomous hyperthyroidism (Graves' vs Toxic Nodule)."
            citation = "NIDDK Thyroid Guidelines Section 3.1 (p. 28)"
            risk = "High - Antithyroid Workup & TSI Panel Required"
        elif payload.tsh > 4.5 and payload.free_t4 >= 0.82:
            pattern = "Subclinical Hypothyroidism Variant"
            summary = f"Elevated TSH ({payload.tsh} mIU/L) with preserved Free T4 ({payload.free_t4} ng/dL). Check anti-TPO antibodies."
            citation = "ATA Guidelines 2023 Section 5.1 (p. 19)"
            risk = "Moderate - Monitor TSH & Anti-TPO Titers"
        else:
            pattern = "Euthyroid Reference Range"
            summary = f"Serum TSH ({payload.tsh} mIU/L) and Free T4 ({payload.free_t4} ng/dL) fall within normal physiological reference boundaries."
            citation = "Mayo Clinic Endocrine Manual (p. 88)"
            risk = "Normal Range"

        return LabInterpretationResponse(
            pattern=pattern,
            summary=summary,
            guideline_citation=citation,
            risk_level=risk,
            rag_guidance_chunk=rag_chunk_text
        )

@app.get("/api/v1/search-docs")
async def search_pdf_documents(
    q: str = Query(..., min_length=2), 
    x_session_id: Optional[str] = Header(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """
    Vector & BM25 search inside indexed medical documents returning section-aware hits.
    Searches across built-in guidelines and user-uploaded PDFs.
    """
    # Server-side event logging
    if x_session_id:
        analytics.log_event(db, current_user.id, x_session_id, "pdf_search", "PDF Search", json.dumps({"query": q}))

    results = rag_engine.hybrid_search(q, top_k=6)
    return {
        "query": q,
        "results_count": len(results),
        "results": [
            {
                "chunk_id": r.get("chunk_id"),
                "document_id": r.get("document_id"),
                "document_name": r.get("document_name"),
                "section": r.get("section"),
                "page_number": r.get("page_number"),
                "text_content": r.get("text_content"),
                "similarity_score": r.get("similarity_score"),
                "source_type": r.get("source_type", "curated")
            }
            for r in results
        ]
    }

# --- PDF Upload & Indexing Endpoints ---

@app.post("/api/v1/upload-pdf")
async def upload_pdf_document(file: UploadFile = File(...), current_user: models.User = Depends(auth.get_current_user)):
    """
    Uploads, validates, extracts, embeds, and indexes a PDF document into ChromaDB & BM25.
    Returns status, SHA-256 hash, page count, and chunk count.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")
        
    try:
        contents = await file.read()
        result = rag_engine.index_pdf_document(contents, file.filename)
        
        if result.get("status") == "duplicate":
            return JSONResponse(
                status_code=409,
                content={
                    "status": "duplicate",
                    "message": "This document has already been indexed.",
                    "document": result.get("document")
                }
            )
            
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"[FastAPI Error] Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF indexing failure: {str(e)}")

@app.get("/api/v1/imported-documents")
async def get_imported_documents(current_user: models.User = Depends(auth.get_current_user)):
    """
    Returns list of all user-imported documents and their indexing metadata.
    """
    docs = rag_engine.get_imported_documents()
    return {
        "count": len(docs),
        "documents": docs
    }

@app.delete("/api/v1/imported-documents/{doc_id}")
async def delete_imported_document(doc_id: str = Path(..., description="Internal document ID"), current_user: models.User = Depends(auth.get_current_user)):
    """
    Deletes a user-uploaded PDF from ChromaDB vector store, BM25 index, and storage.
    """
    try:
        result = rag_engine.delete_pdf_document(doc_id)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/api/v1/pdf-content/{doc_id}/page/{page_num}")
async def get_pdf_page_content(doc_id: str, page_num: int, current_user: models.User = Depends(auth.get_current_user)):
    """
    Returns text snippet and metadata for a specific PDF page for modal viewing.
    """
    content = rag_engine.get_page_content(doc_id, page_num)
    return content
