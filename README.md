---
title: Medflow Medical AI Assistant
emoji: ??
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Medflow Medical AI Assistant (Medflow20 Core RAG Engine)

Specialized Thyroid Medical AI Platform built for clinical decision support, ATA guideline-grounded retrieval, lab interpretation (functioning, post-thyroidectomy, congenital), and PDF document vector search.

## Features
- **Medflow20 Core RAG Engine**: Hybrid Dense (BAAI/bge-small-en-v1.5 + ChromaDB) + Sparse (BM25) with Reciprocal Rank Fusion (RRF).
- **Grounded Evidence & Citations**: Section-aware page-level citations and verified ATA guideline grounding.
- **Specialized Lab Interpreter**: Multi-pathway assessment for functioning thyroid, post-thyroidectomy/ablated, congenital hypothyroidism, and pediatric cases.
- **Dynamic PDF Search & Vector Indexing**: Upload custom medical PDFs to index into ChromaDB & BM25 in real-time.
- **Production Single-Container Architecture**: FastAPI backend serving both API endpoints (/api/v1/*) and static SPA frontend (index.html) on port 7860.

## API Endpoints
- GET / — Medflow Single-Page Application (Web Interface)
- GET /health — System Health & RAG Engine Readiness
- POST /api/v1/query — Live Hybrid RAG Query & Synthesis
- POST /api/v1/interpret-labs — Specialized Thyroid Lab Interpretation
- GET /api/v1/search-docs — Vector & BM25 Document Search
- POST /api/v1/upload-pdf — Dynamic Medical PDF Indexing
- GET /api/v1/imported-documents — List Imported Documents
