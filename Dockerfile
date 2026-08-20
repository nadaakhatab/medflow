# ==============================================================================
# Medflow Medical RAG - Production Multi-Stage / Optimized Dockerfile
# Specialized Thyroid Medical AI Platform (Medflow20 Core Engine)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    APP_ENV=production \
    DATA_DIR=/app/data \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Install system utilities needed for medical PDF processing and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache BAAI/bge-small-en-v1.5 weights into the image
# This ensures zero download latency on container startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy application source code and curated guidelines
COPY backend/ /app/backend/
COPY medflow20/ /app/medflow20/
COPY index.html /app/index.html

# Create persistent data volume directory and ensure write permissions for non-root users
RUN mkdir -p /app/data /app/backend/uploaded_pdfs /app/medflow20/uploaded_pdfs && \
    chmod -R 777 /app/data /app/backend /app/medflow20

# Expose production port (Hugging Face Spaces default Docker port is 7860)
EXPOSE 7860

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://127.0.0.1:/health || exit 1

# Production entrypoint: start FastAPI backend serving both REST API and SPA frontend
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port "]
