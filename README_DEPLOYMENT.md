# Medflow Production Deployment Guide 🚀
*Evidence-Grounded Medical RAG Assistant for Thyroid Endocrinology*

This guide explains how to deploy the **Medflow** platform as a unified, single-service production application accessible via a public HTTPS URL.

---

## 🏗️ Production Architecture Overview

In production, Medflow operates as a **single unified service**:
- **FastAPI** serves both the **REST API** (`/api/v1/*`, `/health`, `/docs`) and the **Single-Page Application** (`/`, `/index.html`).
- Zero cross-origin (CORS) complexity: All browser requests communicate via same-origin relative URLs (`window.location.origin`).
- **Medflow20** core RAG engine runs locally inside the container with pre-cached `BAAI/bge-small-en-v1.5` embeddings and persistent ChromaDB vectors.

```
                      [ Public HTTPS Internet ]
                                 │
                                 ▼
                     https://your-medflow-app.com
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                 FastAPI (Port 8000)             │
        │   • "/" & "/index.html" ──► Serves Web UI       │
        │   • "/api/v1/query"     ──► Medflow20 RAG       │
        │   • "/api/v1/auth/*"    ──► JWT & SQLite Auth   │
        │   • "/health"           ──► System Health Probe │
        └────────────────────────┬────────────────────────┘
                                 │
                 [ Persistent Volume: DATA_DIR ]
                 ├── medflow_auth.db (User & Analytics DB)
                 ├── chroma_db/ (1,904+ Vector Embeddings)
                 └── uploaded_pdfs/ (Imported Medical PDFs)
```

---

## ⚙️ Recommended Resource Allocation

| Resource | Minimum Practical Requirement | Recommended Production Allocation |
|---|---|---|
| **RAM** | `1.5 GB` | `2.0 GB - 4.0 GB` |
| **CPU** | `1 vCPU` | `2 vCPU` |
| **Disk / Storage** | `3.0 GB` | `10.0 GB Persistent Volume` |
| **GPU** | Not required (BGE-small runs on CPU) | Optional (speeds up high-concurrency embeddings) |

---

## 🌐 Option 1: One-Click Cloud Deployment (Render / Railway / Fly.io)

### Deploying to Render.com (Recommended)
1. Fork or push this repository to GitHub/GitLab.
2. In [Render Dashboard](https://dashboard.render.com), click **New +** $\rightarrow$ **Web Service**.
3. Connect your repository.
4. Select **Docker** as the Environment.
5. In **Environment Variables**, configure:
   - `APP_ENV`: `production`
   - `SECRET_KEY`: `(generate a 32+ character random string)`
   - `SECURE_COOKIE`: `true`
   - `DATA_DIR`: `/var/data`
6. In **Disks**, add a Persistent Disk mounted at `/var/data` (Size: 5-10 GB).
7. Click **Create Web Service**. Render will automatically build the Dockerfile, pre-cache the BGE model, and generate a public HTTPS URL (e.g. `https://medflow-medical.onrender.com`).

### Deploying to Railway.app
1. In [Railway.app](https://railway.app), click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
2. Railway detects the `Dockerfile` automatically.
3. Add a Volume mounted at `/app/data`.
4. Under **Variables**, add:
   - `APP_ENV=production`
   - `SECRET_KEY=your_secure_secret_key`
   - `SECURE_COOKIE=true`
   - `DATA_DIR=/app/data`
5. Click **Deploy**. Railway will provide an instant public HTTPS domain.

---

## 🐳 Option 2: Docker / Docker Compose Deployment (Self-Hosted VPS)

### Using Docker Compose
```bash
# 1. Clone the repository on your server
git clone <your-repo-url>
cd project-medflow

# 2. Build and start the container in detached mode
docker-compose up --build -d

# 3. View live startup logs
docker-compose logs -f
```

### Using Plain Docker CLI
```bash
# 1. Build the production image
docker build -t medflow:latest .

# 2. Run container with persistent volume mount
docker run -d \
  --name medflow-app \
  -p 8000:8000 \
  -e APP_ENV=production \
  -e PORT=8000 \
  -e DATA_DIR=/app/data \
  -e SECRET_KEY="generate_random_secret_here" \
  -v medflow_storage:/app/data \
  --restart unless-stopped \
  medflow:latest
```

---

## 🔒 Production Security Checklist

- [x] **No Localhost Dependency**: Frontend dynamically queries `window.location.origin`.
- [x] **Secure Cookies**: JWT cookies marked with `HttpOnly`, `SameSite=Lax`, and `Secure` when on HTTPS.
- [x] **Passwords Hashed**: Native `bcrypt` hashing (12 rounds) with zero plaintext storage.
- [x] **Isolated Runtime Storage**: SQLite, ChromaDB, and uploaded PDFs stored in configurable `DATA_DIR`.
- [x] **Input Safety Guardrails**: Pre-query emergency risk screening and section-aware medical grounding.

---

## 🩺 System Verification & Health Probes

Once deployed, verify your public deployment:
- **Web Interface**: `https://your-domain.com/`
- **Interactive API Docs**: `https://your-domain.com/docs`
- **System Health Status**: `https://your-domain.com/health`
- **RAG Query Test**:
```bash
curl -X POST "https://your-domain.com/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the initial levothyroxine dosage for subclinical hypothyroidism?", "top_k": 3}'
```
