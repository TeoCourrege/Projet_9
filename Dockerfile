# Image de l'API RAG Puls-Events (FastAPI + FAISS + Mistral)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# PyTorch en version CPU (~200 Mo au lieu de plusieurs Go pour la version CUDA par défaut sous Linux).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# Dépendances en premier : cette couche reste en cache tant que requirements.txt ne change pas.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Modèle d'embedding téléchargé au build : le conteneur n'a plus besoin de HuggingFace au démarrage.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY puls_rag/ ./puls_rag/
COPY api/ ./api/
COPY scripts/ ./scripts/

# Utilisateur non-root. data/ et faiss_index/ sont montés en volumes (voir docker-compose.yml).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data faiss_index \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
