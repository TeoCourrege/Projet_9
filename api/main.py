"""API REST du POC Puls-Events.

Lancement : uvicorn api.main:app --host 0.0.0.0 --port 8000
Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
    RebuildRequest,
    RebuildResponse,
    SearchRequest,
    SearchResponse,
)
from puls_rag import __version__
from puls_rag.engine import IndexNotReadyError, LLMUnavailableError, RAGEngine, format_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le moteur RAG une seule fois au démarrage (modèle d'embedding + index FAISS)."""
    if getattr(app.state, "engine", None) is None:
        engine = RAGEngine()
        try:
            engine.load()
        except FileNotFoundError as exc:
            # L'API démarre quand même : /health le signale et /rebuild permet de construire l'index.
            logger.warning("%s", exc)
        app.state.engine = engine
    yield


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "Assistant de recommandation d'événements culturels (POC). "
        "Recherche sémantique FAISS sur les événements Open Agenda + génération de réponse avec Mistral."
    ),
    version=__version__,
    lifespan=lifespan,
)


def get_engine(request: Request) -> RAGEngine:
    return request.app.state.engine


def require_admin(engine: RAGEngine = Depends(get_engine), x_admin_token: str | None = Header(None)) -> None:
    """Protège /rebuild : en-tête X-Admin-Token égal à la variable ADMIN_TOKEN."""
    expected = engine.settings.admin_token
    if not expected:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Reconstruction désactivée : définissez ADMIN_TOKEN")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Jeton d'administration invalide")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health(engine: RAGEngine = Depends(get_engine)):
    """État du service : index chargé, nombre de vecteurs, LLM configuré."""
    return HealthResponse(
        status="ok" if engine.is_ready else "index_missing",
        index_loaded=engine.is_ready,
        n_vectors=engine.n_vectors,
        llm_configured=engine.llm_configured,
        index=engine.manifest,
    )


ENDPOINTS = {
    "POST /ask": "Question en langage naturel -> réponse générée + événements sources",
    "POST /search": "Recherche sémantique seule (sans LLM), paginée",
    "GET /metadata": "Périmètre des données, modèles et paramètres du système",
    "GET /health": "État du service",
    "POST /rebuild": "Reconstruction de l'index (en-tête X-Admin-Token)",
}


@app.get("/metadata", response_model=MetadataResponse, tags=["Monitoring"])
def metadata(engine: RAGEngine = Depends(get_engine)):
    """Décrit le système : ville et période couvertes, modèles d'embedding et de génération,
    paramètres de recherche, taille et date de l'index, routes disponibles."""
    return MetadataResponse(**engine.metadata(), version=__version__, endpoints=ENDPOINTS)


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(payload: AskRequest, engine: RAGEngine = Depends(get_engine)):
    """Pose une question en langage naturel et reçoit une réponse générée à partir des événements indexés,
    accompagnée des événements sources utilisés."""
    try:
        return engine.ask(payload.question, k=payload.k)
    except IndexNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{exc}. Appelez POST /rebuild.") from exc
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:  # erreur du fournisseur LLM (quota, réseau, clé invalide...)
        logger.exception("Échec de la génération")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Erreur lors de l'appel au LLM : {type(exc).__name__}") from exc


@app.post("/search", response_model=SearchResponse, tags=["RAG"])
def search(payload: SearchRequest, engine: RAGEngine = Depends(get_engine)):
    """Recherche sémantique seule (sans LLM) : renvoie les événements les plus proches de la question.
    Fonctionne sans clé Mistral. Paginée par `k` (taille de page) et `offset`."""
    try:
        retrieved = engine.retrieve(payload.question, payload.k, offset=payload.offset)
    except IndexNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{exc}. Appelez POST /rebuild.") from exc
    return {
        "question": payload.question,
        "k": payload.k,
        "offset": payload.offset,
        "next_offset": payload.offset + payload.k if len(retrieved) == payload.k else None,
        "sources": format_sources(retrieved),
    }


@app.post("/rebuild", response_model=RebuildResponse, tags=["Administration"], dependencies=[Depends(require_admin)])
async def rebuild(payload: RebuildRequest | None = None, engine: RAGEngine = Depends(get_engine)):
    """Reconstruit la base vectorielle FAISS (option `refetch` : re-télécharge Open Agenda).
    Protégé par l'en-tête `X-Admin-Token`. Opération longue (plusieurs minutes sur CPU)."""
    payload = payload or RebuildRequest()
    try:
        report = await run_in_threadpool(engine.rebuild, payload.refetch)
    except Exception as exc:
        logger.exception("Échec de la reconstruction")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Échec de la reconstruction : {exc}") from exc
    return RebuildResponse(status="Index reconstruit", **asdict(report))
