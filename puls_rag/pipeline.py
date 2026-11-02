"""Pipeline complet de (re)construction de la base vectorielle :
collecte Open Agenda -> nettoyage -> chunking -> vectorisation -> index FAISS."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from puls_rag import fetch, indexing, preprocess
from puls_rag.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class BuildReport:
    n_raw_events: int
    n_clean_events: int
    n_chunks: int
    n_vectors: int
    index_type: str
    duration_s: float


def build_index(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    refetch: bool = False,
) -> tuple[FAISS, BuildReport]:
    """Reconstruit l'index. Les données brutes sont re-téléchargées si `refetch` ou si absentes."""
    settings = settings or Settings()
    start = time.perf_counter()

    if refetch or not settings.raw_path.exists():
        raw = fetch.fetch_events(settings)
        fetch.save_raw(raw, settings.raw_path)
    else:
        raw = fetch.load_raw(settings.raw_path)

    df = preprocess.clean_events(raw, city=settings.city, history_days=settings.history_days)
    preprocess.save_clean(df, settings.processed_path)
    documents = preprocess.chunk_events(df, settings.chunk_size, settings.chunk_overlap)

    embeddings = embeddings or indexing.get_embeddings(settings)
    vectorstore = indexing.build_vectorstore(documents, embeddings, settings.index_type)

    # Contrôle d'intégrité : chaque chunk doit avoir son vecteur, chaque événement au moins un chunk.
    n_indexed_events = len({d.metadata["uid"] for d in documents})
    if vectorstore.index.ntotal != len(documents) or n_indexed_events != len(df):
        raise RuntimeError(
            f"Indexation incomplète : {vectorstore.index.ntotal} vecteurs pour {len(documents)} chunks, "
            f"{n_indexed_events} événements indexés sur {len(df)}"
        )

    report = BuildReport(
        n_raw_events=len(raw),
        n_clean_events=len(df),
        n_chunks=len(documents),
        n_vectors=vectorstore.index.ntotal,
        index_type=settings.index_type,
        duration_s=round(time.perf_counter() - start, 1),
    )
    indexing.save_vectorstore(
        vectorstore,
        settings.index_dir,
        manifest={
            "city": settings.city,
            "history_days": settings.history_days,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "index_type": settings.index_type,
            "n_events": len(df),
            "n_chunks": len(documents),
        },
    )
    logger.info("Index reconstruit : %s", report)
    return vectorstore, report
