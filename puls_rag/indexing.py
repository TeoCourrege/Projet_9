"""Vectorisation des chunks et construction / persistance de l'index FAISS."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from puls_rag.config import Settings

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
INDEX_TYPES = ("flat", "hnsw", "ivf")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Modèle d'embedding : local (HuggingFace, par défaut) ou API Mistral (`mistral-embed`)."""
    settings = settings or Settings()
    if settings.embedding_provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=settings.embedding_model, api_key=settings.mistral_api_key)

    from langchain_huggingface import HuggingFaceEmbeddings

    # Les modèles E5 attendent des préfixes différents pour les passages et les requêtes.
    is_e5 = "e5" in settings.embedding_model.lower()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32, **({"prompt": "passage: "} if is_e5 else {})},
        query_encode_kwargs={"normalize_embeddings": True, **({"prompt": "query: "} if is_e5 else {})},
    )


def create_faiss_index(vectors: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """Crée l'index FAISS en produit scalaire (= similarité cosinus sur vecteurs normalisés).

    - flat : recherche exacte, idéale jusqu'à ~100k vecteurs (cas du POC) ;
    - hnsw : graphe de proximité, recherche approximative très rapide sans entraînement ;
    - ivf  : partitionnement en clusters (k-means), adapté aux très gros volumes.
    """
    if index_type not in INDEX_TYPES:
        raise ValueError(f"index_type doit être parmi {INDEX_TYPES}, reçu : {index_type!r}")
    n, dim = vectors.shape
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
    else:
        # Règle usuelle : ~sqrt(n) clusters, avec au moins ~39 points par cluster pour l'entraînement.
        nlist = max(1, min(int(math.sqrt(n)), n // 39))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.nprobe = min(16, nlist)
    index.add(vectors)
    return index


def build_vectorstore(documents: list[Document], embeddings: Embeddings, index_type: str = "flat") -> FAISS:
    """Vectorise les chunks et les indexe dans FAISS avec leurs métadonnées (docstore)."""
    if not documents:
        raise ValueError("Aucun document à indexer")
    start = time.perf_counter()
    vectors = np.asarray(embeddings.embed_documents([d.page_content for d in documents]), dtype="float32")
    faiss.normalize_L2(vectors)
    index = create_faiss_index(vectors, index_type)

    ids = [str(i) for i in range(len(documents))]
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(dict(zip(ids, documents))),
        index_to_docstore_id=dict(enumerate(ids)),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    logger.info("%d vecteurs (dim=%d) indexés en %.1fs", index.ntotal, vectors.shape[1], time.perf_counter() - start)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, index_dir: Path, manifest: dict) -> Path:
    """Sauvegarde l'index FAISS, le docstore (métadonnées) et un manifeste de traçabilité."""
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    manifest = {
        **manifest,
        "n_vectors": vectorstore.index.ntotal,
        "dimension": vectorstore.index.d,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_dir


def load_vectorstore(index_dir: Path, embeddings: Embeddings) -> FAISS:
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"Index FAISS introuvable dans {index_dir}. Lancez scripts/build_index.py")
    # Désérialisation pickle autorisée : le fichier est produit par notre propre script de build.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def load_manifest(index_dir: Path) -> dict:
    path = index_dir / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
