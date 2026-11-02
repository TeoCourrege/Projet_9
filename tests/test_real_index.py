"""Tests d'intégration sur les vraies données et le vrai index (vrai modèle d'embedding).

Ignorés tant que `python scripts/build_index.py` n'a pas été lancé.
Lancer seuls : pytest -m integration
"""

import time

import pandas as pd
import pytest

from puls_rag.config import Settings
from puls_rag.engine import RAGEngine
from puls_rag.fetch import load_raw
from puls_rag.indexing import load_manifest
from puls_rag.preprocess import load_clean

SETTINGS = Settings()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (SETTINGS.index_dir / "index.faiss").exists(), reason="index FAISS non construit"),
]


@pytest.fixture(scope="module")
def engine():
    return RAGEngine(SETTINGS).load()


@pytest.fixture(scope="module")
def clean_df():
    return load_clean(SETTINGS.processed_path)


def test_raw_data_was_collected():
    assert len(load_raw(SETTINGS.raw_path)) > 1000


def test_clean_data_matches_filters(clean_df):
    manifest = load_manifest(SETTINGS.index_dir)
    built_at = pd.Timestamp(manifest["built_at"])
    assert len(clean_df) > 1000
    assert clean_df["uid"].is_unique
    assert (clean_df["city"] == SETTINGS.city).all()
    assert (clean_df["last_end"] >= built_at - pd.Timedelta(days=SETTINGS.history_days + 1)).all()
    assert (clean_df["title"].str.len() > 0).all()
    assert (clean_df["description"].str.len() > 0).all()


def test_every_event_and_chunk_is_indexed(engine, clean_df):
    manifest = load_manifest(SETTINGS.index_dir)
    assert engine.n_vectors == manifest["n_chunks"]
    assert manifest["n_events"] == len(clean_df)
    indexed_uids = {doc.metadata["uid"] for doc in engine._vectorstore.docstore._dict.values()}
    assert indexed_uids == set(clean_df["uid"])


def test_semantic_search_finds_relevant_events(engine):
    titles = " ".join(doc.metadata["title"].lower() + doc.page_content.lower()
                      for doc, _ in engine.retrieve("visite guidée d'un musée", k=5))
    assert "musée" in titles or "visite" in titles


def test_retrieval_latency(engine):
    engine.retrieve("concert", k=5)  # préchauffage
    start = time.perf_counter()
    for _ in range(10):
        engine.retrieve("Quels concerts de rap en décembre ?", k=5)
    assert (time.perf_counter() - start) / 10 < 0.5
