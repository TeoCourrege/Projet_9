"""Tests de la vectorisation et de l'index FAISS."""

import time
from functools import partial

import pytest

from puls_rag import indexing, pipeline, preprocess
from tests.conftest import TODAY


def test_all_chunks_are_indexed(vectorstore, documents):
    assert vectorstore.index.ntotal == len(documents)
    assert len(vectorstore.docstore._dict) == len(documents)


def test_metadata_is_stored_with_vectors(vectorstore):
    doc = vectorstore.similarity_search("jazz", k=1)[0]
    assert doc.metadata["uid"] == "1001"
    assert doc.metadata["location_name"] == "Le Rex de Toulouse"


@pytest.mark.parametrize(
    "query, expected_uid",
    [
        ("concert de jazz manouche", "1001"),
        ("exposition de photographie gratuite", "1002"),
        ("pièce de théâtre comique", "1008"),
        ("festival de films documentaires", "1009"),
    ],
)
def test_search_returns_relevant_event(vectorstore, query, expected_uid):
    assert vectorstore.similarity_search(query, k=1)[0].metadata["uid"] == expected_uid


def test_scores_are_cosine_similarities(vectorstore):
    results = vectorstore.similarity_search_with_score("concert jazz", k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivf"])
def test_index_types(documents, embeddings, index_type):
    vs = indexing.build_vectorstore(documents, embeddings, index_type=index_type)
    assert vs.index.ntotal == len(documents)
    assert vs.similarity_search("jazz manouche", k=1)[0].metadata["uid"] == "1001"


def test_unknown_index_type_is_rejected(documents, embeddings):
    with pytest.raises(ValueError):
        indexing.build_vectorstore(documents, embeddings, index_type="lsh")


def test_empty_documents_are_rejected(embeddings):
    with pytest.raises(ValueError):
        indexing.build_vectorstore([], embeddings)


def test_save_and_reload(tmp_path, vectorstore, embeddings):
    indexing.save_vectorstore(vectorstore, tmp_path / "idx", manifest={"city": "Toulouse"})
    reloaded = indexing.load_vectorstore(tmp_path / "idx", embeddings)
    assert reloaded.index.ntotal == vectorstore.index.ntotal
    assert reloaded.similarity_search("jazz", k=1)[0].metadata["uid"] == "1001"
    manifest = indexing.load_manifest(tmp_path / "idx")
    assert manifest["n_vectors"] == vectorstore.index.ntotal and manifest["city"] == "Toulouse"


def test_load_missing_index_raises(tmp_path, embeddings):
    with pytest.raises(FileNotFoundError):
        indexing.load_vectorstore(tmp_path / "nothing", embeddings)


def test_search_is_fast(vectorstore):
    start = time.perf_counter()
    for _ in range(50):
        vectorstore.similarity_search("concert jazz", k=5)
    assert (time.perf_counter() - start) / 50 < 0.05


def test_full_pipeline_from_raw_file(monkeypatch, settings, raw_events, embeddings):
    """Pipeline complet sans réseau : fichier brut -> nettoyage -> chunks -> index persisté."""
    from puls_rag import fetch

    fetch.save_raw(raw_events, settings.raw_path)
    monkeypatch.setattr(pipeline.preprocess, "clean_events", partial(preprocess.clean_events, today=TODAY))
    monkeypatch.setattr(fetch, "fetch_events", lambda *a, **kw: pytest.fail("ne doit pas appeler l'API"))

    vectorstore, report = pipeline.build_index(settings, embeddings)

    assert report.n_raw_events == len(raw_events)
    assert report.n_clean_events == 4
    assert report.n_vectors == report.n_chunks == vectorstore.index.ntotal
    assert (settings.index_dir / "index.faiss").exists()
    assert settings.processed_path.exists()
    assert indexing.load_manifest(settings.index_dir)["n_events"] == 4
