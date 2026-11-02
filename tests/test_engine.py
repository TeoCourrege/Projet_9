"""Tests du moteur RAG (recherche + génération) avec un LLM factice."""

from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from puls_rag.engine import (
    PROMPT,
    IndexNotReadyError,
    LLMUnavailableError,
    RAGEngine,
    format_context,
    french_date,
)


@pytest.fixture
def engine(settings, embeddings, vectorstore, fake_llm):
    return RAGEngine(settings, embeddings=embeddings, vectorstore=vectorstore, llm=fake_llm,
                     today=lambda: date(2026, 10, 25))


def test_ask_returns_answer_and_sources(engine):
    result = engine.ask("Un concert de jazz ?")
    assert result["answer"].startswith("Je vous recommande")
    assert result["sources"][0]["uid"] == "1001"
    assert result["sources"][0]["title"] == "Concert jazz au Rex"
    assert len(result["contexts"]) == len(result["sources"])
    assert result["latency_ms"]["total"] >= result["latency_ms"]["retrieval"]


def test_retrieve_keeps_one_chunk_per_event(engine):
    results = engine.retrieve("festival cinéma documentaire films réalisateurs", k=4)
    uids = [doc.metadata["uid"] for doc, _ in results]
    assert len(uids) == len(set(uids))
    assert uids[0] == "1009"


def test_retrieve_respects_k(engine):
    assert len(engine.retrieve("concert", k=2)) == 2


def test_default_k_comes_from_settings(engine):
    assert len(engine.retrieve("concert")) == engine.settings.top_k


def test_empty_question_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.ask("   ")


def test_engine_without_index_raises(settings, embeddings, fake_llm):
    engine = RAGEngine(settings, embeddings=embeddings, llm=fake_llm)
    assert not engine.is_ready
    with pytest.raises(IndexNotReadyError):
        engine.ask("jazz")


def test_missing_api_key_raises(settings, embeddings, vectorstore):
    engine = RAGEngine(settings, embeddings=embeddings, vectorstore=vectorstore)
    assert not engine.llm_configured
    with pytest.raises(LLMUnavailableError):
        engine.ask("jazz")


def test_retrieve_offset_paginates(engine):
    page1 = [d.metadata["uid"] for d, _ in engine.retrieve("concert", k=2)]
    page2 = [d.metadata["uid"] for d, _ in engine.retrieve("concert", k=2, offset=2)]
    full = [d.metadata["uid"] for d, _ in engine.retrieve("concert", k=4)]
    assert page1 + page2 == full


def test_min_score_filters_weak_matches(engine):
    from dataclasses import replace

    engine.settings = replace(engine.settings, min_score=0.99)
    assert engine.retrieve("concert jazz") == []


def test_no_result_skips_llm(settings, embeddings, vectorstore):
    """Sans événement pertinent : réponse fixe, le LLM n'est pas appelé (même sans clé)."""
    from dataclasses import replace

    engine = RAGEngine(replace(settings, min_score=0.99), embeddings=embeddings, vectorstore=vectorstore)
    result = engine.ask("Quel temps fera-t-il demain ?")
    assert result["found"] is False
    assert result["sources"] == []
    assert "aucun événement" in result["answer"]


def test_metadata_describes_system(engine):
    meta = engine.metadata()
    assert meta["city"] == "Toulouse"
    assert meta["retrieval"]["top_k"] == 3
    assert meta["n_vectors"] == engine.n_vectors


def test_prompt_receives_context_date_and_question(settings, embeddings, vectorstore):
    """Vérifie ce qui est réellement envoyé au LLM."""
    captured = {}

    class SpyLLM(FakeListChatModel):
        def _call(self, messages, *args, **kwargs):
            captured["messages"] = messages
            return super()._call(messages, *args, **kwargs)

    engine = RAGEngine(settings, embeddings=embeddings, vectorstore=vectorstore,
                       llm=SpyLLM(responses=["ok"]), today=lambda: date(2026, 10, 25))
    engine.ask("Une exposition de photographie ?")
    system, human = captured["messages"]
    assert "dimanche 25 octobre 2026" in system.content
    assert "UNIQUEMENT" in system.content
    assert "Exposition photographie au Château d'Eau" in system.content
    assert human.content == "Une exposition de photographie ?"


def test_format_context_numbers_events(vectorstore):
    docs = vectorstore.similarity_search("jazz", k=2)
    context = format_context(docs)
    assert "[Événement 1]" in context and "[Événement 2]" in context
    assert "Lien : https://openagenda.com/test/events/concert-jazz" in context


def test_prompt_variables():
    assert set(PROMPT.input_variables) == {"city", "today", "context", "question"}


def test_french_date():
    assert french_date(date(2026, 12, 12)) == "samedi 12 décembre 2026"


def test_rebuild_swaps_index(monkeypatch, engine, vectorstore):
    from puls_rag import engine as engine_module
    from puls_rag.pipeline import BuildReport

    report = BuildReport(10, 4, 6, 6, "flat", 0.1)
    monkeypatch.setattr(engine_module.pipeline, "build_index", lambda *a, **kw: (vectorstore, report))
    engine._vectorstore = None
    assert engine.rebuild() == report
    assert engine.is_ready
