"""Tests de l'API FastAPI (TestClient, moteur RAG factice injecté)."""

from datetime import date

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from api.main import app
from puls_rag.engine import RAGEngine
from puls_rag.pipeline import BuildReport


@pytest.fixture
def engine(settings, embeddings, vectorstore, fake_llm):
    return RAGEngine(settings, embeddings=embeddings, vectorstore=vectorstore, llm=fake_llm,
                     today=lambda: date(2026, 10, 25))


@pytest.fixture
def client(engine):
    app.state.engine = engine  # le lifespan ne recharge pas un moteur déjà présent
    with TestClient(app) as c:
        yield c
    app.state.engine = None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["n_vectors"] > 0
    assert body["llm_configured"] is True


def test_ask(client):
    response = client.post("/ask", json={"question": "Un concert de jazz ?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["sources"]) == 2
    assert body["sources"][0]["title"] == "Concert jazz au Rex"
    assert {"uid", "title", "dates", "location_name", "url", "score"} <= body["sources"][0].keys()


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "    "}, {}, {"question": "jazz", "k": 0}, {"question": "jazz", "k": 100},
     {"question": "x" * 1001}],
)
def test_ask_invalid_payload(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_requires_json_body(client):
    assert client.post("/ask", content="pas du json").status_code == 422


def test_ask_without_index_returns_503(client, engine):
    engine._vectorstore = None
    response = client.post("/ask", json={"question": "jazz"})
    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"]


def test_ask_without_api_key_returns_503(client, engine):
    engine._llm = None
    response = client.post("/ask", json={"question": "jazz"})
    assert response.status_code == 503
    assert "MISTRAL_API_KEY" in response.json()["detail"]


def test_ask_llm_failure_returns_502(client, engine):
    class BrokenLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise ConnectionError("Mistral injoignable")

    engine._llm = BrokenLLM(responses=["unused"])
    assert client.post("/ask", json={"question": "jazz"}).status_code == 502


def test_search_works_without_llm(client, engine):
    engine._llm = None
    response = client.post("/search", json={"question": "exposition photographie", "k": 1})
    assert response.status_code == 200
    assert response.json()["sources"][0]["uid"] == "1002"


def test_rebuild_requires_token(client):
    assert client.post("/rebuild").status_code == 401
    assert client.post("/rebuild", headers={"X-Admin-Token": "mauvais"}).status_code == 401


def test_rebuild_disabled_without_admin_token(client, engine, monkeypatch):
    from dataclasses import replace

    engine.settings = replace(engine.settings, admin_token="")
    assert client.post("/rebuild", headers={"X-Admin-Token": "x"}).status_code == 403


def test_rebuild_with_token(client, engine, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "rebuild", lambda refetch=False: calls.append(refetch) or BuildReport(10, 4, 6, 6, "flat", 0.5))
    response = client.post("/rebuild", headers={"X-Admin-Token": "secret-test"}, json={"refetch": True})
    assert response.status_code == 200
    assert response.json()["n_vectors"] == 6
    assert calls == [True]


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Toulouse"
    assert body["index_loaded"] is True
    assert body["n_vectors"] > 0
    assert body["llm"]["configured"] is True
    assert {"provider", "model"} <= body["embedding"].keys()
    assert "POST /ask" in body["endpoints"]


def test_metadata_without_index(client, engine):
    engine._vectorstore = None
    body = client.get("/metadata").json()
    assert body["index_loaded"] is False
    assert body["n_vectors"] == 0


def test_ask_returns_found_flag(client):
    assert client.post("/ask", json={"question": "Un concert de jazz ?"}).json()["found"] is True


def test_ask_without_matching_event(client, engine):
    from dataclasses import replace

    engine.settings = replace(engine.settings, min_score=0.99)
    body = client.post("/ask", json={"question": "météo demain"}).json()
    assert body["found"] is False
    assert body["sources"] == []
    assert "aucun événement" in body["answer"]


def test_search_pagination(client):
    first = client.post("/search", json={"question": "concert", "k": 2}).json()
    second = client.post("/search", json={"question": "concert", "k": 2, "offset": 2}).json()
    assert first["next_offset"] == 2
    first_uids = {s["uid"] for s in first["sources"]}
    second_uids = {s["uid"] for s in second["sources"]}
    assert len(second_uids) == 2 and not first_uids & second_uids


def test_search_last_page_has_no_next_offset(client):
    body = client.post("/search", json={"question": "concert", "k": 5, "offset": 100}).json()
    assert body["sources"] == []
    assert body["next_offset"] is None


def test_search_invalid_offset(client):
    assert client.post("/search", json={"question": "concert", "offset": -1}).status_code == 422


def test_openapi_documents_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/ask", "/search", "/rebuild", "/health", "/metadata"} <= paths.keys()
    assert client.get("/docs").status_code == 200
