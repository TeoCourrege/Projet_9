"""Tests de la collecte et du prétraitement des données Open Agenda."""

from datetime import date, timedelta

import pandas as pd
import pytest

from puls_rag import fetch
from puls_rag.preprocess import chunk_events, clean_text, event_header
from tests.conftest import TODAY


# ----- Collecte -----
def test_where_clause_filters_city_and_period():
    where = fetch.build_where_clause("Toulouse", date(2025, 10, 25))
    assert 'location_city="Toulouse"' in where
    assert "lastdate_end >= date'2025-10-25'" in where


def test_where_clause_escapes_quotes():
    assert fetch.build_where_clause('Tou"louse', date(2025, 1, 1)).startswith('location_city="Toulouse" AND')


def test_fetch_events_calls_export_endpoint(monkeypatch, settings, raw_events):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return raw_events

    def fake_get(url, params, timeout):
        calls.update(url=url, params=params)
        return FakeResponse()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    events = fetch.fetch_events(settings, today=TODAY)

    assert len(events) == len(raw_events) > 0
    assert calls["url"].endswith("/exports/json")
    assert (TODAY - timedelta(days=365)).isoformat() in calls["params"]["where"]
    assert "title_fr" in calls["params"]["select"]


def test_fetch_events_rejects_unexpected_payload(monkeypatch, settings):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "quota"}

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **kw: FakeResponse())
    with pytest.raises(ValueError):
        fetch.fetch_events(settings, today=TODAY)


def test_raw_save_load_roundtrip(tmp_path, raw_events):
    path = fetch.save_raw(raw_events, tmp_path / "raw" / "events.json")
    assert fetch.load_raw(path) == raw_events


# ----- Nettoyage -----
def test_clean_text_removes_html():
    clean = clean_text("<p>Concert <b>jazz</b>&nbsp;à Paris</p><p>Entrée libre</p>")
    assert "<" not in clean and ">" not in clean
    assert "Concert jazz à Paris" in clean
    assert "Entrée libre" in clean


@pytest.mark.parametrize("value", [None, "", float("nan")])
def test_clean_text_handles_missing(value):
    assert clean_text(value) == ""


def test_clean_events_keeps_only_valid_events(clean_df):
    # 1003 trop ancien, 1004 à Lyon, 1005 annulé, 1006 agenda emploi, 1010 sans titre, 1001 en double.
    assert sorted(clean_df["uid"]) == ["1001", "1002", "1008", "1009"]


def test_no_duplicates(clean_df):
    assert clean_df["uid"].is_unique


def test_duplicate_keeps_most_recent_version(clean_df):
    jazz = clean_df.set_index("uid").loc["1001"]
    assert "Django Reinhardt" in jazz["description"]


def test_events_are_recent(clean_df):
    cutoff = pd.Timestamp(TODAY - timedelta(days=365), tz="UTC")
    assert (clean_df["last_end"] >= cutoff).all()


def test_events_are_in_target_city(clean_df):
    assert (clean_df["city"] == "Toulouse").all()


def test_missing_description_falls_back_to_summary(clean_df):
    theatre = clean_df.set_index("uid").loc["1008"]
    assert theatre["description"] == "Une comédie de boulevard hilarante"


def test_status_label_is_parsed(clean_df):
    assert set(clean_df["status"]) <= {"Programmé", ""}


# ----- Chunking -----
def test_every_event_has_chunks(clean_df, documents):
    assert {d.metadata["uid"] for d in documents} == set(clean_df["uid"])


def test_long_description_is_split(documents):
    festival_chunks = [d for d in documents if d.metadata["uid"] == "1009"]
    assert len(festival_chunks) > 1
    assert all(d.metadata["n_chunks"] == len(festival_chunks) for d in festival_chunks)


def test_chunk_respects_max_size(clean_df):
    chunk_size = 300
    for doc in chunk_events(clean_df, chunk_size=chunk_size, chunk_overlap=50):
        description = doc.page_content.split("Description : ", 1)[1]
        assert len(description) <= chunk_size


def test_each_chunk_carries_header_and_metadata(documents):
    for doc in documents:
        assert doc.page_content.startswith("Titre : ")
        assert "Dates : " in doc.page_content
        for key in ("uid", "title", "dates", "location_name", "url", "first_begin"):
            assert key in doc.metadata


def test_header_contains_key_information(clean_df):
    header = event_header(clean_df.set_index("uid").loc["1002"].to_dict() | {"uid": "1002"})
    assert "du 01/11/2026 au 31/01/2027" in header
    assert "Le Château d'Eau" in header
    assert "Gratuit" in header
    assert "à partir de 6 ans" in header
