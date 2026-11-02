"""Fixtures partagées. Les tests unitaires n'appellent ni Internet, ni Mistral, ni le vrai modèle
d'embedding : on utilise un embedding « sac de mots » déterministe et un LLM factice."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from puls_rag.config import Settings  # noqa: E402
from puls_rag.indexing import build_vectorstore  # noqa: E402
from puls_rag.preprocess import chunk_events, clean_events  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 10, 25)


class KeywordEmbeddings(Embeddings):
    """Embedding déterministe : chaque mot (sans accent, minuscule) est haché dans une dimension.
    Deux textes partageant des mots ont donc une similarité cosinus élevée."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        vector = np.zeros(self.dim, dtype="float32")
        for word in re.findall(r"\w{3,}", text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def raw_events() -> list[dict]:
    return json.loads((FIXTURES / "sample_events.json").read_text(encoding="utf-8"))


@pytest.fixture
def clean_df(raw_events):
    return clean_events(raw_events, city="Toulouse", history_days=365, today=TODAY)


@pytest.fixture
def documents(clean_df):
    return chunk_events(clean_df, chunk_size=300, chunk_overlap=50)


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def vectorstore(documents, embeddings):
    return build_vectorstore(documents, embeddings, index_type="flat")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        raw_path=tmp_path / "raw.json",
        processed_path=tmp_path / "clean.json",
        index_dir=tmp_path / "faiss_index",
        mistral_api_key="",
        admin_token="secret-test",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
    )


@pytest.fixture
def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["Je vous recommande le Concert jazz au Rex le 14/11/2026."] * 50)
