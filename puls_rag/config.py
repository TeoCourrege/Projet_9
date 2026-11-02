"""Configuration centralisée du projet, lue depuis les variables d'environnement (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Paramètres du pipeline RAG. Chaque champ peut être surchargé par une variable d'environnement."""

    # --- Données Open Agenda ---
    city: str = field(default_factory=lambda: _env("OA_CITY", "Toulouse"))
    history_days: int = field(default_factory=lambda: int(_env("OA_HISTORY_DAYS", "365")))
    oa_base_url: str = field(
        default_factory=lambda: _env(
            "OA_BASE_URL",
            "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda",
        )
    )

    # --- Chemins ---
    raw_path: Path = field(default_factory=lambda: Path(_env("RAW_PATH", str(ROOT_DIR / "data/raw/events_raw.json"))))
    processed_path: Path = field(
        default_factory=lambda: Path(_env("PROCESSED_PATH", str(ROOT_DIR / "data/processed/events_clean.json")))
    )
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT_DIR / "faiss_index"))))
    test_set_path: Path = field(
        default_factory=lambda: Path(_env("TEST_SET_PATH", str(ROOT_DIR / "data/test_set.json")))
    )

    # --- Découpage et vectorisation ---
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "huggingface"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    index_type: str = field(default_factory=lambda: _env("INDEX_TYPE", "flat"))

    # --- Génération ---
    mistral_api_key: str = field(default_factory=lambda: _env("MISTRAL_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "mistral-small-latest"))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    # Similarité cosinus minimale d'un événement pour être retenu (0 = pas de seuil).
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0")))

    # --- API ---
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
