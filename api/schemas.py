"""Schémas Pydantic des requêtes et réponses de l'API (validation + documentation Swagger)."""

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=1000, examples=["Quels concerts de rap sont prévus en décembre ?"]
    )
    k: int = Field(5, ge=1, le=20, description="Nombre d'événements récupérés dans FAISS")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("La question ne peut pas être vide")
        return value.strip()


class SearchRequest(AskRequest):
    offset: int = Field(0, ge=0, le=100, description="Nombre d'événements à sauter (pagination)")


class Source(BaseModel):
    uid: str
    title: str
    dates: str
    location_name: str
    url: str
    score: float = Field(description="Similarité cosinus avec la question (1 = identique)")


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool = Field(description="False si aucun événement ne correspond : la réponse est alors un message fixe")
    sources: list[Source]
    latency_ms: dict[str, int]
    model: str


class SearchResponse(BaseModel):
    question: str
    k: int
    offset: int
    next_offset: int | None = Field(description="Offset de la page suivante (None si dernière page)")
    sources: list[Source]


class MetadataResponse(BaseModel):
    service: str
    version: str
    city: str
    history_days: int
    data_source: str
    embedding: dict
    llm: dict
    retrieval: dict
    index_loaded: bool
    n_vectors: int
    n_events: int | None
    index_built_at: str | None
    endpoints: dict[str, str]


class RebuildRequest(BaseModel):
    refetch: bool = Field(False, description="Re-télécharger les événements depuis Open Agenda avant l'indexation")


class RebuildResponse(BaseModel):
    status: str
    n_raw_events: int
    n_clean_events: int
    n_chunks: int
    n_vectors: int
    index_type: str
    duration_s: float


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    n_vectors: int
    llm_configured: bool
    index: dict
