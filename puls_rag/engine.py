"""Moteur RAG : recherche FAISS + génération Mistral, orchestrées avec LangChain (LCEL)."""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from puls_rag import indexing, pipeline
from puls_rag.config import Settings
from puls_rag.preprocess import PARIS_TZ

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es l'assistant de Puls-Events, spécialisé dans la recommandation d'événements culturels à {city}.
Nous sommes le {today}.

Règles :
- Réponds UNIQUEMENT à partir des événements listés dans le contexte ci-dessous. N'invente jamais d'événement, de date, de lieu ni de tarif.
- Si aucun événement du contexte ne répond à la demande, dis-le clairement. Tu peux alors suggérer un événement proche du contexte en précisant qu'il ne correspond pas exactement.
- Tiens compte de la date du jour : précise si un événement est déjà passé.
- Réponds en français, de façon concise et structurée. Pour chaque événement recommandé, indique le titre, les dates, le lieu et le tarif s'il est connu.
- Ne parle pas du « contexte » ou des « documents » : adresse-toi naturellement à l'utilisateur.

Contexte (événements trouvés) :
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

NO_RESULT_ANSWER = (
    "Je n'ai trouvé aucun événement à {city} correspondant à votre demande. "
    "Essayez de reformuler la question ou d'élargir vos critères (dates, type d'événement, lieu)."
)

FRENCH_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


class IndexNotReadyError(RuntimeError):
    """L'index FAISS n'est pas chargé (jamais construit ou introuvable)."""


class LLMUnavailableError(RuntimeError):
    """Le LLM n'est pas configuré (clé MISTRAL_API_KEY absente)."""


def french_date(day: date) -> str:
    return f"{FRENCH_DAYS[day.weekday()]} {day.day} {FRENCH_MONTHS[day.month - 1]} {day.year}"


def format_context(documents: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        url = doc.metadata.get("url")
        blocks.append(f"[Événement {i}]\n{doc.page_content}" + (f"\nLien : {url}" if url else ""))
    return "\n\n".join(blocks)


def format_sources(retrieved: list[tuple[Document, float]]) -> list[dict]:
    """Métadonnées des événements sources renvoyées à l'utilisateur."""
    return [
        {**{key: doc.metadata.get(key, "") for key in ("uid", "title", "dates", "location_name", "url")},
         "score": round(score, 4)}
        for doc, score in retrieved
    ]


def build_llm(settings: Settings) -> BaseChatModel:
    if not settings.mistral_api_key:
        raise LLMUnavailableError("MISTRAL_API_KEY absente : ajoutez-la dans le fichier .env")
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=settings.llm_model,
        api_key=settings.mistral_api_key,
        temperature=settings.temperature,
        max_retries=3,
        timeout=60,
    )


class RAGEngine:
    """Point d'entrée unique du système RAG, partagé par l'API, les scripts et les tests.

    Les dépendances (embeddings, vectorstore, LLM) sont injectables pour les tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        vectorstore: FAISS | None = None,
        llm: BaseChatModel | None = None,
        today: Callable[[], date] | None = None,
    ):
        self.settings = settings or Settings()
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._llm = llm
        self._today = today or (lambda: datetime.now(PARIS_TZ).date())
        self._lock = threading.Lock()
        self.manifest: dict = {}

    # ----- Chargement -----
    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = indexing.get_embeddings(self.settings)
        return self._embeddings

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def llm_configured(self) -> bool:
        return self._llm is not None or bool(self.settings.mistral_api_key)

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None

    @property
    def n_vectors(self) -> int:
        return self._vectorstore.index.ntotal if self._vectorstore is not None else 0

    def load(self) -> "RAGEngine":
        """Charge l'index persisté sur disque (appelé une seule fois au démarrage de l'API)."""
        self._vectorstore = indexing.load_vectorstore(self.settings.index_dir, self.embeddings)
        self.manifest = indexing.load_manifest(self.settings.index_dir)
        logger.info("Index chargé : %d vecteurs", self.n_vectors)
        return self

    def rebuild(self, refetch: bool = False) -> pipeline.BuildReport:
        """Reconstruit l'index puis le remplace à chaud : les requêtes en cours utilisent l'ancien."""
        with self._lock:
            vectorstore, report = pipeline.build_index(self.settings, self.embeddings, refetch=refetch)
            self._vectorstore = vectorstore
            self.manifest = indexing.load_manifest(self.settings.index_dir)
        return report

    # ----- Recherche -----
    def retrieve(self, question: str, k: int | None = None, offset: int = 0) -> list[tuple[Document, float]]:
        """Top-k chunks les plus similaires, en ne gardant que le meilleur chunk par événement
        (évite qu'un événement à longue description occupe tout le contexte).

        `offset` saute les premiers événements (pagination de /search). Les événements sous
        `settings.min_score` sont écartés."""
        if self._vectorstore is None:
            raise IndexNotReadyError("Index FAISS non chargé")
        question = question.strip()
        if not question:
            raise ValueError("La question est vide")
        k = k or self.settings.top_k
        wanted = k + offset
        candidates = self._vectorstore.similarity_search_with_score(question, k=wanted * 4)
        results, seen = [], set()
        for doc, score in candidates:
            uid = doc.metadata.get("uid")
            if uid in seen or score < self.settings.min_score:
                continue
            seen.add(uid)
            results.append((doc, float(score)))
            if len(results) == wanted:
                break
        return results[offset:]

    def metadata(self) -> dict:
        """Description du système pour les services consommateurs : périmètre, modèles, index."""
        s = self.settings
        return {
            "service": "Puls-Events RAG",
            "city": s.city,
            "history_days": s.history_days,
            "data_source": "Open Agenda (evenements-publics-openagenda, OpenDataSoft)",
            "embedding": {"provider": s.embedding_provider, "model": s.embedding_model},
            "llm": {"model": s.llm_model, "temperature": s.temperature, "configured": self.llm_configured},
            "retrieval": {"top_k": s.top_k, "min_score": s.min_score, "index_type": s.index_type},
            "index_loaded": self.is_ready,
            "n_vectors": self.n_vectors,
            "n_events": self.manifest.get("n_events"),
            "index_built_at": self.manifest.get("built_at"),
        }

    # ----- Génération -----
    def ask(self, question: str, k: int | None = None) -> dict:
        start = time.perf_counter()
        retrieved = self.retrieve(question, k)
        t_retrieval = time.perf_counter() - start

        documents = [doc for doc, _ in retrieved]
        if documents:
            chain = PROMPT | self.llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "city": self.settings.city,
                    "today": french_date(self._today()),
                    "context": format_context(documents),
                    "question": question.strip(),
                }
            )
        else:
            # Aucun événement assez proche : réponse fixe, sans appel au LLM (pas de risque d'invention).
            answer = NO_RESULT_ANSWER.format(city=self.settings.city)
        total = time.perf_counter() - start
        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "found": bool(documents),
            "sources": format_sources(retrieved),
            "contexts": [doc.page_content for doc in documents],
            "latency_ms": {"retrieval": round(t_retrieval * 1000), "total": round(total * 1000)},
            "model": getattr(self.llm, "model", type(self.llm).__name__) if documents else self.settings.llm_model,
        }
