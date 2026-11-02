"""Métriques d'évaluation du système RAG.

Trois niveaux, du moins coûteux au plus coûteux :
1. Recherche (sans LLM)     : hit rate@k et MRR sur les événements attendus du jeu annoté.
2. Réponse vs référence      : exact match, F1 lexical, similarité sémantique -> classe
                               correcte / partiellement correcte / incorrecte.
3. Ragas (LLM juge Mistral)  : faithfulness, answer relevancy, context precision, context recall.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

# Seuils de similarité cosinus pour classer les réponses. Les embeddings E5 donnent des scores
# élevés même pour des textes peu liés (~0.75) : seuils à ajuster en relisant quelques réponses.
CORRECT_THRESHOLD = 0.88
PARTIAL_THRESHOLD = 0.80

REQUIRED_FIELDS = {"id", "question", "ground_truth", "expected_uids", "category"}


def load_test_set(path: Path) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in items:
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"Question {item.get('id')} : champs manquants {missing}")
    return items


# ----- 1. Recherche -----
def retrieval_metrics(expected_uids: list[str], retrieved_uids: list[str]) -> dict:
    """hit = au moins un événement attendu retrouvé ; reciprocal_rank = 1 / rang du premier."""
    if not expected_uids:
        return {"hit": None, "reciprocal_rank": None}
    for rank, uid in enumerate(retrieved_uids, start=1):
        if uid in expected_uids:
            return {"hit": 1.0, "reciprocal_rank": 1.0 / rank}
    return {"hit": 0.0, "reciprocal_rank": 0.0}


# ----- 2. Réponse vs référence -----
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """F1 sur les mots (métrique SQuAD) : recouvrement lexical entre réponse et référence."""
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(prediction: str, reference: str, embeddings: Embeddings) -> float:
    vectors = np.asarray(embeddings.embed_documents([prediction, reference]), dtype="float32")
    a, b = vectors
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def classify(similarity: float) -> str:
    if similarity >= CORRECT_THRESHOLD:
        return "correcte"
    if similarity >= PARTIAL_THRESHOLD:
        return "partiellement correcte"
    return "incorrecte"


def answer_metrics(prediction: str, reference: str, embeddings: Embeddings) -> dict:
    similarity = semantic_similarity(prediction, reference, embeddings)
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": round(token_f1(prediction, reference), 4),
        "semantic_similarity": round(similarity, 4),
        "verdict": classify(similarity),
    }


def summarize(rows: list[dict], keys: list[str]) -> dict:
    """Moyenne de chaque métrique en ignorant les valeurs absentes (None / NaN)."""
    summary = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
        summary[key] = round(float(np.mean(values)), 4) if values else None
    return summary


FAITHFULNESS_ALERT = 0.7


def error_report(rows: list[dict]) -> str:
    """Rapport Markdown des erreurs : recherches manquées, réponses incorrectes, hallucinations
    probables (faithfulness faible), et taux d'erreur par catégorie."""
    missed = [r for r in rows if r.get("hit") == 0.0]
    wrong = [r for r in rows if r.get("verdict") == "incorrecte"]
    unfaithful = [r for r in rows if (r.get("faithfulness") is not None and not np.isnan(r["faithfulness"])
                                      and r["faithfulness"] < FAITHFULNESS_ALERT)]

    lines = ["# Analyse des erreurs", "", f"{len(rows)} questions évaluées.", ""]
    lines += ["| Catégorie | Questions | Recherche manquée | Réponse incorrecte |", "|---|---|---|---|"]
    for cat in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == cat]
        lines.append(f"| {cat} | {len(group)} | {sum(r in missed for r in group)} | {sum(r in wrong for r in group)} |")

    def section(title: str, items: list[dict], detail) -> None:
        lines.extend(["", f"## {title} ({len(items)})", ""])
        lines.extend(f"- **{r['id']}** ({r['category']}) : {r['question']}\n  - {detail(r)}" for r in items)
        if not items:
            lines.append("Aucune.")

    section("Recherches manquées (événement attendu absent du top-k)", missed,
            lambda r: f"retrouvés : {r.get('retrieved_titles') or '—'}")
    section("Réponses classées incorrectes", wrong,
            lambda r: f"similarité {r.get('semantic_similarity')} · réponse : {(r.get('answer') or '')[:200]}")
    section(f"Faithfulness < {FAITHFULNESS_ALERT} (hallucination probable)", unfaithful,
            lambda r: f"faithfulness {r['faithfulness']:.2f}")
    return "\n".join(lines) + "\n"


# ----- 3. Ragas -----
def ragas_evaluate(rows: list[dict], llm, embeddings: Embeddings) -> list[dict]:
    """Calcule les métriques Ragas avec Mistral comme LLM juge. Retourne une ligne par question."""
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"] or [""],
                "reference": r["ground_truth"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        # L'offre gratuite de Mistral limite le débit : peu d'appels en parallèle.
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=10, max_wait=60),
    )
    df = result.to_pandas()
    metric_cols = [c for c in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall") if c in df]
    return df[metric_cols].to_dict(orient="records")
