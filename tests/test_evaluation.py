"""Tests des métriques d'évaluation et du jeu de test annoté."""

import pytest

from puls_rag import evaluation
from puls_rag.config import Settings


def test_test_set_is_valid():
    items = evaluation.load_test_set(Settings().test_set_path)
    assert len(items) >= 10
    assert len({i["id"] for i in items}) == len(items)
    for item in items:
        assert item["question"].strip() and item["ground_truth"].strip()
        assert item["category"] in {"factuelle", "recommandation", "hors_perimetre"}
        assert (item["category"] == "hors_perimetre") == (item["expected_uids"] == [])


def test_test_set_missing_field_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[{"id": "x", "question": "q"}]', encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_test_set(path)


@pytest.mark.parametrize(
    "expected, retrieved, hit, rr",
    [(["a"], ["a", "b"], 1.0, 1.0), (["b"], ["a", "b", "c"], 1.0, 0.5), (["z"], ["a", "b"], 0.0, 0.0), ([], ["a"], None, None)],
)
def test_retrieval_metrics(expected, retrieved, hit, rr):
    assert evaluation.retrieval_metrics(expected, retrieved) == {"hit": hit, "reciprocal_rank": rr}


def test_exact_match_ignores_case_accents_punctuation():
    assert evaluation.exact_match("Le Rex, à 20h !", "le rex a 20h") == 1.0
    assert evaluation.exact_match("Le Rex", "Le Metronum") == 0.0


def test_token_f1():
    assert evaluation.token_f1("concert jazz au Rex", "concert jazz au Rex") == 1.0
    assert evaluation.token_f1("concert jazz", "exposition photo") == 0.0
    assert 0 < evaluation.token_f1("concert jazz samedi", "concert jazz au Rex") < 1


def test_semantic_similarity(embeddings):
    same = evaluation.semantic_similarity("concert jazz Rex", "concert jazz Rex", embeddings)
    different = evaluation.semantic_similarity("concert jazz Rex", "exposition photographie", embeddings)
    assert same == pytest.approx(1.0, abs=1e-5)
    assert different < same


@pytest.mark.parametrize("similarity, verdict", [(0.95, "correcte"), (0.84, "partiellement correcte"), (0.5, "incorrecte")])
def test_classify(similarity, verdict):
    assert evaluation.classify(similarity) == verdict


def test_answer_metrics_keys(embeddings):
    metrics = evaluation.answer_metrics("Concert jazz au Rex", "Concert jazz au Rex le 14/11", embeddings)
    assert set(metrics) == {"exact_match", "token_f1", "semantic_similarity", "verdict"}


def test_summarize_ignores_missing_values():
    rows = [{"hit": 1.0, "f1": 0.5}, {"hit": 0.0, "f1": None}, {"hit": None, "f1": 1.0}]
    assert evaluation.summarize(rows, ["hit", "f1"]) == {"hit": 0.5, "f1": 0.75}


def test_error_report_lists_failures():
    rows = [
        {"id": "q1", "category": "factuelle", "question": "Psykup ?", "hit": 1.0, "verdict": "correcte",
         "faithfulness": 0.9, "retrieved_titles": "Psykup"},
        {"id": "q2", "category": "factuelle", "question": "Jazz ?", "hit": 0.0, "verdict": "incorrecte",
         "semantic_similarity": 0.7, "answer": "Rien", "faithfulness": 0.4, "retrieved_titles": "Rap | Rock"},
        {"id": "q3", "category": "hors_perimetre", "question": "Météo ?", "hit": None, "verdict": "correcte"},
    ]
    report = evaluation.error_report(rows)
    assert "## Recherches manquées (événement attendu absent du top-k) (1)" in report
    assert "**q2**" in report and "Rap | Rock" in report
    assert "## Réponses classées incorrectes (1)" in report
    assert "Faithfulness < 0.7 (hallucination probable) (1)" in report
    assert "| factuelle | 2 | 1 | 1 |" in report
    assert "**q1**" not in report
