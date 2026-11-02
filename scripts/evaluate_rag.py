"""Évalue le système RAG sur le jeu de test annoté (data/test_set.json).

Usage :
    python scripts/evaluate_rag.py                   # recherche + réponses + Ragas
    python scripts/evaluate_rag.py --no-ragas        # sans LLM juge (plus rapide, moins d'appels API)
    python scripts/evaluate_rag.py --retrieval-only  # recherche seule, sans clé Mistral
    python scripts/evaluate_rag.py --min-hit-rate 0.7  # code retour 1 si sous le seuil (CI)

Résultats écrits dans reports/ : eval_results.csv (détail), eval_summary.json (moyennes)
et eval_errors.md (analyse des erreurs : recherches manquées, réponses incorrectes, hallucinations).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from puls_rag import evaluation  # noqa: E402
from puls_rag.config import ROOT_DIR, Settings  # noqa: E402
from puls_rag.engine import RAGEngine  # noqa: E402

REPORT_DIR = ROOT_DIR / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=None, help="nombre d'événements récupérés (défaut : TOP_K)")
    parser.add_argument("--retrieval-only", action="store_true", help="n'évaluer que la recherche FAISS")
    parser.add_argument("--no-ragas", action="store_true", help="ne pas lancer les métriques Ragas")
    parser.add_argument("--min-hit-rate", type=float, default=None, help="seuil minimal de hit rate (CI)")
    parser.add_argument("--pause", type=float, default=1.0, help="pause entre deux appels Mistral (s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = Settings()
    engine = RAGEngine(settings).load()
    k = args.k or settings.top_k
    test_set = evaluation.load_test_set(settings.test_set_path)
    print(f"Évaluation sur {len(test_set)} questions (k={k}, index : {engine.n_vectors} vecteurs)\n")

    rows = []
    for item in test_set:
        if args.retrieval_only:
            retrieved = engine.retrieve(item["question"], k)
            result = {"answer": "", "contexts": [d.page_content for d, _ in retrieved],
                      "sources": [{"uid": d.metadata["uid"], "title": d.metadata["title"]} for d, _ in retrieved],
                      "latency_ms": {"total": None}}
        else:
            result = engine.ask(item["question"], k)
            time.sleep(args.pause)

        retrieved_uids = [s["uid"] for s in result["sources"]]
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "retrieved_titles": " | ".join(s["title"] for s in result["sources"]),
            "latency_ms": result["latency_ms"]["total"],
            **evaluation.retrieval_metrics(item["expected_uids"], retrieved_uids),
        }
        if not args.retrieval_only:
            row.update(evaluation.answer_metrics(result["answer"], item["ground_truth"], engine.embeddings))
        rows.append(row)
        status = {1.0: "trouvé", 0.0: "MANQUÉ", None: "hors périmètre"}[row["hit"]]
        print(f"[{item['id']}] recherche : {status:<14} {row.get('verdict', '')}")

    metric_keys = ["hit", "reciprocal_rank"]
    if not args.retrieval_only:
        metric_keys += ["exact_match", "token_f1", "semantic_similarity", "latency_ms"]
        if not args.no_ragas:
            print("\nCalcul des métriques Ragas (LLM juge : %s)..." % settings.llm_model)
            for row, scores in zip(rows, evaluation.ragas_evaluate(rows, engine.llm, engine.embeddings)):
                row.update(scores)
            metric_keys += [c for c in ("faithfulness", "answer_relevancy",
                                        "llm_context_precision_with_reference", "context_recall") if c in rows[0]]

    summary = {
        "n_questions": len(rows),
        "k": k,
        "llm_model": None if args.retrieval_only else settings.llm_model,
        "embedding_model": settings.embedding_model,
        "index": engine.manifest,
        "metrics": evaluation.summarize(rows, metric_keys),
        "metrics_by_category": {
            cat: evaluation.summarize([r for r in rows if r["category"] == cat], metric_keys)
            for cat in sorted({r["category"] for r in rows})
        },
    }
    if "verdict" in rows[0]:
        summary["verdicts"] = pd.Series([r["verdict"] for r in rows]).value_counts().to_dict()

    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "_retrieval" if args.retrieval_only else ""
    pd.DataFrame(rows).drop(columns=["contexts"]).to_csv(REPORT_DIR / f"eval_results{suffix}.csv", index=False)
    (REPORT_DIR / f"eval_summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / f"eval_errors{suffix}.md").write_text(evaluation.error_report(rows), encoding="utf-8")

    print("\n=== Résumé ===")
    for key, value in summary["metrics"].items():
        print(f"  {key:<40} {value}")
    if "verdicts" in summary:
        print(f"  {'verdicts':<40} {summary['verdicts']}")
    print(f"\nDétail : {REPORT_DIR / f'eval_results{suffix}.csv'}")
    print(f"Erreurs : {REPORT_DIR / f'eval_errors{suffix}.md'}")

    hit_rate = summary["metrics"]["hit"]
    if args.min_hit_rate is not None and (hit_rate is None or hit_rate < args.min_hit_rate):
        print(f"ÉCHEC : hit rate {hit_rate} < {args.min_hit_rate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
