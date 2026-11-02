"""Compare les modèles d'embedding (E5 local vs mistral-embed) sur la qualité de la recherche.

Pour chaque modèle : vectorisation des mêmes chunks (data/processed/events_clean.json),
index FAISS dédié, puis hit rate@k, MRR et latence de recherche sur le jeu de test annoté.
Sans LLM de génération : seule la clé Mistral est nécessaire, pour mistral-embed.

Usage :
    python scripts/build_index.py                 # produit data/processed/events_clean.json
    python scripts/compare_embeddings.py
    python scripts/compare_embeddings.py --providers huggingface   # un seul modèle

Résultat : reports/embeddings_comparison.json + tableau affiché.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puls_rag import evaluation, indexing, preprocess  # noqa: E402
from puls_rag.config import ROOT_DIR, Settings  # noqa: E402
from puls_rag.engine import RAGEngine  # noqa: E402

MODELS = {
    "huggingface": "intfloat/multilingual-e5-small",
    "mistral": "mistral-embed",
}


def evaluate_provider(base: Settings, provider: str, documents, test_set, k: int) -> dict:
    settings = replace(base, embedding_provider=provider, embedding_model=MODELS[provider])
    embeddings = indexing.get_embeddings(settings)

    start = time.perf_counter()
    vectorstore = indexing.build_vectorstore(documents, embeddings, settings.index_type)
    build_s = time.perf_counter() - start

    engine = RAGEngine(settings, embeddings=embeddings, vectorstore=vectorstore)
    rows, latencies = [], []
    for item in test_set:
        t0 = time.perf_counter()
        retrieved = engine.retrieve(item["question"], k)
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({**item, **evaluation.retrieval_metrics(item["expected_uids"],
                                                            [d.metadata["uid"] for d, _ in retrieved])})

    metrics = evaluation.summarize(rows, ["hit", "reciprocal_rank"])
    return {
        "model": settings.embedding_model,
        "dimension": vectorstore.index.d,
        "hit_rate": metrics["hit"],
        "mrr": metrics["reciprocal_rank"],
        "missed": [r["id"] for r in rows if r["hit"] == 0.0],
        "index_build_s": round(build_s, 1),
        "mean_search_ms": round(sum(latencies) / len(latencies), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--providers", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = Settings()
    if not settings.processed_path.exists():
        print(f"{settings.processed_path} absent : lancez d'abord scripts/build_index.py")
        return 1
    if "mistral" in args.providers and not settings.mistral_api_key:
        print("MISTRAL_API_KEY absente : nécessaire pour mistral-embed (ou --providers huggingface)")
        return 1

    k = args.k or settings.top_k
    documents = preprocess.chunk_events(preprocess.load_clean(settings.processed_path),
                                        settings.chunk_size, settings.chunk_overlap)
    test_set = evaluation.load_test_set(settings.test_set_path)
    print(f"{len(documents)} chunks, {len(test_set)} questions, k={k}\n")

    results = {}
    for provider in args.providers:
        print(f"-> {provider} ({MODELS[provider]})...")
        results[provider] = evaluate_provider(settings, provider, documents, test_set, k)

    print(f"\n{'modèle':<32} {'dim':>5} {'hit@k':>7} {'MRR':>7} {'build (s)':>10} {'recherche (ms)':>15}")
    for r in results.values():
        print(f"{r['model']:<32} {r['dimension']:>5} {r['hit_rate']:>7} {r['mrr']:>7} "
              f"{r['index_build_s']:>10} {r['mean_search_ms']:>15}")

    out = ROOT_DIR / "reports" / "embeddings_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"k": k, "n_chunks": len(documents), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
