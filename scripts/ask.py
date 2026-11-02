"""Chatbot en ligne de commande (sans passer par l'API).

Usage :
    python scripts/ask.py "Quels concerts de jazz en novembre ?"
    python scripts/ask.py            # mode interactif (quitter avec Entrée vide)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puls_rag.engine import RAGEngine  # noqa: E402


def answer(engine: RAGEngine, question: str) -> None:
    result = engine.ask(question)
    print(f"\n{result['answer']}\n")
    print("Sources :")
    for s in result["sources"]:
        print(f"  - {s['title']} | {s['dates']} | {s['location_name']} (score {s['score']})")
    print(f"({result['latency_ms']['total']} ms, modèle {result['model']})\n")


if __name__ == "__main__":
    rag = RAGEngine().load()
    if len(sys.argv) > 1:
        answer(rag, " ".join(sys.argv[1:]))
    else:
        while question := input("Question > ").strip():
            answer(rag, question)
