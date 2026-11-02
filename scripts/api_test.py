"""Test fonctionnel de l'API en cours d'exécution (locale ou conteneur Docker).

Usage :
    python scripts/api_test.py                          # http://localhost:8000
    python scripts/api_test.py --url http://localhost:8000 --question "Un concert de jazz ?"
"""

import argparse
import os
import sys
import time

import httpx

DEFAULT_QUESTIONS = [
    "Je cherche un concert de rap à Toulouse en décembre.",
    "Quelles activités au musée pour les enfants ?",
    "Y a-t-il des expositions de photographie gratuites ?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--question", action="append", help="question(s) à poser (répétable)")
    args = parser.parse_args()

    failures = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not condition
        print(f"[{'OK ' if condition else 'ERR'}] {name}{' - ' + detail if detail else ''}")

    with httpx.Client(base_url=args.url, timeout=120) as client:
        health = client.get("/health")
        body = health.json()
        check("GET /health", health.status_code == 200 and body["index_loaded"], f"{body['n_vectors']} vecteurs")
        check("LLM configuré", body["llm_configured"])

        check("POST /ask question vide -> 422", client.post("/ask", json={"question": "  "}).status_code == 422)
        check("POST /ask sans corps -> 422", client.post("/ask").status_code == 422)
        check("POST /rebuild sans jeton -> 401/403", client.post("/rebuild").status_code in (401, 403))

        meta = client.get("/metadata")
        check("GET /metadata", meta.status_code == 200 and meta.json()["index_loaded"],
              f"{meta.json().get('city')}, {meta.json().get('n_events')} événements")

        search = client.post("/search", json={"question": "concert", "k": 3})
        check("POST /search", search.status_code == 200 and len(search.json()["sources"]) == 3)
        page2 = client.post("/search", json={"question": "concert", "k": 3, "offset": 3})
        check("POST /search page 2 (offset)", page2.status_code == 200 and not (
            {s["uid"] for s in search.json()["sources"]} & {s["uid"] for s in page2.json()["sources"]}))

        for question in args.question or DEFAULT_QUESTIONS:
            start = time.perf_counter()
            response = client.post("/ask", json={"question": question, "k": 5})
            elapsed = time.perf_counter() - start
            ok = response.status_code == 200 and bool(response.json().get("answer"))
            check(f"POST /ask « {question} »", ok, f"{response.status_code} en {elapsed:.1f}s")
            if ok:
                data = response.json()
                print(f"\nfound={data['found']}\n" + data["answer"])
                print("\nSources :")
                for s in data["sources"]:
                    print(f"  - {s['title']} ({s['dates']}, {s['location_name']}) score={s['score']}")
                print()
            else:
                print(f"     {response.text[:300]}")

        token = os.getenv("ADMIN_TOKEN")
        if token:
            bad = client.post("/rebuild", headers={"X-Admin-Token": token + "x"})
            check("POST /rebuild mauvais jeton -> 401", bad.status_code == 401)

    print(f"\n{'Tous les tests sont passés' if failures == 0 else f'{failures} échec(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
