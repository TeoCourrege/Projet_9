# API REST Puls-Events : référence

Base URL locale : `http://localhost:8000`. Documentation interactive : `/docs` (Swagger) et `/redoc`.
Toutes les réponses sont en JSON (UTF-8). Collection Postman : [`postman_collection.json`](postman_collection.json).

| Méthode | Route | Usage métier | Authentification |
|---|---|---|---|
| `POST` | [`/ask`](#post-ask) | Poser une question, recevoir une recommandation rédigée et ses sources | — |
| `POST` | [`/search`](#post-search) | Lister les événements les plus proches d'une requête, sans LLM, page par page | — |
| `GET` | [`/metadata`](#get-metadata) | Connaître le périmètre couvert (ville, période, modèles, taille de l'index) | — |
| `GET` | [`/health`](#get-health) | Supervision : le service est-il prêt ? | — |
| `POST` | [`/rebuild`](#post-rebuild) | Reconstruire l'index après une mise à jour des données | `X-Admin-Token` |

---

## `POST /ask`

**Corps**

| Champ | Type | Obligatoire | Contraintes | Défaut |
|---|---|---|---|---|
| `question` | string | oui | 1 à 1 000 caractères, pas uniquement des espaces | — |
| `k` | int | non | 1 à 20 : nombre d'événements fournis au LLM | 5 |

**Réponse 200**

```json
{
  "question": "Quand joue Psykup et à quel prix ?",
  "answer": "Psykup joue le samedi 12 décembre 2026 à 20h00 au Metronum. Places à partir de 25,5 €.",
  "found": true,
  "sources": [
    {
      "uid": "17120831",
      "title": "Psykup",
      "dates": "Samedi 12 décembre, 20h00",
      "location_name": "Le Metronum",
      "url": "https://openagenda.com/...",
      "score": 0.8912
    }
  ],
  "latency_ms": {"retrieval": 18, "total": 1430},
  "model": "mistral-small-latest"
}
```

- `found` vaut `false` quand aucun événement ne dépasse le seuil `MIN_SCORE`. `answer` contient alors un message fixe invitant à reformuler, `sources` est vide et le LLM n'est pas appelé.
- `score` est la similarité cosinus entre la question et l'événement (1 = identique).

## `POST /search`

Recherche sémantique seule : rapide, gratuite, et toujours disponible même si Mistral est indisponible.

| Champ | Type | Contraintes | Défaut |
|---|---|---|---|
| `question` | string | comme `/ask` | — |
| `k` | int | 1 à 20 : taille de page | 5 |
| `offset` | int | 0 à 100 : nombre d'événements à sauter | 0 |

**Réponse 200**

```json
{
  "question": "concert",
  "k": 5,
  "offset": 0,
  "next_offset": 5,
  "sources": [ { "uid": "...", "title": "...", "dates": "...", "location_name": "...", "url": "...", "score": 0.85 } ]
}
```

`next_offset` donne l'offset de la page suivante, et vaut `null` sur la dernière page.

## `GET /metadata`

```json
{
  "service": "Puls-Events RAG",
  "version": "1.0.0",
  "city": "Toulouse",
  "history_days": 365,
  "data_source": "Open Agenda (evenements-publics-openagenda, OpenDataSoft)",
  "embedding": {"provider": "huggingface", "model": "intfloat/multilingual-e5-small"},
  "llm": {"model": "mistral-small-latest", "temperature": 0.1, "configured": true},
  "retrieval": {"top_k": 5, "min_score": 0.0, "index_type": "flat"},
  "index_loaded": true,
  "n_vectors": 5900,
  "n_events": 4000,
  "index_built_at": "2026-10-27T09:12:00+00:00",
  "endpoints": {"POST /ask": "..."}
}
```

## `GET /health`

```json
{"status": "ok", "index_loaded": true, "n_vectors": 5900, "llm_configured": true, "index": { "...": "manifeste de l'index" }}
```

`status` vaut `index_missing` tant que l'index n'est pas construit : l'API démarre quand même, pour permettre un `/rebuild`.

## `POST /rebuild`

En-tête `X-Admin-Token: <ADMIN_TOKEN>`. Corps optionnel : `{"refetch": true}` pour re-télécharger Open Agenda avant l'indexation. L'opération dure plusieurs minutes sur CPU. Le nouvel index remplace l'ancien à chaud.

```json
{"status": "Index reconstruit", "n_raw_events": 5470, "n_clean_events": 4000, "n_chunks": 5900, "n_vectors": 5900, "index_type": "flat", "duration_s": 312.4}
```

---

## Erreurs

Toutes les erreurs ont la forme `{"detail": "..."}`. Pour `422`, `detail` est la liste des champs invalides.

| Code | Cas | Action côté client |
|---|---|---|
| `422` | Question vide, trop longue, `k` ou `offset` hors bornes, JSON invalide | Corriger la requête |
| `401` | `X-Admin-Token` absent ou invalide | — |
| `403` | `/rebuild` désactivé (`ADMIN_TOKEN` non défini) | — |
| `503` | Index non construit, ou `MISTRAL_API_KEY` absente | Appeler `/rebuild`, configurer la clé. `/search` reste utilisable |
| `502` | Erreur du fournisseur LLM (quota, réseau) | Réessayer plus tard, ou se rabattre sur `/search` |
| `500` | Échec de la reconstruction de l'index | Voir les logs |

---

## Exemples

### curl

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Une activité au musée pour mon bébé ?", "k": 5}'

curl -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d '{"question": "concert de jazz", "k": 5, "offset": 5}'

curl localhost:8000/metadata
curl localhost:8000/health

curl -X POST localhost:8000/rebuild -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Content-Type: application/json" -d '{"refetch": false}'
```

### PowerShell

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"question": "Y a-t-il une soirée techno gratuite début décembre ?"}'
```

### Python (service consommateur)

```python
import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=60) as api:
    response = api.post("/ask", json={"question": "Un concert de rap en décembre ?"})
    if response.status_code == 502:          # Mistral indisponible : repli sur la recherche seule
        response = api.post("/search", json={"question": "Un concert de rap en décembre ?"})
    response.raise_for_status()
    data = response.json()

    if data.get("found", True):
        print(data.get("answer", ""))
        for event in data["sources"]:
            print(f"- {event['title']} | {event['dates']} | {event['location_name']} | {event['url']}")
    else:
        print("Aucun événement correspondant")
```

`scripts/api_test.py` enchaîne ces appels sur une API lancée et vérifie les codes de retour.
