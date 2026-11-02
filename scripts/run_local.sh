#!/usr/bin/env bash
# Lance l'ensemble du projet en local (macOS / Linux), sans Docker :
#   1. environnement virtuel + dépendances (si absents)
#   2. vérification de .env
#   3. construction de l'index FAISS (si absent)
#   4. démarrage de l'API sur http://localhost:8000 (Swagger : /docs)
#
# Usage : ./scripts/run_local.sh [--rebuild] [--port 8000]
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8000
REBUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=1 ;;
    --port) PORT="$2"; shift ;;
    *) echo "Option inconnue : $1"; exit 1 ;;
  esac
  shift
done

if [[ ! -d .venv ]]; then
  echo "==> Création de l'environnement virtuel"
  python3.11 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements-dev.txt
fi
PY=.venv/bin/python

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> .env créé depuis .env.example : renseignez MISTRAL_API_KEY (et ADMIN_TOKEN) puis relancez."
  exit 1
fi

"$PY" scripts/check_env.py

if [[ $REBUILD -eq 1 || ! -f faiss_index/index.faiss ]]; then
  echo "==> Construction de l'index FAISS (5 à 10 min la première fois)"
  "$PY" scripts/build_index.py $([[ $REBUILD -eq 1 ]] && echo --refetch)
fi

echo "==> API sur http://localhost:$PORT (documentation : http://localhost:$PORT/docs)"
exec "$PY" -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
