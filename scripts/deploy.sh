#!/usr/bin/env bash
# Déploie l'API dans Docker (macOS / Linux) :
#   1. construction de l'image
#   2. construction de l'index DANS le conteneur (si absent, ou --rebuild)
#   3. démarrage du service puis attente du healthcheck /health
#
# Usage : ./scripts/deploy.sh [--rebuild]
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .env ]] || { echo ".env absent : copiez .env.example et renseignez MISTRAL_API_KEY"; exit 1; }

echo "==> Construction de l'image"
docker compose build

if [[ "${1:-}" == "--rebuild" || ! -f faiss_index/index.faiss ]]; then
  echo "==> Construction de l'index dans le conteneur"
  docker compose run --rm api python scripts/build_index.py --refetch
fi

echo "==> Démarrage du service"
docker compose up -d

for _ in $(seq 1 30); do
  if curl -fs http://localhost:8000/health > /dev/null; then
    echo "==> API prête : http://localhost:8000/docs"
    curl -s http://localhost:8000/health; echo
    exit 0
  fi
  sleep 2
done
echo "L'API ne répond pas : voir docker compose logs api"
exit 1
