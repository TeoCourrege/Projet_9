# Déploie l'API dans Docker (Windows / PowerShell) :
#   1. construction de l'image
#   2. construction de l'index DANS le conteneur (si absent, ou -Rebuild)
#   3. démarrage du service puis attente du healthcheck /health
#
# Usage : .\scripts\deploy.ps1 [-Rebuild]
param([switch]$Rebuild)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .env)) { Write-Host ".env absent : copiez .env.example et renseignez MISTRAL_API_KEY"; exit 1 }

Write-Host "==> Construction de l'image"
docker compose build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Rebuild -or -not (Test-Path faiss_index/index.faiss)) {
    Write-Host "==> Construction de l'index dans le conteneur"
    docker compose run --rm api python scripts/build_index.py --refetch
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "==> Démarrage du service"
docker compose up -d

for ($i = 0; $i -lt 30; $i++) {
    try {
        $health = Invoke-RestMethod http://localhost:8000/health
        Write-Host "==> API prête : http://localhost:8000/docs"
        $health | ConvertTo-Json
        exit 0
    } catch { Start-Sleep -Seconds 2 }
}
Write-Host "L'API ne répond pas : voir docker compose logs api"
exit 1
