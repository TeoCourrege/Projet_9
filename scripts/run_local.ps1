# Lance l'ensemble du projet en local (Windows / PowerShell), sans Docker :
#   1. environnement virtuel + dépendances (si absents)
#   2. vérification de .env
#   3. construction de l'index FAISS (si absent)
#   4. démarrage de l'API sur http://localhost:8000 (Swagger : /docs)
#
# Usage : .\scripts\run_local.ps1 [-Rebuild] [-Port 8000]
param(
    [switch]$Rebuild,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) {
    Write-Host "==> Création de l'environnement virtuel"
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
}
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "==> .env créé depuis .env.example : renseignez MISTRAL_API_KEY (et ADMIN_TOKEN) puis relancez."
    exit 1
}

& $py scripts/check_env.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Rebuild -or -not (Test-Path faiss_index/index.faiss)) {
    Write-Host "==> Construction de l'index FAISS (5 à 10 min la première fois)"
    if ($Rebuild) { & $py scripts/build_index.py --refetch } else { & $py scripts/build_index.py }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "==> API sur http://localhost:$Port (documentation : http://localhost:$Port/docs)"
& $py -m uvicorn api.main:app --host 0.0.0.0 --port $Port
