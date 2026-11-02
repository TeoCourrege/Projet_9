"""Lance les tests pytest et écrit un résumé lisible dans reports/tests/.

Usage :
    python scripts/run_tests.py

Les tests d'intégration s'exécutent automatiquement si l'index existe (scripts/build_index.py),
sinon ils sont ignorés.

Sorties :
    reports/tests/junit.xml     rapport JUnit (lisible par la CI et les IDE)
    reports/tests/summary.md    résumé par module : passés / échoués / ignorés, durée, échecs détaillés
    reports/tests/coverage.txt  couverture de code par fichier (si pytest-cov est installé)
"""

import argparse
import importlib.util
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "tests"


def summarize(junit: Path, coverage: str | None) -> str:
    stats = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "time": 0.0})
    failures = []
    for case in ET.parse(junit).getroot().iter("testcase"):
        module = case.get("classname", "").split(".")[1] if "." in case.get("classname", "") else case.get("classname")
        entry = stats[module]
        entry["time"] += float(case.get("time", 0))
        if case.find("failure") is not None or case.find("error") is not None:
            entry["failed"] += 1
            node = case.find("failure") if case.find("failure") is not None else case.find("error")
            failures.append(f"- `{module}::{case.get('name')}` : {(node.get('message') or '').splitlines()[0][:200]}")
        elif case.find("skipped") is not None:
            entry["skipped"] += 1
        else:
            entry["passed"] += 1

    total = {key: sum(s[key] for s in stats.values()) for key in ("passed", "failed", "skipped", "time")}
    lines = [
        "# Résultats des tests",
        "",
        f"Exécution du {datetime.now():%d/%m/%Y à %H:%M} · Python {sys.version.split()[0]}",
        "",
        f"**{total['passed']} passés · {total['failed']} échoués · {total['skipped']} ignorés** "
        f"en {total['time']:.1f} s",
        "",
        "| Module | Périmètre | Passés | Échoués | Ignorés | Durée (s) |",
        "|---|---|---|---|---|---|",
    ]
    scopes = {
        "test_data_pipeline": "collecte, nettoyage, filtres, chunking",
        "test_indexing": "vectorisation, index FAISS, persistance",
        "test_engine": "recherche, prompt, génération",
        "test_api": "routes, validation, erreurs, sécurité",
        "test_evaluation": "jeu annoté, métriques",
        "test_real_index": "intégration : vraies données et vrai index",
    }
    for module, s in sorted(stats.items()):
        lines.append(f"| `{module}` | {scopes.get(module, '')} | {s['passed']} | {s['failed']} | "
                     f"{s['skipped']} | {s['time']:.2f} |")
    if failures:
        lines += ["", "## Échecs", "", *failures]
    if total["skipped"]:
        lines += ["", "Les tests ignorés sont les tests d'intégration : ils demandent un index construit "
                      "(`scripts/build_index.py`) et s'exécutent automatiquement dès qu'il existe."]
    if coverage:
        lines += ["", "## Couverture de code", "", "```", coverage.strip(), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    junit = OUT / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", f"--junitxml={junit}", "-o", "junit_family=xunit1"]
    has_cov = importlib.util.find_spec("pytest_cov") is not None
    if has_cov:
        cmd += ["--cov=puls_rag", "--cov=api", "--cov-report=term-missing"]

    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(result.stdout[-3000:], result.stderr[-2000:], sep="")

    coverage = None
    if has_cov and "TOTAL" in result.stdout:
        start = result.stdout.find("Name ")
        coverage = result.stdout[start:result.stdout.find("\n", result.stdout.find("TOTAL"))]
        (OUT / "coverage.txt").write_text(coverage + "\n", encoding="utf-8")

    if junit.exists():
        (OUT / "summary.md").write_text(summarize(junit, coverage), encoding="utf-8")
        print(f"Résumé : {OUT / 'summary.md'}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
