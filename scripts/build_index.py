"""Reconstruit la base vectorielle FAISS à partir des données Open Agenda.

Usage :
    python scripts/build_index.py              # réutilise data/raw si présent
    python scripts/build_index.py --refetch    # re-télécharge les événements
    python scripts/build_index.py --index-type hnsw
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puls_rag.config import Settings  # noqa: E402
from puls_rag.pipeline import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refetch", action="store_true", help="re-télécharger les données Open Agenda")
    parser.add_argument("--index-type", choices=["flat", "hnsw", "ivf"], help="algorithme FAISS")
    parser.add_argument("--city", help="ville ciblée (défaut : OA_CITY ou Toulouse)")
    args = parser.parse_args()

    if args.index_type:
        os.environ["INDEX_TYPE"] = args.index_type
    if args.city:
        os.environ["OA_CITY"] = args.city

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    settings = Settings()
    _, report = build_index(settings, refetch=args.refetch)
    print("\nIndex construit dans", settings.index_dir)
    for key, value in asdict(report).items():
        print(f"  {key:<15} {value}")


if __name__ == "__main__":
    main()
