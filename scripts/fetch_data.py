"""Étape 1a : télécharge les événements Open Agenda (ville + 1 an d'historique + à venir) dans data/raw/."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puls_rag.config import Settings  # noqa: E402
from puls_rag.fetch import fetch_events, save_raw  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    events = fetch_events(settings)
    path = save_raw(events, settings.raw_path)
    print(f"{len(events)} événements enregistrés dans {path}")
