"""Étape 1b : nettoie les événements bruts et affiche un aperçu du jeu de données obtenu."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puls_rag.config import Settings  # noqa: E402
from puls_rag.fetch import load_raw  # noqa: E402
from puls_rag.preprocess import chunk_events, clean_events, save_clean  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    raw = load_raw(settings.raw_path)
    df = clean_events(raw, city=settings.city, history_days=settings.history_days)
    save_clean(df, settings.processed_path)
    chunks = chunk_events(df, settings.chunk_size, settings.chunk_overlap)

    print(f"Événements bruts      : {len(raw)}")
    print(f"Événements nettoyés   : {len(df)}")
    print(f"Chunks                : {len(chunks)}")
    print(f"Période couverte      : {df['first_begin'].min():%d/%m/%Y} -> {df['last_end'].max():%d/%m/%Y}")
    print(f"Sans description longue (résumé utilisé) : {(df['description'] == df['summary']).sum()}")
    print(f"Lieux distincts       : {df['location_name'].nunique()}")
    print("\nMots-clés les plus fréquents :")
    print(df["keywords"].explode().str.lower().value_counts().head(15).to_string())
    print(f"\nFichier : {settings.processed_path}")
