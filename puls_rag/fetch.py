"""Collecte des événements Open Agenda.

Source : le jeu de données public « evenements-publics-openagenda », publié par Open Agenda
sur OpenDataSoft. Il agrège tous les agendas publics d'Open Agenda et s'interroge sans clé,
avec un filtrage côté serveur (ODSQL) sur la ville et la période.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from puls_rag.config import Settings

logger = logging.getLogger(__name__)

# Champs utiles pour le RAG : on évite de télécharger les champs inutilisés (images, contacts...).
FIELDS = [
    "uid",
    "canonicalurl",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "updatedat",
    "daterange_fr",
    "firstdate_begin",
    "firstdate_end",
    "lastdate_begin",
    "lastdate_end",
    "location_name",
    "location_address",
    "location_district",
    "location_postalcode",
    "location_city",
    "location_department",
    "location_coordinates",
    "attendancemode",
    "onlineaccesslink",
    "status",
    "age_min",
    "age_max",
    "originagenda_title",
]


def build_where_clause(city: str, since: date) -> str:
    """Filtre ODSQL : événements de la ville dont la dernière date se termine après `since`.

    On garde donc les événements passés de moins d'un an ET tous les événements à venir.
    """
    safe_city = city.replace('"', "")
    return f'location_city="{safe_city}" AND lastdate_end >= date\'{since.isoformat()}\''


def fetch_events(settings: Settings | None = None, today: date | None = None, timeout: int = 120) -> list[dict]:
    """Télécharge tous les événements filtrés (endpoint d'export : pas de pagination nécessaire)."""
    settings = settings or Settings()
    today = today or datetime.now().date()
    since = today - timedelta(days=settings.history_days)

    params = {"where": build_where_clause(settings.city, since), "select": ",".join(FIELDS)}
    url = f"{settings.oa_base_url}/exports/json"
    logger.info("Téléchargement des événements de %s depuis le %s", settings.city, since)

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    events = response.json()
    if not isinstance(events, list):
        raise ValueError(f"Réponse inattendue de l'API Open Agenda : {type(events).__name__}")

    logger.info("%d événements récupérés", len(events))
    return events


def save_raw(events: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_raw(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
