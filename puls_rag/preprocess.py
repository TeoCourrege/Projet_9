"""Nettoyage des événements bruts et découpage en chunks LangChain."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PARIS_TZ = ZoneInfo("Europe/Paris")
EXCLUDED_STATUSES = {"Annulé"}
# Agendas sources hors périmètre culturel (emploi, industrie, rénovation...) : motifs insensibles à la casse.
EXCLUDED_AGENDA_PATTERNS = (
    "france travail",
    "semaine de l'industrie",
    "renov'energie",
    "geovelo",
    "semaine des métiers",
    "solidarités",
    "irsd",
)

COLUMNS = [
    "uid",
    "title",
    "summary",
    "description",
    "keywords",
    "conditions",
    "daterange",
    "first_begin",
    "last_end",
    "location_name",
    "address",
    "district",
    "postal_code",
    "city",
    "latitude",
    "longitude",
    "status",
    "attendance_mode",
    "age_min",
    "age_max",
    "origin_agenda",
    "url",
    "updated_at",
]


def clean_text(text: str | None) -> str:
    """Supprime le HTML et normalise les espaces. Retourne '' pour une valeur vide."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    if "<" in text:
        soup = BeautifulSoup(text, "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for block in soup.find_all(["p", "li", "div", "h1", "h2", "h3", "h4"]):
            block.insert_after("\n")
        text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _label(value) -> str:
    """Extrait le libellé français des champs Open Agenda encodés en JSON (status, attendancemode)."""
    if not value:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        label = value.get("label", {})
        return label.get("fr", "") if isinstance(label, dict) else str(label)
    return str(value)


def _to_datetime(value) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def normalize_event(raw: dict) -> dict:
    """Transforme un enregistrement brut Open Agenda en dictionnaire plat et propre."""
    coords = raw.get("location_coordinates") or {}
    keywords = raw.get("keywords_fr") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    return {
        "uid": str(raw.get("uid", "")).strip(),
        "title": clean_text(raw.get("title_fr")),
        "summary": clean_text(raw.get("description_fr")),
        "description": clean_text(raw.get("longdescription_fr")),
        "keywords": [clean_text(k) for k in keywords if clean_text(k)],
        "conditions": clean_text(raw.get("conditions_fr")),
        "daterange": clean_text(raw.get("daterange_fr")),
        "first_begin": _to_datetime(raw.get("firstdate_begin")),
        "last_end": _to_datetime(raw.get("lastdate_end")),
        "location_name": clean_text(raw.get("location_name")),
        "address": clean_text(raw.get("location_address")),
        "district": clean_text(raw.get("location_district")),
        "postal_code": clean_text(raw.get("location_postalcode")),
        "city": clean_text(raw.get("location_city")),
        "latitude": coords.get("lat"),
        "longitude": coords.get("lon"),
        "status": _label(raw.get("status")),
        "attendance_mode": _label(raw.get("attendancemode")),
        "age_min": raw.get("age_min"),
        "age_max": raw.get("age_max"),
        "origin_agenda": clean_text(raw.get("originagenda_title")),
        "url": raw.get("canonicalurl") or "",
        "updated_at": raw.get("updatedat") or "",
    }


def clean_events(raw_events: list[dict], city: str, history_days: int = 365, today: date | None = None) -> pd.DataFrame:
    """Normalise, filtre (ville, période, statut) et déduplique les événements."""
    today = today or datetime.now(PARIS_TZ).date()
    df = pd.DataFrame([normalize_event(e) for e in raw_events], columns=COLUMNS)
    n_start = len(df)

    df = df[(df["uid"] != "") & (df["title"] != "")]
    df = df[df["city"].str.casefold() == city.casefold()]

    cutoff = pd.Timestamp(today - timedelta(days=history_days), tz="UTC")
    df = df[df["last_end"].notna() & (df["last_end"] >= cutoff)]
    df = df[~df["status"].isin(EXCLUDED_STATUSES)]
    excluded_agenda = df["origin_agenda"].str.casefold().apply(
        lambda agenda: any(pattern in agenda for pattern in EXCLUDED_AGENDA_PATTERNS)
    )
    df = df[~excluded_agenda]

    # Doublons : même identifiant, ou même événement publié dans plusieurs agendas.
    df = df.sort_values("updated_at", ascending=False)
    df = df.drop_duplicates(subset="uid")
    df = df.drop_duplicates(subset=["title", "location_name", "first_begin"])

    # Description manquante : on se rabat sur le résumé, sinon sur le titre.
    df["description"] = df["description"].where(df["description"] != "", df["summary"])
    df["description"] = df["description"].where(df["description"] != "", df["title"])

    df = df.sort_values("first_begin").reset_index(drop=True)
    logger.info("Nettoyage : %d -> %d événements", n_start, len(df))
    return df


def save_clean(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", force_ascii=False, indent=1, date_format="iso")
    return path


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", dtype={"uid": str, "postal_code": str})
    for col in ("first_begin", "last_end"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def format_date(ts) -> str:
    if ts is None or pd.isna(ts):
        return "date inconnue"
    return pd.Timestamp(ts).tz_convert(PARIS_TZ).strftime("%d/%m/%Y")


def _age_text(age_min, age_max) -> str:
    has_min = age_min is not None and not pd.isna(age_min)
    has_max = age_max is not None and not pd.isna(age_max)
    if has_min and has_max:
        return f"de {int(age_min)} à {int(age_max)} ans"
    if has_min:
        return f"à partir de {int(age_min)} ans"
    if has_max:
        return f"jusqu'à {int(age_max)} ans"
    return ""


def event_header(event: dict) -> str:
    """En-tête structuré répété dans chaque chunk : chaque morceau reste autoporteur
    (titre, dates, lieu) même si la description est découpée."""
    lines = [f"Titre : {event['title']}"]
    dates = f"du {format_date(event['first_begin'])} au {format_date(event['last_end'])}"
    if event.get("daterange"):
        dates += f" ({event['daterange']})"
    lines.append(f"Dates : {dates}")
    place = ", ".join(p for p in (event.get("location_name"), event.get("address")) if p)
    if place:
        lines.append(f"Lieu : {place}")
    if event.get("district"):
        lines.append(f"Quartier : {event['district']}")
    keywords = event.get("keywords") or []
    if len(keywords):
        lines.append(f"Mots-clés : {', '.join(keywords)}")
    if event.get("conditions"):
        lines.append(f"Tarif / conditions : {event['conditions']}")
    age = _age_text(event.get("age_min"), event.get("age_max"))
    if age:
        lines.append(f"Public : {age}")
    if event.get("status") and event["status"] != "Programmé":
        lines.append(f"Statut : {event['status']}")
    if event.get("summary") and event["summary"] != event.get("description"):
        lines.append(f"Résumé : {event['summary']}")
    return "\n".join(lines)


def event_metadata(event: dict) -> dict:
    keywords = event.get("keywords") or []
    return {
        "uid": event["uid"],
        "title": event["title"],
        "first_begin": event["first_begin"].isoformat() if pd.notna(event["first_begin"]) else "",
        "last_end": event["last_end"].isoformat() if pd.notna(event["last_end"]) else "",
        "dates": f"du {format_date(event['first_begin'])} au {format_date(event['last_end'])}",
        "location_name": event.get("location_name", ""),
        "address": event.get("address", ""),
        "city": event.get("city", ""),
        "keywords": ", ".join(keywords),
        "conditions": event.get("conditions", ""),
        "url": event.get("url", ""),
    }


def chunk_events(df: pd.DataFrame, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Document]:
    """Découpe chaque événement en chunks « en-tête + morceau de description ».

    La taille de chunk vise la description : l'en-tête est ajouté par-dessus, ce qui reste
    sous la limite de 512 tokens du modèle d'embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for event in df.to_dict(orient="records"):
        header = event_header(event)
        metadata = event_metadata(event)
        parts = splitter.split_text(event["description"]) or [""]
        for i, part in enumerate(parts):
            documents.append(
                Document(
                    page_content=f"{header}\nDescription : {part}".strip(),
                    metadata={**metadata, "chunk_id": i, "n_chunks": len(parts)},
                )
            )
    logger.info("%d événements découpés en %d chunks", len(df), len(documents))
    return documents