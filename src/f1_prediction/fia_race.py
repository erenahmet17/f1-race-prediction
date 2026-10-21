"""FIA Final Race Classification belgesini bulur, saklar ve okur."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .fia_event import fia_event_page_name
from .race import RaceResultError


FIA_BASE_URL = "https://www.fia.com"
DOCUMENT_PATTERN = re.compile(
    r"^Doc\s+(?P<number>\d+)\s+-\s+Final Race Classification\s+"
    r"Published on\s+(?P<date>\d{2}\.\d{2}\.\d{2})\s+"
    r"(?P<time>\d{2}:\d{2})\s+(?P<timezone>[A-Z]+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FiaRaceDocument:
    event_name: str
    year: int
    document_number: str
    published_at: pd.Timestamp
    documents_page_url: str
    pdf_url: str


@dataclass(frozen=True)
class FiaRaceSnapshot:
    classification: pd.DataFrame
    document: FiaRaceDocument
    pdf_sha256: str
    pdf_path: Path | None


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._href: str | None = None
        self._text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a" or self._href is not None:
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((" ".join("".join(self._text).split()), self._href))
            self._href = None
            self._text = []


def _publication_timestamp(date_text: str, time_text: str) -> pd.Timestamp:
    naive = datetime.strptime(f"{date_text} {time_text}", "%d.%m.%y %H:%M")
    return pd.Timestamp(naive.replace(tzinfo=ZoneInfo("Europe/Paris")))


def parse_final_race_links(
    html: str,
    *,
    event_name: str,
    year: int,
    documents_page_url: str,
) -> list[FiaRaceDocument]:
    """Yalnız tam adlı FIA Final Race Classification bağlantılarını döndürür."""
    parser = _AnchorParser()
    parser.feed(html)
    documents: list[FiaRaceDocument] = []
    for text, href in parser.links:
        match = DOCUMENT_PATTERN.fullmatch(text)
        if match is None:
            continue
        published_at = _publication_timestamp(match["date"], match["time"])
        if published_at.year != year:
            continue
        documents.append(
            FiaRaceDocument(
                event_name=event_name,
                year=year,
                document_number=match["number"],
                published_at=published_at,
                documents_page_url=documents_page_url,
                pdf_url=urljoin(documents_page_url, href),
            )
        )
    return sorted(documents, key=lambda item: item.published_at)


def discover_final_race_document(
    event_name: str,
    year: int,
    *,
    timeout: float = 60.0,
    get: Callable[..., requests.Response] = requests.get,
) -> FiaRaceDocument:
    page_name = fia_event_page_name(event_name)
    page_url = f"{FIA_BASE_URL}/documents/event/{quote(page_name)}"
    response = get(
        page_url,
        timeout=timeout,
        headers={"User-Agent": "f1-prediction-v2/1.0"},
    )
    response.raise_for_status()
    documents = parse_final_race_links(
        response.text,
        event_name=event_name,
        year=year,
        documents_page_url=page_url,
    )
    if not documents:
        raise RaceResultError(
            f"{year} {event_name} için FIA Final Race Classification henüz yayımlanmadı"
        )
    return documents[-1]


def _download_pdf(
    document: FiaRaceDocument,
    *,
    timeout: float,
    get: Callable[..., requests.Response] = requests.get,
) -> bytes:
    response = get(
        document.pdf_url,
        timeout=timeout,
        headers={"User-Agent": "f1-prediction-v2/1.0"},
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RaceResultError("FIA final yarış bağlantısı geçerli bir PDF döndürmedi")
    return response.content


def _integer_word(word: dict[str, object]) -> int | None:
    text = str(word.get("text", "")).strip()
    return int(text) if re.fullmatch(r"\d{1,2}", text) else None


def _center_x(word: dict[str, object]) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2


def parse_classification_page_words(
    words: list[dict[str, object]], page_width: float
) -> pd.DataFrame:
    """FIA zaman çizelgesindeki sıralı ve NOT CLASSIFIED araçları okur."""
    if page_width <= 0:
        raise RaceResultError("PDF sayfa genişliği geçersiz")
    normalized = [
        dict(word, normalized=str(word.get("text", "")).strip().upper())
        for word in words
    ]
    header_tops = [
        float(word["top"])
        for word in normalized
        if word["normalized"] == "NO" and _center_x(word) / page_width < 0.10
    ]
    not_classified_tops = [
        float(word["top"])
        for word in normalized
        if word["normalized"] == "CLASSIFIED"
    ]
    fastest_tops = [
        float(word["top"])
        for word in normalized
        if word["normalized"] == "FASTEST"
    ]
    if not header_tops or not not_classified_tops:
        return pd.DataFrame(columns=["CarNumber", "RaceOfficialPosition"])

    header_top = min(header_tops)
    not_classified_top = min(not_classified_tops)
    fastest_top = min(
        (top for top in fastest_tops if top > not_classified_top),
        default=float("inf"),
    )
    rows: list[dict[str, object]] = []

    for word in normalized:
        position = _integer_word(word)
        if position is None:
            continue
        top = float(word["top"])
        position_x = _center_x(word) / page_width
        if not (header_top < top < not_classified_top and 0.035 <= position_x <= 0.065):
            continue
        cars = [
            _integer_word(other)
            for other in normalized
            if abs(float(other["top"]) - top) <= 2.0
            and 0.070 <= _center_x(other) / page_width <= 0.105
            and _integer_word(other) is not None
        ]
        if len(cars) == 1:
            rows.append(
                {"CarNumber": str(cars[0]), "RaceOfficialPosition": position}
            )

    for word in normalized:
        car_number = _integer_word(word)
        if car_number is None:
            continue
        top = float(word["top"])
        car_x = _center_x(word) / page_width
        if not_classified_top < top < fastest_top and 0.070 <= car_x <= 0.105:
            rows.append(
                {"CarNumber": str(car_number), "RaceOfficialPosition": pd.NA}
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["CarNumber", "RaceOfficialPosition"])
    if frame["CarNumber"].duplicated().any():
        raise RaceResultError("FIA sınıflandırmasında tekrarlanan araç var")
    known = frame["RaceOfficialPosition"].dropna().astype(int).tolist()
    if sorted(known) != list(range(1, len(known) + 1)):
        raise RaceResultError("FIA yarış pozisyonları kesintisiz okunamadı")
    return frame.reset_index(drop=True)


def parse_final_race_pdf(
    content: bytes, document: FiaRaceDocument
) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("FIA PDF okuyucusu için 'pdfplumber' kurulmalı") from exc

    with pdfplumber.open(BytesIO(content)) as pdf:
        text = " ".join((page.extract_text() or "") for page in pdf.pages).upper()
        if "FINAL RACE CLASSIFICATION" not in text:
            raise RaceResultError("PDF, FIA Final Race Classification belgesi değil")
        if str(document.year) not in text:
            raise RaceResultError("FIA yarış PDF sezonu istenen sezonla uyuşmuyor")
        if not re.search(
            rf"\b(?:DOC(?:UMENT)?\s*)?{re.escape(document.document_number)}\b",
            text,
        ):
            raise RaceResultError("FIA yarış PDF belge numarası bağlantıyla uyuşmuyor")
        parsed = [
            parse_classification_page_words(page.extract_words(), float(page.width))
            for page in pdf.pages
        ]
        classification = max(parsed, key=len)
        if len(classification) < 2:
            raise RaceResultError("FIA PDF'sinden yarış sınıflandırması okunamadı")
        return classification


def _attach_driver_ids(
    classification: pd.DataFrame, race_results: pd.DataFrame
) -> pd.DataFrame:
    required = {"DriverId", "DriverNumber"}
    missing = sorted(required.difference(race_results.columns))
    if missing:
        raise RaceResultError(
            f"Yarış pilot eşleştirmesi eksik: {', '.join(missing)}"
        )
    mapping = race_results[["DriverId", "DriverNumber"]].copy()
    mapping["CarNumber"] = (
        pd.to_numeric(mapping["DriverNumber"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )
    if mapping["CarNumber"].isna().any() or mapping["DriverId"].isna().any():
        raise RaceResultError("Yarış pilot/araç kimliği boş olamaz")
    if mapping["CarNumber"].duplicated().any():
        raise RaceResultError("Yarış verisinde tekrarlanan araç numarası var")

    merged = classification.merge(
        mapping[["CarNumber", "DriverId"]], on="CarNumber", how="left"
    )
    if merged["DriverId"].isna().any():
        unknown = sorted(merged.loc[merged["DriverId"].isna(), "CarNumber"])
        raise RaceResultError(f"FIA yarış araçları pilotlarla eşleşmedi: {unknown}")
    expected = set(mapping["DriverId"])
    observed = set(merged["DriverId"])
    if expected != observed:
        raise RaceResultError(
            "FIA yarış pilot listesi FastF1 ile uyuşmuyor. "
            f"Eksik: {sorted(expected - observed)}; Fazla: {sorted(observed - expected)}"
        )
    return merged


def _store_pdf(
    content: bytes,
    document: FiaRaceDocument,
    raw_data_dir: Path,
    sha256: str,
) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", document.event_name.lower()).strip("_")
    directory = raw_data_dir / "fia" / str(document.year) / slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"final_race_classification_doc_{document.document_number}_{sha256[:12]}.pdf"
    )
    if not path.exists():
        path.write_bytes(content)
    return path


def fetch_fia_final_race_classification(
    event_name: str,
    year: int,
    race_results: pd.DataFrame,
    *,
    raw_data_dir: Path | None = None,
    timeout: float = 60.0,
) -> FiaRaceSnapshot:
    document = discover_final_race_document(event_name, year, timeout=timeout)
    content = _download_pdf(document, timeout=timeout)
    sha256 = hashlib.sha256(content).hexdigest()
    parsed = parse_final_race_pdf(content, document)
    classification = _attach_driver_ids(parsed, race_results)
    path = (
        _store_pdf(content, document, raw_data_dir, sha256)
        if raw_data_dir is not None
        else None
    )
    return FiaRaceSnapshot(
        classification=classification,
        document=document,
        pdf_sha256=sha256,
        pdf_path=path,
    )
