"""FIA karar sayfasından final start gridini bulur, saklar ve okur."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote, urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .fia_event import fia_event_page_name
from .grid import (
    FINAL_DOCUMENT_TYPE,
    FinalGridNotReadyError,
    GridDataError,
    GridDocumentMetadata,
    build_final_grid_features,
)


FIA_BASE_URL = "https://www.fia.com"
DOCUMENT_PATTERN = re.compile(
    r"^Doc\s+(?P<number>\d+)\s+-\s+Final Starting Grid\s+"
    r"Published on\s+(?P<date>\d{2}\.\d{2}\.\d{2})\s+"
    r"(?P<time>\d{2}:\d{2})\s+(?P<timezone>[A-Z]+)$",
    flags=re.IGNORECASE,
)

PENALTY_COLUMNS = (
    "GridPenaltyNotePresent",
    "GridPenaltyApplied",
    "GridPenaltyPositions",
    "GridPenaltyPowerUnit",
    "GridPenaltySporting",
    "GridPenaltyParcFerme",
    "GridPenaltyTechnical",
    "GridPenaltyPitLane",
    "GridPermissionToStart",
    "GridPenaltyOther",
    "GridPenaltyReason",
    "GridPenaltyText",
)


@dataclass(frozen=True)
class FiaGridDocument:
    event_name: str
    year: int
    document_number: str
    published_at: pd.Timestamp
    documents_page_url: str
    pdf_url: str


@dataclass(frozen=True)
class FiaGridSnapshot:
    features: pd.DataFrame
    document: FiaGridDocument
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
        attributes = dict(attrs)
        if attributes.get("href"):
            self._href = attributes["href"]
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            text = " ".join("".join(self._text).split())
            self.links.append((text, self._href))
            self._href = None
            self._text = []


def _publication_timestamp(date_text: str, time_text: str) -> pd.Timestamp:
    naive = datetime.strptime(f"{date_text} {time_text}", "%d.%m.%y %H:%M")
    # FIA sayfası etikette CET yazsa da yaz aylarında yerel Avrupa saatini gösterir.
    return pd.Timestamp(naive.replace(tzinfo=ZoneInfo("Europe/Paris")))


def parse_final_grid_links(
    html: str,
    *,
    event_name: str,
    year: int,
    documents_page_url: str,
) -> list[FiaGridDocument]:
    """FIA HTML sayfasındaki tam adlı final-grid belgelerini döndürür."""
    parser = _AnchorParser()
    parser.feed(html)
    documents: list[FiaGridDocument] = []
    for text, href in parser.links:
        match = DOCUMENT_PATTERN.fullmatch(text)
        if match is None:
            continue
        published_at = _publication_timestamp(match["date"], match["time"])
        if published_at.year != year:
            continue
        documents.append(
            FiaGridDocument(
                event_name=event_name,
                year=year,
                document_number=match["number"],
                published_at=published_at,
                documents_page_url=documents_page_url,
                pdf_url=urljoin(documents_page_url, href),
            )
        )
    return sorted(documents, key=lambda item: item.published_at)


def discover_final_grid_document(
    event_name: str,
    year: int,
    *,
    timeout: float = 60.0,
    get: Callable[..., requests.Response] = requests.get,
) -> FiaGridDocument:
    """Etkinlik sayfasında o yıla ait son FIA final-grid belgesini bulur."""
    page_name = fia_event_page_name(event_name)
    page_url = f"{FIA_BASE_URL}/documents/event/{quote(page_name)}"
    response = get(
        page_url,
        timeout=timeout,
        headers={"User-Agent": "f1-prediction-v2/1.0"},
    )
    response.raise_for_status()
    documents = parse_final_grid_links(
        response.text,
        event_name=event_name,
        year=year,
        documents_page_url=page_url,
    )
    if not documents:
        raise FinalGridNotReadyError(
            f"{year} {event_name} için FIA Final Starting Grid henüz yayımlanmadı; "
            "tahmin oluşturulamaz."
        )
    return documents[-1]


def download_final_grid_pdf(
    document: FiaGridDocument,
    *,
    timeout: float = 60.0,
    get: Callable[..., requests.Response] = requests.get,
) -> bytes:
    response = get(
        document.pdf_url,
        timeout=timeout,
        headers={"User-Agent": "f1-prediction-v2/1.0"},
    )
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        raise GridDataError("FIA final-grid bağlantısı geçerli bir PDF döndürmedi")
    return content


def _normal_text(value: object) -> str:
    return " ".join(str(value).replace("\xa0", " ").split()).upper()


def _empty_penalty_notes() -> pd.DataFrame:
    return pd.DataFrame(columns=["CarNumber", *PENALTY_COLUMNS])


def parse_penalty_notes(text: str) -> pd.DataFrame:
    """Final-grid metnindeki araç bazlı FIA ceza notlarını sınıflandırır."""
    lines = [" ".join(line.replace("\xa0", " ").split()) for line in text.splitlines()]
    penalty_index = next(
        (index for index, line in enumerate(lines) if "PENALTIES" in line.upper()),
        None,
    )
    if penalty_index is None:
        return _empty_penalty_notes()

    raw_records: list[tuple[list[str], str]] = []
    current_cars: list[str] = []
    current_text: list[str] = []

    def flush() -> None:
        if current_cars and current_text:
            raw_records.append((current_cars.copy(), " ".join(current_text)))

    note_pattern = re.compile(
        r"^CARS?\s+(?P<cars>\d{1,2}(?:\s*(?:&|,|AND)\s*\d{1,2})*)\s*-\s*(?P<reason>.*)$",
        flags=re.IGNORECASE,
    )
    for line in lines[penalty_index + 1 :]:
        if not line:
            continue
        if "THE STEWARDS" in line.upper() or line.startswith("©"):
            flush()
            break
        match = note_pattern.match(line)
        if match is not None:
            flush()
            current_cars = re.findall(r"\d{1,2}", match["cars"])
            current_text = [match["reason"]]
        elif current_cars:
            continuation = _normal_text(line)
            if re.search(
                r"PENALTY|PIT LANE|POWER UNIT|PU ELEMENT|PARC FERM|STEWARDS|DOCUMENT|"
                r"IMPEDE|COLLISION|TECHNICAL|GEARBOX|PERMITTED TO START|ADDITIONAL",
                continuation,
            ):
                current_text.append(line)
            else:
                flush()
                break
    else:
        flush()

    if not raw_records:
        if re.search(r"\bCARS?\s+\d{1,2}\b", " ".join(lines[penalty_index + 1 :]), re.I):
            raise GridDataError("FIA ceza notları araç bazında güvenle okunamadı")
        return _empty_penalty_notes()

    records: list[dict[str, object]] = []
    for cars, reason in raw_records:
        normalized = _normal_text(reason)
        position_match = re.search(
            r"\b(\d{1,2})\s+(?:PLACE|POSITION)\s+GRID\s+PENALTY\b",
            normalized,
        )
        positions = int(position_match.group(1)) if position_match else pd.NA
        power_unit = bool(re.search(r"\bPOWER UNIT\b|\bPU ELEMENT", normalized))
        sporting = bool(
            re.search(
                r"IMPED|COLLISION|YELLOW FLAG|DRIVING|REPRIMAND|UNSAFE|START PROCEDURE",
                normalized,
            )
        )
        parc_ferme = bool(re.search(r"PARC FERM", normalized))
        technical = bool(
            re.search(r"GEARBOX|SUSPENSION|TECHNICAL|SET[ -]?UP|CHASSIS", normalized)
        )
        pit_lane = "PIT LANE" in normalized
        permission = "PERMITTED TO START" in normalized
        applied = bool(
            "PENALTY" in normalized
            or pit_lane
            or "BACK OF THE GRID" in normalized
        )
        category_flags = {
            "POWER_UNIT": power_unit,
            "SPORTING": sporting,
            "PARC_FERME": parc_ferme,
            "TECHNICAL": technical,
            "PIT_LANE": pit_lane,
        }
        categories = [name for name, active in category_flags.items() if active]
        if len(categories) > 1:
            category = "MULTIPLE"
        elif categories:
            category = categories[0]
        elif permission:
            category = "PERMISSION_TO_START"
        elif applied:
            category = "OTHER"
        else:
            category = "NOTE_ONLY"
        other = bool(applied and not categories)
        for car_number in cars:
            records.append(
                {
                    "CarNumber": str(int(car_number)),
                    "GridPenaltyNotePresent": True,
                    "GridPenaltyApplied": applied,
                    "GridPenaltyPositions": positions,
                    "GridPenaltyPowerUnit": power_unit,
                    "GridPenaltySporting": sporting,
                    "GridPenaltyParcFerme": parc_ferme,
                    "GridPenaltyTechnical": technical,
                    "GridPenaltyPitLane": pit_lane,
                    "GridPermissionToStart": permission,
                    "GridPenaltyOther": other,
                    "GridPenaltyReason": category,
                    "GridPenaltyText": reason,
                }
            )

    frame = pd.DataFrame(records)
    if frame["CarNumber"].duplicated().any():
        rows: list[dict[str, object]] = []
        for car_number, notes in frame.groupby("CarNumber", sort=False):
            positions = pd.to_numeric(notes["GridPenaltyPositions"], errors="coerce")
            flags = {
                column: bool(notes[column].astype(bool).any())
                for column in PENALTY_COLUMNS
                if column.startswith("GridPenalty")
                and column not in {"GridPenaltyPositions", "GridPenaltyReason", "GridPenaltyText"}
            }
            categories = sorted(
                set(notes["GridPenaltyReason"].astype(str)) - {"NOTE_ONLY"}
            )
            rows.append(
                {
                    "CarNumber": car_number,
                    **flags,
                    "GridPenaltyPositions": (
                        positions.sum(min_count=1) if positions.notna().any() else pd.NA
                    ),
                    "GridPermissionToStart": bool(
                        notes["GridPermissionToStart"].astype(bool).any()
                    ),
                    "GridPenaltyReason": (
                        "MULTIPLE" if len(categories) > 1 else (categories[0] if categories else "NOTE_ONLY")
                    ),
                    "GridPenaltyText": " | ".join(notes["GridPenaltyText"].astype(str)),
                }
            )
        frame = pd.DataFrame(rows)
    return frame[["CarNumber", *PENALTY_COLUMNS]]


def _integer_word(word: dict[str, object]) -> int | None:
    text = str(word.get("text", "")).strip()
    return int(text) if re.fullmatch(r"\d{1,2}", text) else None


def _word_center_x(word: dict[str, object]) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2


def _in_any_band(value: float, bands: Iterable[tuple[float, float]]) -> bool:
    return any(low <= value <= high for low, high in bands)


def parse_grid_page_words(
    words: list[dict[str, object]], page_width: float
) -> pd.DataFrame:
    """İki sütunlu FIA grid sayfasını kelime koordinatlarından çözer."""
    if page_width <= 0:
        raise GridDataError("PDF sayfa genişliği geçersiz")

    normalized = [dict(word, normalized=_normal_text(word.get("text", ""))) for word in words]
    pit_tops = [
        float(word["top"])
        for word in normalized
        if word["normalized"] == "DRIVERS"
    ]
    penalty_tops = [
        float(word["top"])
        for word in normalized
        if word["normalized"] == "PENALTIES"
    ]
    pit_top = min(pit_tops) if pit_tops else None
    penalty_top = min(penalty_tops) if penalty_tops else float("inf")
    grid_bottom = pit_top if pit_top is not None else penalty_top

    position_bands = ((0.075, 0.115), (0.51, 0.56))
    car_bands = ((0.11, 0.155), (0.55, 0.60))
    positions: list[dict[str, object]] = []

    for word in normalized:
        position = _integer_word(word)
        if position is None or position < 1 or position > 30:
            continue
        top = float(word["top"])
        center = _word_center_x(word) / page_width
        if top >= grid_bottom or not _in_any_band(center, position_bands):
            continue

        candidates = []
        for other in normalized:
            car_number = _integer_word(other)
            if car_number is None:
                continue
            other_center = _word_center_x(other) / page_width
            horizontal_gap = float(other["x0"]) - float(word["x1"])
            if (
                _in_any_band(other_center, car_bands)
                and 4 <= horizontal_gap <= 30
                and abs(float(other["top"]) - top) <= 4.5
            ):
                candidates.append((abs(float(other["top"]) - top), car_number))
        if len(candidates) != 1:
            continue
        positions.append(
            {
                "CarNumber": str(candidates[0][1]),
                "GridPosition": position,
                "StartsFromPitLane": False,
                "SourceTop": top,
            }
        )

    pit_rows: list[dict[str, object]] = []
    if pit_top is not None:
        for word in normalized:
            car_number = _integer_word(word)
            if car_number is None:
                continue
            top = float(word["top"])
            center = _word_center_x(word) / page_width
            if (
                pit_top < top < penalty_top
                and _in_any_band(center, car_bands)
            ):
                pit_rows.append(
                    {
                        "CarNumber": str(car_number),
                        "GridPosition": pd.NA,
                        "StartsFromPitLane": True,
                        "SourceTop": top,
                    }
                )

    frame = pd.DataFrame(positions + pit_rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["CarNumber", "GridPosition", "StartsFromPitLane"]
        )
    frame = frame.drop_duplicates(["CarNumber", "StartsFromPitLane"])
    if frame["CarNumber"].duplicated().any():
        raise GridDataError("PDF gridinde aynı araç birden fazla kez bulundu")

    on_grid = frame[~frame["StartsFromPitLane"]].sort_values("GridPosition")
    expected_positions = list(range(1, len(on_grid) + 1))
    if on_grid["GridPosition"].tolist() != expected_positions:
        raise GridDataError(
            "PDF grid pozisyonları kesintisiz okunamadı; otomatik tahmin durduruldu"
        )
    return frame.sort_values(
        ["StartsFromPitLane", "GridPosition", "SourceTop"], na_position="last"
    ).drop(columns="SourceTop").reset_index(drop=True)


def parse_final_grid_pdf(pdf_content: bytes, document: FiaGridDocument) -> pd.DataFrame:
    """FIA PDF'sini doğrular ve araç numarası bazında final gridi çıkarır."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "FIA PDF okuyucusu için 'pdfplumber' kurulmalı."
        ) from exc

    with pdfplumber.open(BytesIO(pdf_content)) as pdf:
        all_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        normalized_text = _normal_text(all_text)
        if "FINAL STARTING GRID" not in normalized_text:
            raise GridDataError("PDF, FIA Final Starting Grid belgesi değil")
        if str(document.year) not in normalized_text:
            raise GridDataError("PDF sezonu istenen sezonla uyuşmuyor")
        if not re.search(
            rf"\b(?:DOC(?:UMENT)?\s*)?{re.escape(document.document_number)}\b",
            normalized_text,
        ):
            raise GridDataError("PDF belge numarası FIA bağlantısıyla uyuşmuyor")

        parsed_pages = [
            parse_grid_page_words(page.extract_words(), float(page.width))
            for page in pdf.pages
        ]
        grid = max(parsed_pages, key=len)
        if len(grid) < 2:
            raise GridDataError("FIA PDF'sinden final grid okunamadı")
        penalty_notes = parse_penalty_notes(all_text)
        grid = grid.merge(penalty_notes, on="CarNumber", how="left")
        boolean_columns = [
            column
            for column in PENALTY_COLUMNS
            if column.startswith("GridPenalty")
            and column not in {"GridPenaltyPositions", "GridPenaltyReason", "GridPenaltyText"}
        ] + ["GridPermissionToStart"]
        for column in boolean_columns:
            grid[column] = grid[column].astype("boolean").fillna(False).astype(bool)
        grid["GridPenaltyReason"] = grid["GridPenaltyReason"].fillna("NONE")
        grid["GridPenaltyText"] = grid["GridPenaltyText"].fillna("")
        grid["GridPenaltyPositions"] = pd.to_numeric(
            grid["GridPenaltyPositions"], errors="coerce"
        ).astype("Int64")
        return grid


def attach_driver_ids(
    grid: pd.DataFrame, qualifying_results: pd.DataFrame
) -> pd.DataFrame:
    """FIA araç numaralarını FastF1'in kalıcı pilot kimlikleriyle eşleştirir."""
    required = {"DriverId", "DriverNumber"}
    missing = sorted(required.difference(qualifying_results.columns))
    if missing:
        raise GridDataError(
            f"Qualifying pilot eşleştirmesi eksik: {', '.join(missing)}"
        )
    mapping = qualifying_results[["DriverId", "DriverNumber"]].copy()
    mapping["CarNumber"] = (
        pd.to_numeric(mapping["DriverNumber"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )
    if mapping["CarNumber"].isna().any() or mapping["DriverId"].isna().any():
        raise GridDataError("Qualifying pilot/araç kimliği boş olamaz")
    if mapping["CarNumber"].duplicated().any():
        raise GridDataError("Qualifying verisinde tekrarlanan araç numarası var")

    merged = grid.merge(mapping[["CarNumber", "DriverId"]], on="CarNumber", how="left")
    if merged["DriverId"].isna().any():
        unknown = sorted(merged.loc[merged["DriverId"].isna(), "CarNumber"])
        raise GridDataError(f"FIA gridindeki araçlar pilotlarla eşleşmedi: {unknown}")
    return merged


def _store_pdf_snapshot(
    content: bytes,
    document: FiaGridDocument,
    raw_data_dir: Path,
    sha256: str,
) -> Path:
    event_slug = re.sub(r"[^a-z0-9]+", "_", document.event_name.lower()).strip("_")
    directory = raw_data_dir / "fia" / str(document.year) / event_slug
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"final_starting_grid_doc_{document.document_number}_{sha256[:12]}.pdf"
    )
    if not path.exists():
        path.write_bytes(content)
    return path


def fetch_fia_final_grid(
    event_name: str,
    year: int,
    qualifying_results: pd.DataFrame,
    *,
    raw_data_dir: Path | None = None,
    timeout: float = 60.0,
) -> FiaGridSnapshot:
    """FIA final gridini bulur, PDF'yi doğrular ve model girdisine dönüştürür."""
    document = discover_final_grid_document(event_name, year, timeout=timeout)
    pdf_content = download_final_grid_pdf(document, timeout=timeout)
    sha256 = hashlib.sha256(pdf_content).hexdigest()
    raw_grid = parse_final_grid_pdf(pdf_content, document)
    mapped = attach_driver_ids(raw_grid, qualifying_results)
    metadata = GridDocumentMetadata(
        document_type=FINAL_DOCUMENT_TYPE,
        source=document.pdf_url,
        published_at=document.published_at,
        document_number=document.document_number,
    )
    features = build_final_grid_features(
        mapped,
        metadata,
        expected_driver_ids=qualifying_results["DriverId"].tolist(),
    )
    pdf_path = (
        _store_pdf_snapshot(pdf_content, document, raw_data_dir, sha256)
        if raw_data_dir is not None
        else None
    )
    return FiaGridSnapshot(
        features=features,
        document=document,
        pdf_sha256=sha256,
        pdf_path=pdf_path,
    )
