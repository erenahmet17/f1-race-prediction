"""OpenF1 tur verisini projenin kullandığı FastF1 tur biçimine çevirir."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import requests


OPENF1_BASE_URL = "https://api.openf1.org/v1"
LAP_COLUMNS = (
    "DriverNumber",
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "Stint",
    "PitOutTime",
    "PitInTime",
    "IsAccurate",
    "Deleted",
    "TrackStatus",
)
REQUIRED_LAP_FIELDS = {
    "driver_number",
    "lap_number",
    "date_start",
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "is_pit_out_lap",
}
_SESSION_DATE_TOLERANCE = pd.Timedelta(hours=3)
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_WAIT_SECONDS = 20.0
_ACCURATE_TRACK_STATUSES = {"1", "2", "12"}
_SECTOR_SUM_TOLERANCE_SECONDS = 0.003


class OpenF1Error(RuntimeError):
    """OpenF1 verisi alınamadığında veya beklenen yapıda olmadığında oluşur."""


@dataclass(frozen=True)
class OpenF1Session:
    session_key: int
    circuit_key: int
    circuit_name: str


def _fetch(
    endpoint: str,
    *,
    get: Callable[..., requests.Response],
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 60.0,
    **params: object,
) -> pd.DataFrame:
    url = f"{OPENF1_BASE_URL}/{endpoint}"
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            response = get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise OpenF1Error(f"OpenF1 {endpoint} isteği başarısız: {exc}") from exc
        if response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
            break
        sleep(_RATE_LIMIT_WAIT_SECONDS)
    # OpenF1 sonuç bulamadığında boş liste yerine 404 döndürür.
    if response.status_code == 404:
        return pd.DataFrame()
    if response.status_code != 200:
        raise OpenF1Error(f"OpenF1 {endpoint} isteği HTTP {response.status_code} döndürdü")
    try:
        return pd.DataFrame(response.json())
    except ValueError as exc:
        raise OpenF1Error(f"OpenF1 {endpoint} yanıtı okunamadı") from exc


def find_openf1_session(
    season: int,
    session_name: str,
    start_utc: pd.Timestamp | None,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> OpenF1Session:
    """Seansı takvim tarihine göre bulur; ertelenen yarışların eski kaydını eler."""
    if start_utc is None or pd.isna(start_utc):
        raise OpenF1Error(f"{session_name} seansının başlangıç saati bilinmiyor")
    sessions = _fetch("sessions", get=get, year=season, session_name=session_name)
    if sessions.empty:
        raise OpenF1Error(f"OpenF1'de {season} {session_name} seansı yok")
    starts = pd.to_datetime(sessions["date_start"], utc=True, format="ISO8601")
    gaps = (starts - start_utc).abs()
    nearest = gaps.idxmin()
    if gaps[nearest] > _SESSION_DATE_TOLERANCE:
        raise OpenF1Error(
            f"OpenF1'de {start_utc} tarihli {session_name} seansı bulunamadı"
        )
    session = sessions.loc[nearest]
    return OpenF1Session(
        session_key=int(session["session_key"]),
        circuit_key=int(session["circuit_key"]),
        circuit_name=str(session["circuit_short_name"]),
    )


def _laps_by_stint(stints: pd.DataFrame) -> pd.DataFrame:
    columns = ["driver_number", "lap_number", "Compound", "TyreLife", "Stint"]
    required = {"driver_number", "stint_number", "lap_start", "lap_end"}
    if stints.empty or not required.issubset(stints.columns):
        return pd.DataFrame(columns=columns)
    rows = (
        stints.dropna(subset=sorted(required))
        .sort_values(["driver_number", "stint_number"])
        .copy()
    )
    rows["lap_number"] = [
        list(range(int(start), int(end) + 1))
        for start, end in zip(rows["lap_start"], rows["lap_end"])
    ]
    laps = rows.explode("lap_number").dropna(subset="lap_number")
    laps["lap_number"] = laps["lap_number"].astype(int)
    # FastF1 bu yaşı turun sonundaki değerle verir; OpenF1 stint başındakini verir.
    laps["TyreLife"] = (
        pd.to_numeric(laps["tyre_age_at_start"], errors="coerce")
        + (laps["lap_number"] - laps["lap_start"])
        + 1
    )
    laps = laps.rename(columns={"compound": "Compound", "stint_number": "Stint"})
    return laps[columns].drop_duplicates(["driver_number", "lap_number"])


def _track_status_intervals(
    race_control: pd.DataFrame, session_end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Yarış kontrol mesajlarından FastF1'in pist durumu kodlarını türetir."""
    if race_control.empty or "date" not in race_control.columns:
        return []
    messages = race_control.assign(
        date=pd.to_datetime(race_control["date"], utc=True, format="ISO8601")
    ).sort_values("date", kind="stable")

    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    yellow_since: dict[object, pd.Timestamp] = {}
    neutralized_since: tuple[pd.Timestamp, str] | None = None
    red_since: pd.Timestamp | None = None
    for _, message in messages.iterrows():
        moment = message["date"]
        category = message.get("category")
        flag = message.get("flag")
        scope = message.get("scope")
        text = str(message.get("message", "")).upper()
        if category == "SafetyCar":
            if "DEPLOYED" in text and neutralized_since is None:
                code = "6" if "VSC" in text or "VIRTUAL" in text else "4"
                neutralized_since = (moment, code)
        elif category != "Flag":
            continue
        elif flag in ("YELLOW", "DOUBLE YELLOW"):
            key = message.get("sector") if scope == "Sector" else "track"
            yellow_since.setdefault(key, moment)
        elif scope == "Sector" and flag == "CLEAR":
            started = yellow_since.pop(message.get("sector"), None)
            if started is not None:
                intervals.append((started, moment, "2"))
        elif scope == "Track" and flag == "RED":
            red_since = moment if red_since is None else red_since
        elif scope == "Track" and flag in ("CLEAR", "GREEN"):
            intervals.extend((started, moment, "2") for started in yellow_since.values())
            yellow_since = {}
            if neutralized_since is not None:
                intervals.append((neutralized_since[0], moment, neutralized_since[1]))
                neutralized_since = None
            if red_since is not None:
                intervals.append((red_since, moment, "5"))
                red_since = None

    intervals.extend((started, session_end, "2") for started in yellow_since.values())
    if neutralized_since is not None:
        intervals.append((neutralized_since[0], session_end, neutralized_since[1]))
    if red_since is not None:
        intervals.append((red_since, session_end, "5"))
    return intervals


def _lap_track_status(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]],
) -> str:
    if pd.isna(start) or pd.isna(end):
        return "1"
    overlapping = [item for item in intervals if item[0] < end and item[1] > start]
    codes = {code for _, _, code in overlapping}
    if not any(begin <= start and finish >= end for begin, finish, _ in overlapping):
        codes.add("1")
    return "".join(sorted(codes))


def build_fastf1_style_laps(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    drivers: pd.DataFrame,
    race_control: pd.DataFrame,
) -> pd.DataFrame:
    """OpenF1 tablolarını FastF1 Laps sütunlarına ve doğruluk kurallarına çevirir.

    FastF1'in tur doğruluğu kuralları uygulanır; yalnız tur bitiş saatine dayanan
    kontrol atlanır, çünkü OpenF1'in ``date_start`` alanı bu hassasiyette değildir.
    Silinen turlar işaretlenmez: geçmiş eğitim satırları da yarış kontrol mesajları
    yüklenmeden üretildi.
    """
    missing = sorted(REQUIRED_LAP_FIELDS.difference(laps.columns))
    if missing:
        raise OpenF1Error(f"OpenF1 tur verisinde eksik alanlar var: {', '.join(missing)}")

    work = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    work["date_start"] = pd.to_datetime(
        work["date_start"], utc=True, format="ISO8601", errors="coerce"
    )
    duration = pd.to_numeric(work["lap_duration"], errors="coerce")
    sectors = work[
        ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
    ].apply(pd.to_numeric, errors="coerce")
    pit_out = work["is_pit_out_lap"].astype("boolean").fillna(False).astype(bool)

    by_driver = work.groupby("driver_number")
    next_start = by_driver["date_start"].shift(-1)
    follows_pit_out = pit_out.groupby(work["driver_number"]).shift(-1, fill_value=False)
    # FastF1 pilotun seanstaki son turunu da pite giriş sayar.
    last_lap = by_driver.cumcount(ascending=False).eq(0)
    pit_in = follows_pit_out.astype(bool) | last_lap

    lap_end = (work["date_start"] + pd.to_timedelta(duration, unit="s")).fillna(next_start)
    known_times = pd.concat([work["date_start"], lap_end]).dropna()
    session_end = known_times.max() if not known_times.empty else pd.Timestamp.now(tz="UTC")
    intervals = _track_status_intervals(race_control, session_end)
    status = pd.Series(
        [
            _lap_track_status(start, end, intervals)
            for start, end in zip(work["date_start"], lap_end)
        ],
        index=work.index,
    )
    previous_status = status.groupby(work["driver_number"]).shift(1)

    sector_sum_ok = (
        sectors.notna().all(axis=1)
        & duration.notna()
        & (
            (sectors.sum(axis=1) - duration).abs()
            <= _SECTOR_SUM_TOLERANCE_SECONDS
        )
    )
    work["IsAccurate"] = (
        ~pit_out
        & ~pit_in
        & status.isin(_ACCURATE_TRACK_STATUSES)
        & sector_sum_ok
        & previous_status.ne("4")
    )
    work["LapTime"] = pd.to_timedelta(duration, unit="s")
    work["LapNumber"] = work["lap_number"]
    work["DriverNumber"] = work["driver_number"]
    # Tur süzme kuralları bu iki alanın yalnız dolu olup olmadığına bakar.
    work["PitOutTime"] = work["date_start"].where(pit_out)
    work["PitInTime"] = work["date_start"].where(pit_in)
    work["Deleted"] = False
    work["TrackStatus"] = status

    identity = pd.DataFrame(columns=["driver_number", "Driver", "Team"])
    if not drivers.empty and {"driver_number", "name_acronym"}.issubset(drivers.columns):
        identity = (
            drivers.reindex(columns=["driver_number", "name_acronym", "team_name"])
            .rename(columns={"name_acronym": "Driver", "team_name": "Team"})
            .drop_duplicates("driver_number")
        )
    work = work.merge(identity, on="driver_number", how="left")
    work = work.merge(
        _laps_by_stint(stints), on=["driver_number", "lap_number"], how="left"
    )
    return work.loc[:, list(LAP_COLUMNS)]


def fetch_openf1_laps(
    session_key: int,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> pd.DataFrame:
    """Bir seansın turlarını FastF1 biçiminde döndürür."""
    laps = _fetch("laps", get=get, session_key=session_key)
    if laps.empty:
        raise OpenF1Error(f"OpenF1'de {session_key} seansı için tur verisi yok")
    return build_fastf1_style_laps(
        laps,
        _fetch("stints", get=get, session_key=session_key),
        _fetch("drivers", get=get, session_key=session_key),
        _fetch("race_control", get=get, session_key=session_key),
    )
