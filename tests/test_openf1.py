from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from f1_prediction.openf1 import (
    LAP_COLUMNS,
    OpenF1Cache,
    OpenF1Error,
    OpenF1Session,
    _fetch,
    build_fastf1_style_laps,
    find_openf1_session,
)


def lap(
    number: int,
    start: str,
    duration: float | None = 90.0,
    *,
    driver: int = 1,
    pit_out: bool = False,
    sectors: tuple[float, float, float] | None = (30.0, 30.0, 30.0),
) -> dict[str, object]:
    first, second, third = sectors if sectors is not None else (None, None, None)
    return {
        "driver_number": driver,
        "lap_number": number,
        "date_start": f"2026-10-02T{start}+00:00",
        "lap_duration": duration,
        "duration_sector_1": first,
        "duration_sector_2": second,
        "duration_sector_3": third,
        "is_pit_out_lap": pit_out,
    }


def laps_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            lap(1, "10:00:00", None, pit_out=True, sectors=None),
            lap(2, "10:02:00"),
            lap(3, "10:03:30"),
            lap(4, "10:05:00"),
            lap(5, "10:08:00", None, pit_out=True, sectors=None),
            lap(6, "10:10:00"),
        ]
    )


def stints_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_number": [1, 1],
            "stint_number": [1, 2],
            "lap_start": [1, 5],
            "lap_end": [4, 6],
            "compound": ["SOFT", "HARD"],
            "tyre_age_at_start": [0, 3],
        }
    )


def drivers_table() -> pd.DataFrame:
    return pd.DataFrame(
        {"driver_number": [1], "name_acronym": ["NOR"], "team_name": ["McLaren"]}
    )


def flag(moment: str, flag_name: str, scope: str, sector: float | None = None) -> dict:
    return {
        "date": f"2026-10-02T{moment}+00:00",
        "category": "Flag",
        "flag": flag_name,
        "scope": scope,
        "sector": sector,
        "message": f"{flag_name} {scope}",
    }


def test_laps_use_fastf1_columns_tyres_and_pit_rules() -> None:
    laps = build_fastf1_style_laps(
        laps_table(), stints_table(), drivers_table(), pd.DataFrame()
    ).set_index("LapNumber")

    assert set(laps.reset_index().columns) == set(LAP_COLUMNS)
    assert laps.loc[2, "Driver"] == "NOR"
    assert laps.loc[2, "Team"] == "McLaren"
    assert laps.loc[2, "LapTime"] == pd.Timedelta(seconds=90)
    assert laps.loc[3, "Compound"] == "SOFT"
    assert laps.loc[3, "Stint"] == 1
    # FastF1 lastik yaşını OpenF1'den bir fazla verir.
    assert laps.loc[1, "TyreLife"] == 1
    assert laps.loc[6, "TyreLife"] == 5
    assert pd.notna(laps.loc[1, "PitOutTime"])
    assert pd.isna(laps.loc[2, "PitOutTime"])
    # Sonraki turu pit çıkışı olan tur ve son tur pite giriş sayılır.
    assert pd.notna(laps.loc[4, "PitInTime"])
    assert pd.notna(laps.loc[6, "PitInTime"])
    assert pd.isna(laps.loc[3, "PitInTime"])
    assert laps["IsAccurate"].to_dict() == {
        1: False,
        2: True,
        3: True,
        4: False,
        5: False,
        6: False,
    }


def test_sector_sum_mismatch_and_missing_sector_are_not_accurate() -> None:
    table = pd.DataFrame(
        [
            lap(1, "10:00:00"),
            lap(2, "10:01:30", sectors=(30.0, 30.0, 30.5)),
            lap(3, "10:03:00", sectors=None),
            lap(4, "10:04:30"),
            lap(5, "10:06:00"),
        ]
    )
    accurate = build_fastf1_style_laps(
        table, stints_table(), drivers_table(), pd.DataFrame()
    ).set_index("LapNumber")["IsAccurate"]

    assert accurate[1]
    assert not accurate[2]
    assert not accurate[3]
    assert accurate[4]


def test_track_status_follows_sector_yellow_until_cleared() -> None:
    table = pd.DataFrame(
        [
            lap(1, "10:00:00"),
            lap(2, "10:01:30"),
            lap(3, "10:03:00"),
            lap(4, "10:04:30"),
            lap(5, "10:06:00"),
        ]
    )
    race_control = pd.DataFrame(
        [
            flag("10:02:00", "YELLOW", "Sector", 4.0),
            flag("10:02:10", "DOUBLE YELLOW", "Sector", 5.0),
            flag("10:02:30", "CLEAR", "Sector", 4.0),
            flag("10:03:20", "CLEAR", "Sector", 5.0),
        ]
    )

    laps = build_fastf1_style_laps(
        table, stints_table(), drivers_table(), race_control
    ).set_index("LapNumber")

    assert laps.loc[1, "TrackStatus"] == "1"
    assert laps.loc[2, "TrackStatus"] == "12"
    assert laps.loc[3, "TrackStatus"] == "12"
    assert laps.loc[4, "TrackStatus"] == "1"


def test_safety_car_stays_until_track_clear_and_taints_following_lap() -> None:
    table = pd.DataFrame(
        [
            lap(1, "10:00:00"),
            lap(2, "10:01:30"),
            lap(3, "10:03:00"),
            lap(4, "10:04:30"),
            lap(5, "10:06:00"),
        ]
    )
    race_control = pd.DataFrame(
        [
            {
                "date": "2026-10-02T10:01:00+00:00",
                "category": "SafetyCar",
                "flag": None,
                "scope": None,
                "sector": None,
                "message": "SAFETY CAR DEPLOYED",
            },
            flag("10:04:20", "CLEAR", "Track"),
        ]
    )

    laps = build_fastf1_style_laps(
        table, stints_table(), drivers_table(), race_control
    ).set_index("LapNumber")

    assert laps.loc[1, "TrackStatus"] == "14"
    assert laps.loc[2, "TrackStatus"] == "4"
    assert laps.loc[3, "TrackStatus"] == "14"
    # Güvenlik aracının bittiği tur doğru sayılmaz; pist açıldıktan sonraki tur sayılır.
    assert not laps.loc[3, "IsAccurate"]
    assert laps.loc[4, "TrackStatus"] == "1"
    assert laps.loc[4, "IsAccurate"]


def test_virtual_safety_car_is_coded_separately() -> None:
    race_control = pd.DataFrame(
        [
            {
                "date": "2026-10-02T10:02:00+00:00",
                "category": "SafetyCar",
                "flag": None,
                "scope": None,
                "sector": None,
                "message": "VSC DEPLOYED",
            },
            flag("10:03:10", "CLEAR", "Track"),
        ]
    )

    laps = build_fastf1_style_laps(
        laps_table(), stints_table(), drivers_table(), race_control
    ).set_index("LapNumber")

    assert laps.loc[2, "TrackStatus"] == "16"


def test_missing_lap_fields_are_rejected() -> None:
    with pytest.raises(OpenF1Error, match="eksik alanlar"):
        build_fastf1_style_laps(
            laps_table().drop(columns="lap_duration"),
            stints_table(),
            drivers_table(),
            pd.DataFrame(),
        )


def response(status: int, payload: object = None) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, json=lambda: payload)


def test_fetch_treats_404_as_empty_result() -> None:
    assert _fetch("laps", get=lambda *a, **k: response(404, {"detail": "none"})).empty


def test_fetch_retries_when_rate_limited() -> None:
    responses = [response(429), response(200, [{"lap_number": 1}])]
    waits: list[float] = []

    frame = _fetch(
        "laps",
        get=lambda *a, **k: responses.pop(0),
        sleep=waits.append,
    )

    assert frame["lap_number"].tolist() == [1]
    assert len(waits) == 1


@pytest.mark.parametrize("status", [401, 403, 500])
def test_fetch_reports_access_and_server_errors(status: int) -> None:
    with pytest.raises(OpenF1Error, match=f"HTTP {status}"):
        _fetch("laps", get=lambda *a, **k: response(status))


def test_fetch_wraps_connection_errors() -> None:
    def fail(*args, **kwargs):
        raise requests.ConnectionError("no route")

    with pytest.raises(OpenF1Error, match="başarısız"):
        _fetch("laps", get=fail)


def sessions_response(*args, **kwargs) -> SimpleNamespace:
    return response(
        200,
        [
            {
                "session_key": 1,
                "circuit_key": 63,
                "circuit_short_name": "Sakhir",
                "date_start": "2026-04-10T11:30:00+00:00",
            },
            {
                "session_key": 2,
                "circuit_key": 63,
                "circuit_short_name": "Sakhir",
                "date_start": "2026-10-02T08:00:00+00:00",
            },
        ],
    )


def test_session_is_matched_by_date_not_by_stale_duplicate() -> None:
    session = find_openf1_session(
        2026,
        "Practice 1",
        pd.Timestamp("2026-10-02T08:00:00Z"),
        get=sessions_response,
    )

    assert session.session_key == 2
    assert session.circuit_key == 63
    assert session.circuit_name == "Sakhir"


def test_session_outside_date_tolerance_is_rejected() -> None:
    with pytest.raises(OpenF1Error, match="bulunamadı"):
        find_openf1_session(
            2026,
            "Practice 1",
            pd.Timestamp("2026-07-01T08:00:00Z"),
            get=sessions_response,
        )


def test_session_without_known_start_is_rejected() -> None:
    with pytest.raises(OpenF1Error, match="başlangıç saati"):
        find_openf1_session(2026, "Practice 1", None, get=sessions_response)


def test_session_knows_whether_it_has_ended() -> None:
    past = pd.Timestamp("2000-01-01", tz="UTC")
    future = pd.Timestamp("2099-01-01", tz="UTC")

    assert OpenF1Session(1, 2, "A", past).has_ended
    assert not OpenF1Session(1, 2, "A", future).has_ended
    assert not OpenF1Session(1, 2, "A").has_ended


def test_session_end_is_read_from_openf1() -> None:
    def with_end(*args, **kwargs) -> SimpleNamespace:
        return response(
            200,
            [
                {
                    "session_key": 2,
                    "circuit_key": 63,
                    "circuit_short_name": "Sakhir",
                    "date_start": "2026-10-02T08:00:00+00:00",
                    "date_end": "2026-10-02T09:00:00+00:00",
                }
            ],
        )

    session = find_openf1_session(
        2026, "Practice 1", pd.Timestamp("2026-10-02T08:00:00Z"), get=with_end
    )

    assert session.end_utc == pd.Timestamp("2026-10-02T09:00:00Z")


def test_cached_laps_match_the_original_after_a_round_trip(tmp_path: Path) -> None:
    race_control = pd.DataFrame([flag("10:02:00", "YELLOW", "Sector", 4.0)])
    laps = build_fastf1_style_laps(
        laps_table(), stints_table(), drivers_table(), race_control
    )
    cache = OpenF1Cache.for_round(tmp_path, 2026, 19)

    assert cache.read_laps("Practice 1") is None
    path = cache.write_laps("Practice 1", laps)
    restored = cache.read_laps("Practice 1")

    assert path == tmp_path / "openf1" / "2026" / "round_19" / "practice_1_laps.csv"
    assert not path.with_suffix(".csv.tmp").exists()
    assert list(restored.columns) == list(laps.columns)
    pd.testing.assert_series_equal(restored["LapTime"], laps["LapTime"])
    assert restored["TrackStatus"].tolist() == laps["TrackStatus"].tolist()
    assert restored["TrackStatus"].map(type).eq(str).all()
    assert restored["IsAccurate"].tolist() == laps["IsAccurate"].tolist()
    assert restored["PitInTime"].isna().tolist() == laps["PitInTime"].isna().tolist()
    assert restored["TyreLife"].tolist() == laps["TyreLife"].tolist()


def test_cache_keeps_track_identity(tmp_path: Path) -> None:
    cache = OpenF1Cache.for_round(tmp_path, 2026, 19)

    assert cache.read_track() is None
    cache.write_track("65", "Mexico City")

    assert cache.read_track() == ("65", "Mexico City")
