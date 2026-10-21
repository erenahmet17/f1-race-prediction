from f1_prediction.fia_event import fia_event_page_name


def test_barcelona_uses_fia_barcelona_catalunya_page() -> None:
    assert (
        fia_event_page_name("Barcelona Grand Prix")
        == "Barcelona-Catalunya Grand Prix"
    )


def test_unaliased_event_name_is_unchanged() -> None:
    assert fia_event_page_name("Italian Grand Prix") == "Italian Grand Prix"
