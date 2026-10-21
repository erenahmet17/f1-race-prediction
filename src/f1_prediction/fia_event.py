"""FastF1 ve FIA etkinlik adları arasındaki doğrulanmış eşlemeler."""

from __future__ import annotations


FIA_EVENT_NAME_ALIASES = {
    "Barcelona Grand Prix": "Barcelona-Catalunya Grand Prix",
}


def fia_event_page_name(event_name: str) -> str:
    """FIA belge sayfasında kullanılan etkinlik adını döndürür."""
    normalized = str(event_name).strip()
    return FIA_EVENT_NAME_ALIASES.get(normalized, normalized)
