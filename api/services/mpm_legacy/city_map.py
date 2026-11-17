"""Mapa provisional entre ciudades legacy y series climáticas de la app."""

from __future__ import annotations

from typing import Mapping, Sequence


REQUIRED_TEMP_KEYS = ("bolivar", "cordoba_sucre", "cesar")

# Paridad parcial: la app solo expone tres series. Baq y Car comparten Bolívar.
LEGACY_CITY_MAP = {
    "baq": "bolivar",
    "car": "bolivar",
    "mon": "cordoba_sucre",
    "sm": "cesar",
}


def map_legacy_temperatures(
    temperatures: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Expande tres series app a Baq/Car/Mon/SM."""

    return {
        legacy: [float(value) for value in temperatures[source]]
        for legacy, source in LEGACY_CITY_MAP.items()
    }
