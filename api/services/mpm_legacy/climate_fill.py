"""Relleno determinista de clima con perfiles sintéticos."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Mapping, Sequence

from .city_map import REQUIRED_TEMP_KEYS


SCENARIO_NAMES = {0: "caluroso", 1: "soleado", 2: "fresco"}

_BASE_TEMPERATURES = {
    0: {"bolivar": 29.25, "cordoba_sucre": 28.75, "cesar": 29.75},
    1: {"bolivar": 28.5, "cordoba_sucre": 28.0, "cesar": 29.0},
    2: {"bolivar": 27.75, "cordoba_sucre": 27.25, "cesar": 28.25},
}
_WEEKDAY_OFFSETS = (0.0, 0.2, 0.3, 0.1, -0.1, -0.2, -0.3)


def fill_temperature_gaps(
    dates: Sequence[date],
    temperatures: Mapping[str, Sequence[float | None]],
    scenario: int,
) -> dict[str, list[float]]:
    """Completa valores ausentes; paridad parcial al no existir ClimaRef.mat."""

    if scenario not in SCENARIO_NAMES:
        raise ValueError("scenario debe ser 0, 1 o 2")
    missing_keys = set(REQUIRED_TEMP_KEYS) - set(temperatures)
    if missing_keys:
        raise ValueError(f"Faltan series de temperatura: {sorted(missing_keys)}")

    result: dict[str, list[float]] = {}
    for city in REQUIRED_TEMP_KEYS:
        values = temperatures[city]
        if len(values) != len(dates):
            raise ValueError(f"La serie {city} no está alineada con dates")
        filled: list[float] = []
        for day, raw_value in zip(dates, values):
            value = float(raw_value) if raw_value is not None else float("nan")
            if not isfinite(value):
                value = (
                    _BASE_TEMPERATURES[scenario][city]
                    + _WEEKDAY_OFFSETS[day.weekday()]
                )
            filled.append(round(value, 1))
        result[city] = filled
    return result
