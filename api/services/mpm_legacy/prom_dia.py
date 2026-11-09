"""Medianas históricas por tipo de día."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import nan
from statistics import median
from typing import AbstractSet, Iterable

from .day_types import TipoDia, day_type


DemandRecord = tuple[date, float]


@dataclass(frozen=True)
class PromDia:
    ordinario: float
    sabado: float
    domingo_festivo: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.ordinario, self.sabado, self.domingo_festivo)

    def for_type(self, tipo: TipoDia | int) -> float:
        return {
            TipoDia.ORDINARIO: self.ordinario,
            TipoDia.SABADO: self.sabado,
            TipoDia.DOMINGO_FESTIVO: self.domingo_festivo,
        }[TipoDia(tipo)]


def _median_or_nan(values: list[float]) -> float:
    return float(median(values)) if values else nan


def calculate_prom_dia(
    records: Iterable[DemandRecord], holidays: AbstractSet[date]
) -> PromDia:
    """Calcula medianas, no medias, como el fragmento MATLAB."""

    grouped: dict[TipoDia, list[float]] = {tipo: [] for tipo in TipoDia}
    for day, demand in records:
        grouped[day_type(day, holidays)].append(float(demand))
    return PromDia(
        ordinario=_median_or_nan(grouped[TipoDia.ORDINARIO]),
        sabado=_median_or_nan(grouped[TipoDia.SABADO]),
        domingo_festivo=_median_or_nan(grouped[TipoDia.DOMINGO_FESTIVO]),
    )
