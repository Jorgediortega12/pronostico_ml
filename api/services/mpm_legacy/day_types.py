"""Clasificación de días compatible con ``weekday`` de MATLAB."""

from __future__ import annotations

from datetime import date
from enum import IntEnum
from typing import AbstractSet, Iterable


class TipoDia(IntEnum):
    """Códigos usados por el MPM legacy."""

    DOMINGO_FESTIVO = -1
    ORDINARIO = 0
    SABADO = 1


def matlab_weekday(day: date) -> int:
    """Retorna 1=domingo, 2=lunes, ..., 7=sábado."""

    return ((day.weekday() + 1) % 7) + 1


def day_type(day: date, holidays: AbstractSet[date]) -> TipoDia:
    """Clasifica un día; un festivo prevalece sobre el día de semana."""

    if day in holidays or matlab_weekday(day) == 1:
        return TipoDia.DOMINGO_FESTIVO
    if matlab_weekday(day) == 7:
        return TipoDia.SABADO
    return TipoDia.ORDINARIO


def classify_days(
    dates: Iterable[date], holidays: AbstractSet[date]
) -> list[TipoDia]:
    return [day_type(day, holidays) for day in dates]
