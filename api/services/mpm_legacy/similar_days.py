"""Matching de días similares (weekday + TipoDia + temp Baq).

Réplica del bloque ``for i = 1:3 / switch i`` del fragmento MATLAB: el
relleno de ``AssembledData(3,:)`` NO se hace día por día con la temperatura
de cada día futuro, sino con un barrido por tipo de día que escribe el mismo
valor en todas las columnas que comparten ``(weekday, TipoDia)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isnan, nan
from statistics import mean
from typing import AbstractSet, Iterable, Sequence

from .day_types import TipoDia, day_type, matlab_weekday


DESV_THRESHOLD = 7e5
DESV_NO_MATCH = 8e5

# MATLAB case 1: DiasFaltantes = 2:6 (lunes..viernes en weekday de MATLAB).
ORDINARY_WEEKDAYS = (2, 3, 4, 5, 6)
# MATLAB case 2: DiasFaltantes = 7.
SATURDAY_WEEKDAY = 7


@dataclass(frozen=True)
class HistoricDay:
    day: date
    demand: float
    temp_baq: float
    tipo: TipoDia


def round_temp(value: float) -> float:
    """``round`` con semántica MATLAB: NaN se propaga en vez de lanzar.

    Un histórico sin temperatura (p. ej. días del mes anterior que solo
    traen demanda) queda con NaN y, como ``nan != nan``, jamás produce un
    match por temperatura: equivale a excluirlo del pool de ``RefDays``.
    """

    temp = float(value)
    if isnan(temp):
        return nan
    return float(round(temp))


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _mean_or_nan(values: Sequence[float]) -> float:
    """``mean`` de MATLAB: promedio de vector vacío es NaN."""

    return float(mean(values)) if values else nan


def _first_temp_match(
    pool: Iterable[HistoricDay],
    *,
    weekday: int,
    tipo: TipoDia | int,
    target_temp: float,
) -> HistoricDay | None:
    """MATLAB: primera fila de ``RefDays`` igual a ``PairDays(ii,:)``."""

    if isnan(target_temp):
        return None
    for item in pool:
        if matlab_weekday(item.day) != weekday:
            continue
        if int(item.tipo) != int(tipo):
            continue
        if round_temp(item.temp_baq) == target_temp:
            return item
    return None


def _resolve_demand(
    candidate: HistoricDay | None,
    actual_pool: Sequence[HistoricDay],
    fallback_pool: Sequence[HistoricDay],
) -> float:
    """Criterio ``Desv > 7e5`` del fragmento.

    Sin día similar ``Desv = 8e5`` y se cae a la media. Con día similar pero
    sin mes actual, ``mean([])`` es NaN y ``NaN > 7e5`` es falso, así que
    MATLAB conserva el día similar.
    """

    actual_demands = [item.demand for item in actual_pool]
    if candidate is None:
        desv = DESV_NO_MATCH
    else:
        desv = _mean_or_nan(actual_demands) - candidate.demand

    if candidate is not None and not (desv > DESV_THRESHOLD):
        return float(candidate.demand)
    if actual_demands:
        return _mean_or_zero(actual_demands)
    return _mean_or_zero([item.demand for item in fallback_pool])


def select_similar_demand(
    *,
    weekday: int,
    tipo: TipoDia,
    temp_baq: float,
    historic: Sequence[HistoricDay],
    current_month: Sequence[HistoricDay],
) -> float:
    """Resuelve un único ``(weekday, TipoDia, temp)`` contra el histórico."""

    pool = [item for item in historic if int(item.tipo) == int(tipo)]
    actual = [item for item in current_month if int(item.tipo) == int(tipo)]
    candidate = _first_temp_match(
        pool, weekday=weekday, tipo=tipo, target_temp=round_temp(temp_baq)
    )
    return _resolve_demand(candidate, actual, pool)


def _anchor_index(
    weekdays: Sequence[int],
    tipos: Sequence[TipoDia],
    weekday: int,
    tipo: TipoDia | None,
) -> int | None:
    """MATLAB: ``find(weekday(Fecha) == d [& TipoDia == 0],1,'first')``."""

    for idx, value in enumerate(weekdays):
        if value != weekday:
            continue
        if tipo is not None and tipos[idx] != tipo:
            continue
        return idx
    return None


def _columns_for(
    weekdays: Sequence[int],
    tipos: Sequence[TipoDia],
    weekday: int,
    tipo: TipoDia | int,
) -> list[int]:
    """MATLAB: ``ismember(AssembledData(1:2,:)', PairDays(ii,1:2),'rows')``."""

    return [
        idx
        for idx in range(len(weekdays))
        if weekdays[idx] == weekday and int(tipos[idx]) == int(tipo)
    ]


def _sweep_ordinary(
    values: list[float],
    weekdays: Sequence[int],
    tipos: Sequence[TipoDia],
    temps: Sequence[float],
    historic: Sequence[HistoricDay],
    period: AbstractSet[date],
) -> None:
    pool = [item for item in historic if item.tipo == TipoDia.ORDINARIO]
    actual = [item for item in pool if item.day in period]

    for weekday in ORDINARY_WEEKDAYS:
        anchor = _anchor_index(weekdays, tipos, weekday, TipoDia.ORDINARIO)
        if anchor is None:
            continue
        tipo = tipos[anchor]
        columns = _columns_for(weekdays, tipos, weekday, tipo)
        if not columns:
            continue
        candidate = _first_temp_match(
            pool, weekday=weekday, tipo=tipo, target_temp=temps[anchor]
        )
        value = _resolve_demand(candidate, actual, pool)
        for idx in columns:
            values[idx] = value


def _sweep_saturday(
    values: list[float],
    weekdays: Sequence[int],
    tipos: Sequence[TipoDia],
    temps: Sequence[float],
    historic: Sequence[HistoricDay],
    period: AbstractSet[date],
) -> None:
    pool = [item for item in historic if item.tipo == TipoDia.SABADO]
    # MATLAB: si no hay sábados del mes actual, DemandaActual = Demanda.
    actual = [item for item in pool if item.day in period] or list(pool)

    # El anclaje del case 2 no exige TipoDia: si el primer sábado del mes es
    # festivo, PairDays lleva tipo -1 y no habrá match en RefDays.
    anchor = _anchor_index(weekdays, tipos, SATURDAY_WEEKDAY, None)
    if anchor is None:
        return
    tipo = tipos[anchor]
    columns = _columns_for(weekdays, tipos, SATURDAY_WEEKDAY, tipo)
    if not columns:
        return
    candidate = _first_temp_match(
        pool, weekday=SATURDAY_WEEKDAY, tipo=tipo, target_temp=temps[anchor]
    )
    value = _resolve_demand(candidate, actual, pool)
    for idx in columns:
        values[idx] = value


def _sweep_sunday_holiday(
    values: list[float],
    tipos: Sequence[TipoDia],
    historic: Sequence[HistoricDay],
) -> None:
    """MATLAB case 3: media del pool domingo/festivo, sin temperatura."""

    columns = [
        idx for idx, tipo in enumerate(tipos) if tipo == TipoDia.DOMINGO_FESTIVO
    ]
    if not columns:
        return
    pool = [
        item.demand for item in historic if item.tipo == TipoDia.DOMINGO_FESTIVO
    ]
    value = _mean_or_zero(pool)
    for idx in columns:
        values[idx] = value


def fill_base_demands(
    dates: Sequence[date],
    demand_by_date: dict[date, float],
    holidays: AbstractSet[date],
    temps_baq: Sequence[float],
    historic: Sequence[HistoricDay],
) -> list[float]:
    """Construye la fila f3 (``AssembledData(3,:)``) del fragmento.

    Barrido por tipo (case 1/2/3) y, al final, la demanda real observada pisa
    las columnas correspondientes tal como hace
    ``AssembledData(3,1:size(DemandaActual,1)) = DemandaActual(:,2)'``.
    """

    if len(temps_baq) != len(dates):
        raise ValueError("temps_baq no está alineado con dates")

    weekdays = [matlab_weekday(day) for day in dates]
    tipos = [day_type(day, holidays) for day in dates]
    temps = [round_temp(value) for value in temps_baq]
    period = set(dates)

    values = [0.0] * len(dates)
    _sweep_ordinary(values, weekdays, tipos, temps, historic, period)
    _sweep_saturday(values, weekdays, tipos, temps, historic, period)
    _sweep_sunday_holiday(values, tipos, historic)

    for idx, day in enumerate(dates):
        if day in demand_by_date:
            values[idx] = float(demand_by_date[day])
    return values
