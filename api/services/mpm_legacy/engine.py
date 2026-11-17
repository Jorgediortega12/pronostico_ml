"""Orquestación del motor MPM legacy puro."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import AbstractSet, Mapping, Sequence

from .assemble import assemble_month
from .city_map import map_legacy_temperatures
from .climate_adjuster import ClimateAdjuster
from .climate_fill import fill_temperature_gaps
from .day_types import TipoDia, classify_days, day_type
from .prom_dia import calculate_prom_dia
from .similar_days import HistoricDay, fill_base_demands


METHOD_IDS = (1, 2, 3, 4)
METHOD_ROW = {1: 0, 2: 1, 3: 2, 4: 3}  # índice en assembled_totals_gw

# ``previous_days_demand`` admite el contrato histórico (fecha, demanda) y el
# enriquecido (fecha, demanda, temperatura BAQ). Sin temperatura, esos días
# alimentan las medias del pool pero quedan fuera del match por temperatura.
TEMP_UNKNOWN = float("nan")
PreviousDayDemand = tuple[date, float] | tuple[date, float, float]


@dataclass(frozen=True)
class PredictMonthResult:
    daily_predictions: list[float]
    historic_demands: list[float]
    all_dates: list[str]
    day_types: list[int]
    total_gw: float
    assembled_totals_gw: tuple[float, float, float, float]
    series_for_method: list[float]


def _observed_prefix_count(
    dates: Sequence[date], demand_by_date: Mapping[date, float]
) -> int:
    """El fragmento asume observados como prefijo 1..N del mes."""

    count = 0
    for day in dates:
        if day in demand_by_date:
            count += 1
            continue
        break
    return count


def _series_for_method(
    method_id: int,
    assembled_matrix: list[list[float]],
    prom_values: Sequence[float],
    dates: Sequence[date],
    holidays: AbstractSet[date],
    demand_by_date: Mapping[date, float],
) -> list[float]:
    if method_id == 1:
        series = list(assembled_matrix[2])
    elif method_id == 2:
        series = list(assembled_matrix[4])
    elif method_id == 3:
        series = list(assembled_matrix[5])
    else:
        series = [
            float(prom_values[day_type(day, holidays)])
            if day not in demand_by_date
            else float(demand_by_date[day])
            for day in dates
        ]
        # Para id=4 el total es Prom_Dia·conteos; la serie diaria usa mediana
        # del tipo en faltantes y real en observados.
        for idx, day in enumerate(dates):
            if day not in demand_by_date:
                tipo = day_type(day, holidays)
                series[idx] = float(prom_values[tipo])
    return series


def predict_month(
    *,
    dates: Sequence[date],
    demand_by_date: Mapping[date, float],
    previous_days_demand: Sequence[PreviousDayDemand] | None = None,
    holidays: AbstractSet[date] | None = None,
    temps: Mapping[str, Sequence[float | None]],
    scenario: int,
    method_id: int,
    adjuster: ClimateAdjuster | None = None,
) -> PredictMonthResult:
    """Predice el resto del mes según el algoritmo MPM legacy.

    f5/f6 usan un surrogate calibrado o el fallback lineal.
    """

    if method_id not in METHOD_IDS:
        raise ValueError("method_id debe ser 1, 2, 3 o 4")
    if not dates:
        raise ValueError("dates no puede estar vacío")

    holiday_set: set[date] = set(holidays or ())
    previous = list(previous_days_demand or ())
    previous_demand_pairs = [
        (record[0], float(record[1]))
        for record in previous
    ]

    filled_app_temps = fill_temperature_gaps(dates, temps, scenario)
    legacy_temps = map_legacy_temperatures(filled_app_temps)

    historic_records = previous_demand_pairs + [
        (day, float(demand_by_date[day]))
        for day in dates
        if day in demand_by_date
    ]
    prom = calculate_prom_dia(historic_records, holiday_set)

    historic_days: list[HistoricDay] = []
    for record in previous:
        day, demand = record[:2]
        temp_baq = (
            float(record[2])
            if len(record) == 3 and record[2] is not None
            else TEMP_UNKNOWN
        )
        historic_days.append(
            HistoricDay(
                day=day,
                demand=float(demand),
                temp_baq=temp_baq,
                tipo=day_type(day, holiday_set),
            )
        )
    for idx, day in enumerate(dates):
        if day not in demand_by_date:
            continue
        historic_days.append(
            HistoricDay(
                day=day,
                demand=float(demand_by_date[day]),
                temp_baq=float(legacy_temps["baq"][idx]),
                tipo=day_type(day, holiday_set),
            )
        )

    base_demands = fill_base_demands(
        dates=dates,
        demand_by_date=dict(demand_by_date),
        holidays=holiday_set,
        temps_baq=legacy_temps["baq"],
        historic=historic_days,
    )
    observed_count = _observed_prefix_count(dates, demand_by_date)
    assembled = assemble_month(
        dates=dates,
        holidays=holiday_set,
        temps=legacy_temps,
        base_demands=base_demands,
        observed_count=observed_count,
        prom_dia=prom,
        adjuster=adjuster,
    )

    prom_by_tipo = {
        TipoDia.ORDINARIO: prom.ordinario,
        TipoDia.SABADO: prom.sabado,
        TipoDia.DOMINGO_FESTIVO: prom.domingo_festivo,
    }
    series = _series_for_method(
        method_id=method_id,
        assembled_matrix=assembled.matrix,
        prom_values=prom_by_tipo,
        dates=dates,
        holidays=holiday_set,
        demand_by_date=demand_by_date,
    )

    daily_predictions = [
        series[idx] for idx, day in enumerate(dates) if day not in demand_by_date
    ]
    historic_demands = [
        float(demand_by_date[day]) for day in dates if day in demand_by_date
    ]
    totals = assembled.totals_gw
    return PredictMonthResult(
        daily_predictions=daily_predictions,
        historic_demands=historic_demands,
        all_dates=[day.isoformat() for day in dates],
        day_types=[int(value) for value in classify_days(dates, holiday_set)],
        total_gw=float(totals[METHOD_ROW[method_id]]),
        assembled_totals_gw=totals,
        series_for_method=series,
    )
