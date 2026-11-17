"""Ensambla AssembledData (10xn) y PronosticoTotal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isnan
from typing import AbstractSet, Sequence

from .climate_adjust import (
    WH_SCALE,
    apply_linear_baq,
    find_climate_reference,
)
from .climate_adjuster import ClimateAdjuster, get_default_adjuster
from .day_types import TipoDia, day_type, matlab_weekday
from .prom_dia import PromDia


@dataclass
class AssembledMonth:
    """Matriz legacy 10xn más totales en GW."""

    matrix: list[list[float]]
    totals_gw: tuple[float, float, float, float]

    @property
    def f3(self) -> list[float]:
        return list(self.matrix[2])

    @property
    def f4(self) -> list[float]:
        return list(self.matrix[3])

    @property
    def f5(self) -> list[float]:
        return list(self.matrix[4])

    @property
    def f6(self) -> list[float]:
        return list(self.matrix[5])


def count_day_types(
    dates: Sequence[date], holidays: AbstractSet[date]
) -> tuple[int, int, int]:
    """Retorna (NLaborales, NSabados, NFestivos+NDomingos)."""

    n_lab = n_sab = n_fest_dom = 0
    for day in dates:
        tipo = day_type(day, holidays)
        if tipo == TipoDia.ORDINARIO:
            n_lab += 1
        elif tipo == TipoDia.SABADO:
            n_sab += 1
        else:
            n_fest_dom += 1
    return n_lab, n_sab, n_fest_dom


def _safe_prom_total(prom_dia: PromDia, counts: tuple[int, int, int]) -> float:
    values = [
        (prom_dia.ordinario, counts[0]),
        (prom_dia.sabado, counts[1]),
        (prom_dia.domingo_festivo, counts[2]),
    ]
    total = 0.0
    for median_value, count in values:
        if count and not isnan(median_value):
            total += median_value * count
    return total / 1e6


def assemble_month(
    dates: Sequence[date],
    holidays: AbstractSet[date],
    temps: dict[str, Sequence[float]],
    base_demands: Sequence[float],
    observed_count: int,
    prom_dia: PromDia,
    adjuster: ClimateAdjuster | None = None,
) -> AssembledMonth:
    """Arma AssembledData y los 4 totales (f3, f5, f6, prom_dia).

    f4 (lineal) se calcula por fidelidad al dump pero no entra en los 4 totales.
    f5/f6 usan correctores residuales calibrados con fallback lineal.
    """

    climate_adjuster = adjuster or get_default_adjuster()
    n = len(dates)
    weekdays = [matlab_weekday(day) for day in dates]
    tipos = [int(day_type(day, holidays)) for day in dates]
    temps_baq = [round(float(value), 1) for value in temps["baq"]]
    temps_car = [round(float(value), 1) for value in temps["car"]]
    temps_mon = [round(float(value), 1) for value in temps["mon"]]
    temps_sm = [round(float(value), 1) for value in temps["sm"]]

    f3 = [float(value) for value in base_demands]
    f4 = list(f3)
    f5 = list(f3)
    f6 = list(f3)

    if 0 <= observed_count < n:
        ref_slice = slice(0, observed_count) if observed_count else slice(None)
        ref_weekdays = weekdays[ref_slice]
        ref_tipos = tipos[ref_slice]
        ref_temps = temps_baq[ref_slice]
        ref_car = temps_car[ref_slice]
        ref_sm = temps_sm[ref_slice]

        for future_idx in range(observed_count, n):
            found = find_climate_reference(
                target_weekday=weekdays[future_idx],
                target_tipo=tipos[future_idx],
                target_temp_baq=temps_baq[future_idx],
                ref_weekdays=ref_weekdays,
                ref_tipos=ref_tipos,
                ref_temps_baq=ref_temps,
            )
            if found is None:
                continue
            ref_idx, exact_match = found
            if exact_match:
                # Mismo TipoDia + temp Baq: el fragmento no ajusta.
                continue

            d_t_baq = temps_baq[future_idx] - ref_temps[ref_idx]
            d_t_3c = (
                temps_baq[future_idx] - ref_temps[ref_idx],
                temps_car[future_idx] - ref_car[ref_idx],
                temps_sm[future_idx] - ref_sm[ref_idx],
            )
            base = f3[future_idx]
            f4[future_idx] = apply_linear_baq(base, d_t_baq)
            f5[future_idx] = (
                base + climate_adjuster.adjust_baq(d_t_baq) * WH_SCALE
            )
            f6[future_idx] = (
                base + climate_adjuster.adjust_3c(*d_t_3c) * WH_SCALE
            )

    matrix = [
        [float(value) for value in weekdays],
        [float(value) for value in tipos],
        f3,
        f4,
        f5,
        f6,
        temps_baq,
        temps_car,
        temps_mon,
        temps_sm,
    ]

    counts = count_day_types(dates, holidays)
    totals = (
        sum(f3) / 1e6,
        sum(f5) / 1e6,
        sum(f6) / 1e6,
        _safe_prom_total(prom_dia, counts),
    )
    return AssembledMonth(matrix=matrix, totals_gw=totals)
