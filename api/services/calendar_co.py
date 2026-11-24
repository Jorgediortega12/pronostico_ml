"""Calendario laboral de Colombia (único punto de verdad).

Lo usan las features de los modelos v2 (``forecast_rfxgb``: conteos; ``forecast_sarimax``: fracciones) y
``ForecastService.get_number_of_days``. Los artefactos entrenados dependen de estos valores: cualquier cambio
de lógica obliga a reentrenar (``tests/forecast_e2e/test_correcciones_r2.py`` fija los valores 2010-2030).

Requiere ``holidays_co`` (``requirements.txt``); sin respaldo silencioso a otra librería: si falta, falla igual
para todos los consumidores.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


@lru_cache(maxsize=256)
def holidays_year(year: int) -> frozenset:
    """Festivos de Colombia del año como ``frozenset`` de ``datetime.date``."""
    from holidays_co import get_colombia_holidays_by_year

    return frozenset(_as_date(h.date) for h in get_colombia_holidays_by_year(int(year)))


def _as_date(d) -> date:
    return d.date() if hasattr(d, "date") and callable(d.date) else d


@lru_cache(maxsize=4096)
def calendar_counts(year: int, month: int) -> tuple[int, int, int, int]:
    """``(dias, laborables, sabados, domingos_y_festivos)`` del mes.

    Laborable = lunes-viernes no festivo; sábado festivo cuenta como festivo (no como sábado).
    """
    year, month = int(year), int(month)
    hol = holidays_year(year)
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    n_days = (nxt - first).days
    lab = sab = domfest = 0
    for i in range(n_days):
        d = first + timedelta(days=i)
        wd = d.weekday()
        if wd == 6 or d in hol:
            domfest += 1
        elif wd == 5:
            sab += 1
        else:
            lab += 1
    return n_days, lab, sab, domfest
