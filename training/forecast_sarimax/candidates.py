"""Candidatos por etapa. Los órdenes de la etapa 2 salen del top por BIC de s1_aic.csv (train <= 2023-12)."""
from __future__ import annotations

import itertools

from api.services.forecast_sarimax import ModelSpec

# Top-BIC de s1 (ambas transformaciones coinciden en el top): airline y vecinos + variantes d=0 con constante.
S2_ORDERS = [
    ((0, 1, 1), (0, 1, 1, 12), None),
    ((2, 1, 0), (0, 1, 1, 12), None),
    ((0, 1, 1), (1, 1, 1, 12), None),
    ((1, 1, 1), (0, 1, 1, 12), None),
    ((1, 0, 1), (0, 1, 1, 12), "c"),
    ((2, 0, 0), (0, 1, 1, 12), "c"),
]


def stage2() -> dict:
    c = {}
    for (o, so, tr), transform, covid in itertools.product(S2_ORDERS, ("log", "log_daily"), ("nan", "none", "dummy")):
        s = ModelSpec(order=o, seasonal_order=so, trend=tr, transform=transform, covid=covid)
        c[s.name()] = s
    for transform, covid in itertools.product(("log", "log_daily"), ("nan", "none")):
        s = ModelSpec(kind="ets", transform=transform, covid=covid, damped=True)
        c[s.name()] = s
    return c


def stage2b() -> dict:
    """Vecinos de la familia ganadora de la etapa 2 (d=0, D=1, constante = crecimiento interanual medio)."""
    c = {}
    orders = [((1, 0, 0), (0, 1, 1, 12)), ((2, 0, 1), (0, 1, 1, 12)), ((3, 0, 0), (0, 1, 1, 12)),
              ((2, 0, 0), (1, 1, 1, 12)), ((2, 0, 0), (0, 1, 0, 12))]
    for (o, so), transform, covid in itertools.product(orders, ("log", "log_daily"), ("nan", "none")):
        s = ModelSpec(order=o, seasonal_order=so, trend="c", transform=transform, covid=covid)
        c[s.name()] = s
    return c


EXOG_SETS =[("frac_habiles",), ("frac_domfest",), ("frac_habiles", "frac_domfest")]


def stage3(base_specs) -> dict:
    """Exógenas de calendario sobre los mejores specs SARIMA de la etapa 2."""
    from dataclasses import replace

    c = {}
    for b in base_specs:
        for ex in EXOG_SETS:
            s = replace(b, exog=ex)
            c[s.name()] = s
    return c
