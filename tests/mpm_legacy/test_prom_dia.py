"""Tests de Prom_Dia: debe usar MEDIANA, no media."""

from datetime import date
from math import isnan

from api.services.mpm_legacy.prom_dia import calculate_prom_dia


def test_prom_dia_uses_median_not_mean():
    # Lunes-miércoles ordinarios con un outlier: mediana != media.
    records = [
        (date(2026, 5, 4), 10.0),   # lunes
        (date(2026, 5, 5), 20.0),   # martes
        (date(2026, 5, 6), 120.0),  # miércoles (outlier)
    ]
    prom = calculate_prom_dia(records, set())
    assert prom.ordinario == 20.0  # mediana, no 50 (media)


def test_prom_dia_groups_by_type():
    records = [
        (date(2026, 5, 4), 30.0),   # lunes ordinario
        (date(2026, 5, 9), 25.0),   # sábado
        (date(2026, 5, 10), 18.0),  # domingo
        (date(2026, 5, 3), 22.0),   # domingo
    ]
    prom = calculate_prom_dia(records, set())
    assert prom.ordinario == 30.0
    assert prom.sabado == 25.0
    assert prom.domingo_festivo == 20.0  # mediana de [18, 22]


def test_prom_dia_empty_group_is_nan():
    prom = calculate_prom_dia([(date(2026, 5, 4), 30.0)], set())
    assert prom.ordinario == 30.0
    assert isnan(prom.sabado)
    assert isnan(prom.domingo_festivo)
