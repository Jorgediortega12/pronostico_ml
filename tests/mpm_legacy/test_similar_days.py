"""Tests del barrido de días similares (umbral 7e5 / fallback 8e5).

Los casos GOLDEN reproducen a mano el bloque ``for i = 1:3 / switch i`` del
fragmento MATLAB sobre una ventana Jun/2026 con weekday conocido:

    Jun 01 Lun(2)  Jun 02 Mar(3)  Jun 03 Mié(4)  Jun 04 Jue(5)  Jun 05 Vie(6)
    Jun 06 Sáb(7)  Jun 07 Dom(1)  Jun 08 Lun(2)  ...  Jun 12 Vie(6)
"""

import math
from datetime import date

import pytest

from api.services.mpm_legacy.day_types import TipoDia, day_type, matlab_weekday
from api.services.mpm_legacy.similar_days import (
    DESV_NO_MATCH,
    DESV_THRESHOLD,
    HistoricDay,
    fill_base_demands,
    round_temp,
    select_similar_demand,
)


WINDOW = [date(2026, 6, day) for day in range(1, 13)]
FLAT_TEMPS = [30.0] * len(WINDOW)

# Pool histórico del "mes anterior" con temperatura conocida.
#   Lun 25/05 → 29.5 M @30°  (|Desv| < 7e5 con un mes actual de 30 M)
#   Mar 26/05 → 29.6 M @30°  (|Desv| < 7e5)
#   Mié 27/05 → 29.0 M @30°  (Desv = 1e6 > 7e5)
#   Jue 28/05 → 29.8 M @25°  (no hay match de temperatura)
#   Vie 29/05 → 29.9 M @30°
PREVIOUS_ORDINARY = (
    (date(2026, 5, 25), 29_500_000.0, 30.0),
    (date(2026, 5, 26), 29_600_000.0, 30.0),
    (date(2026, 5, 27), 29_000_000.0, 30.0),
    (date(2026, 5, 28), 29_800_000.0, 25.0),
    (date(2026, 5, 29), 29_900_000.0, 30.0),
)
PREVIOUS_SATURDAY = (date(2026, 5, 30), 24_000_000.0, 30.0)
PREVIOUS_SUNDAY = (date(2026, 5, 31), 20_000_000.0, 30.0)


def _hist(day, demand, temp):
    return HistoricDay(day=day, demand=demand, temp_baq=temp, tipo=day_type(day, set()))


def _previous_pool():
    return [
        _hist(*record)
        for record in (*PREVIOUS_ORDINARY, PREVIOUS_SATURDAY, PREVIOUS_SUNDAY)
    ]


def test_thresholds():
    assert DESV_THRESHOLD == 7e5
    assert DESV_NO_MATCH == 8e5


def test_round_temp():
    assert round_temp(30.4) == 30.0
    assert round_temp(30.6) == 31.0


def test_round_temp_propagates_nan_instead_of_raising():
    # round(nan) lanza ValueError en Python 3; MATLAB propaga NaN.
    value = round_temp(float("nan"))
    assert math.isnan(value)
    # NaN nunca iguala a NaN → el día queda fuera del match por temperatura.
    assert round_temp(float("nan")) != round_temp(float("nan"))


def test_observed_days_keep_real_demand():
    dates = [date(2026, 5, 4), date(2026, 5, 5)]
    demand = {date(2026, 5, 4): 30_000_000.0}
    temps = [30.0, 31.0]
    historic = [_hist(date(2026, 5, 4), 30_000_000.0, 30.0)]
    values = fill_base_demands(dates, demand, set(), temps, historic)
    assert values[0] == 30_000_000.0  # observado = real


def test_missing_ordinary_day_gets_similar_or_baseline():
    # Día faltante ordinario: debe recibir un valor > 0 basado en histórico.
    dates = [date(2026, 5, 4), date(2026, 5, 5)]
    demand = {date(2026, 5, 4): 30_000_000.0}
    temps = [30.0, 30.0]
    historic = [
        _hist(date(2026, 4, 27), 29_000_000.0, 30.0),  # lunes previo
        _hist(date(2026, 5, 4), 30_000_000.0, 30.0),
    ]
    values = fill_base_demands(dates, demand, set(), temps, historic)
    assert values[1] > 0.0


# --------------------------------------------------------------------------
# GOLDEN: barrido case 1/2/3 con mes actual observado (Jun 01 = 30 M)
# --------------------------------------------------------------------------


def _golden_values():
    demand = {date(2026, 6, 1): 30_000_000.0}
    historic = _previous_pool() + [_hist(date(2026, 6, 1), 30_000_000.0, 30.0)]
    return fill_base_demands(WINDOW, demand, set(), FLAT_TEMPS, historic)


def test_golden_match_within_threshold_uses_candidate():
    # Lunes: candidato 29.5 M, mean(DemandaActual) = 30 M → Desv = 5e5 ≤ 7e5.
    # Martes: candidato 29.6 M → Desv = 4e5 ≤ 7e5.
    values = _golden_values()
    assert values[7] == 29_500_000.0  # Jun 08, lunes
    assert values[1] == 29_600_000.0  # Jun 02, martes


def test_golden_desv_above_threshold_uses_current_month_mean():
    # Miércoles: candidato 29.0 M → Desv = 30 M - 29 M = 1e6 > 7e5.
    values = _golden_values()
    assert values[2] == 30_000_000.0  # Jun 03
    assert values[9] == 30_000_000.0  # Jun 10


def test_golden_no_similar_day_falls_back_to_baseline():
    # Jueves: el único jueves histórico está a 25° → no hay match de temp,
    # Desv = 8e5 > 7e5 → mean(DemandaActual) = 30 M.
    values = _golden_values()
    assert values[3] == 30_000_000.0  # Jun 04
    assert values[10] == 30_000_000.0  # Jun 11


def test_golden_no_similar_and_empty_current_month_uses_pool_mean():
    # Sin mes actual, MATLAB cae a mean(Demanda(:,2)) del pool ordinario:
    # (29.5 + 29.6 + 29.0 + 29.8 + 29.9) / 5 = 29.56 M
    values = fill_base_demands(WINDOW, {}, set(), FLAT_TEMPS, _previous_pool())
    assert values[3] == 29_560_000.0  # Jun 04, jueves sin match de temp


def test_golden_major8_match_without_current_month_keeps_candidate():
    # MATLAB: Desv = mean([]) - candidato = NaN, y NaN > 7e5 es falso,
    # así que el día similar gana aunque no haya demanda del mes actual.
    values = fill_base_demands(WINDOW, {}, set(), FLAT_TEMPS, _previous_pool())
    assert values[0] == 29_500_000.0  # Jun 01, lunes
    assert values[7] == 29_500_000.0  # Jun 08, lunes


def test_golden_sweep_writes_same_demand_across_matching_columns():
    # case 1 escribe el mismo valor en TODAS las columnas con ese
    # (weekday, TipoDia), no un valor por día con su propia temperatura.
    values = _golden_values()
    for first, second in ((1, 8), (2, 9), (3, 10), (4, 11)):
        assert values[first] == values[second]
        assert matlab_weekday(WINDOW[first]) == matlab_weekday(WINDOW[second])


def test_golden_sweep_ignores_per_day_temperature():
    # Aunque Jun 08 tenga otra temperatura, el barrido usa la del PRIMER día
    # del mes con ese weekday (Jun 01), así que el resultado no cambia.
    demand = {date(2026, 6, 1): 30_000_000.0}
    historic = _previous_pool() + [_hist(date(2026, 6, 1), 30_000_000.0, 30.0)]
    temps = list(FLAT_TEMPS)
    temps[7] = 25.0
    values = fill_base_demands(WINDOW, demand, set(), temps, historic)
    assert values[7] == 29_500_000.0


def test_golden_saturday_and_sunday_sweeps():
    # case 2: sin sábados del mes actual, DemandaActual = pool → Desv = 0.
    # case 3: media del pool domingo/festivo, sin temperatura.
    values = _golden_values()
    assert values[5] == 24_000_000.0  # Jun 06, sábado
    assert values[6] == 20_000_000.0  # Jun 07, domingo


def test_golden_sunday_pool_is_averaged():
    historic = _previous_pool() + [
        _hist(date(2026, 5, 24), 22_000_000.0, 30.0),  # otro domingo
    ]
    values = fill_base_demands(WINDOW, {}, set(), FLAT_TEMPS, historic)
    assert values[6] == 21_000_000.0  # (20 M + 22 M) / 2


def test_golden_holiday_uses_sunday_sweep():
    # Un festivo entre semana es TipoDia -1 y lo pisa el case 3.
    holidays = {date(2026, 6, 3)}
    values = fill_base_demands(WINDOW, {}, holidays, FLAT_TEMPS, _previous_pool())
    assert day_type(date(2026, 6, 3), holidays) == TipoDia.DOMINGO_FESTIVO
    assert values[2] == 20_000_000.0


# --------------------------------------------------------------------------
# GOLDEN: días previos sin temperatura (previous_days_demand)
# --------------------------------------------------------------------------


def test_previous_days_without_temperature_do_not_crash():
    # Antes: round(nan) → ValueError. Ahora quedan fuera del match por temp
    # pero siguen alimentando las medias del pool.
    nan = float("nan")
    historic = [
        _hist(date(2026, 5, 25), 29_500_000.0, nan),
        _hist(date(2026, 5, 26), 29_600_000.0, nan),
        _hist(date(2026, 5, 30), 24_000_000.0, nan),
        _hist(date(2026, 5, 31), 20_000_000.0, nan),
    ]
    values = fill_base_demands(WINDOW, {}, set(), FLAT_TEMPS, historic)
    assert all(math.isfinite(value) for value in values)
    # Ningún match posible → Desv = 8e5 → media del pool ordinario.
    assert values[0] == 29_550_000.0  # (29.5 M + 29.6 M) / 2
    assert values[5] == 24_000_000.0  # sábado: DemandaActual = pool
    assert values[6] == 20_000_000.0  # domingo: media del pool


def test_temp_nan_in_target_day_does_not_crash():
    values = fill_base_demands(
        WINDOW, {}, set(), [float("nan")] * len(WINDOW), _previous_pool()
    )
    # Sin temperatura objetivo no hay match → media del pool ordinario.
    assert values[0] == 29_560_000.0


# --------------------------------------------------------------------------
# select_similar_demand (helper de un solo día)
# --------------------------------------------------------------------------


def test_select_similar_demand_keeps_candidate_without_current_month():
    historic = [_hist(date(2026, 5, 25), 29_500_000.0, 30.0)]
    value = select_similar_demand(
        weekday=2,
        tipo=TipoDia.ORDINARIO,
        temp_baq=30.0,
        historic=historic,
        current_month=[],
    )
    assert value == 29_500_000.0


def test_select_similar_demand_applies_threshold():
    historic = [_hist(date(2026, 5, 25), 29_000_000.0, 30.0)]
    current = [_hist(date(2026, 6, 1), 30_000_000.0, 30.0)]
    value = select_similar_demand(
        weekday=2,
        tipo=TipoDia.ORDINARIO,
        temp_baq=30.0,
        historic=historic,
        current_month=current,
    )
    assert value == 30_000_000.0  # Desv = 1e6 > 7e5


def test_fill_base_demands_rejects_misaligned_temps():
    with pytest.raises(ValueError):
        fill_base_demands(WINDOW, {}, set(), [30.0], _previous_pool())
