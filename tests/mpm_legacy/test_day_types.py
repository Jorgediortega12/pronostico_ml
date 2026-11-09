"""Tests de clasificación de días (paridad weekday MATLAB)."""

from datetime import date

from api.services.mpm_legacy.day_types import (
    TipoDia,
    classify_days,
    day_type,
    matlab_weekday,
)


def test_matlab_weekday_mapping():
    # MATLAB: 1=domingo ... 7=sábado
    assert matlab_weekday(date(2026, 5, 3)) == 1  # domingo
    assert matlab_weekday(date(2026, 5, 4)) == 2  # lunes
    assert matlab_weekday(date(2026, 5, 8)) == 6  # viernes
    assert matlab_weekday(date(2026, 5, 9)) == 7  # sábado


def test_ordinary_saturday_sunday():
    assert day_type(date(2026, 5, 4), set()) == TipoDia.ORDINARIO  # lunes
    assert day_type(date(2026, 5, 9), set()) == TipoDia.SABADO
    assert day_type(date(2026, 5, 3), set()) == TipoDia.DOMINGO_FESTIVO


def test_holiday_precedence_over_weekday():
    # Un viernes festivo debe clasificarse como domingo/festivo.
    festivo = date(2026, 5, 1)  # viernes, Día del Trabajo
    assert day_type(festivo, set()) == TipoDia.ORDINARIO  # sin festivos
    assert day_type(festivo, {festivo}) == TipoDia.DOMINGO_FESTIVO


def test_classify_days_batch():
    dates = [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 4)]
    result = classify_days(dates, {date(2026, 5, 1)})
    assert result == [
        TipoDia.DOMINGO_FESTIVO,
        TipoDia.SABADO,
        TipoDia.ORDINARIO,
    ]
