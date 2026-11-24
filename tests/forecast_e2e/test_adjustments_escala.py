"""Tope / override / error en la escala del front (unitarios, sin BD)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from api.services.forecast_adjustments import (
    KNOWN_MONTH_OVERRIDES,
    apply_known_month_overrides,
    cap_xgb_sarimax_error,
    front_scale_pct_error,
)
from api.services.forecast_history import DivisionMercado, clean_monthly_from_raw


def _df(preds, reals, start="2026-07-01"):
    fechas = pd.date_range(start, periods=len(preds), freq="MS")
    return pd.DataFrame({"fecha": fechas, "prediccion": preds, "valor_real": reals})


def test_cap_sin_eligible_conserva_comportamiento_anterior():
    df = _df([110.0, 100.0], [100.0, 100.0])
    out = cap_xgb_sarimax_error(df, model_id=1)
    err = (out["prediccion"] - out["valor_real"]) / out["valor_real"] * 100
    assert 1.0 <= err.iloc[0] <= 4.0
    assert out["prediccion"].iloc[1] == 100.0


def test_cap_no_actua_en_meses_no_elegibles_post_division():
    # sep-2026 post-división: real ya al 70 %, pronóstico en escala completa (+43 %).
    df = _df([740.0, 840.0], [735.0, 587.0], start="2026-08-01")
    eligible = pd.Series([True, False])
    out = cap_xgb_sarimax_error(df, model_id=1, eligible=eligible)
    assert out["prediccion"].iloc[1] == 840.0  # no se "pega" al real post-división
    assert out["prediccion"].iloc[0] == 740.0  # error < 4 %: no-op


def test_cap_reproduce_bug_415_gwh_sin_eligible():
    """Documenta la causa raíz: sin máscara, el tope pegaba el pronóstico al real ya al 70 % y el front
    lo volvía a multiplicar por 0.7 (~415 GWh)."""
    df = _df([740.0], [587.0], start="2026-09-01")
    capped = cap_xgb_sarimax_error(df, model_id=1)["prediccion"].iloc[0]
    assert capped * 0.7 < 430  # el valor que veía el front
    ok = cap_xgb_sarimax_error(df, model_id=1, eligible=pd.Series([False]))["prediccion"].iloc[0]
    assert ok * 0.7 == pytest.approx(518.0)


def test_front_scale_pct_error_pre_y_post():
    err = front_scale_pct_error(pd.Series([110.0, 100.0]), pd.Series([100.0, 70.0]), pd.Series([False, True]), 0.7)
    assert err.iloc[0] == pytest.approx(10.0)  # pre: (pred*f - real*f)/(real*f)
    assert err.iloc[1] == pytest.approx(0.0)  # post: (pred*f - real)/real


def test_override_requiere_real_coincidente():
    (y, m), val = next(iter(KNOWN_MONTH_OVERRIDES.items()))
    fecha = pd.Timestamp(year=y, month=m, day=1)
    base = pd.DataFrame({"fecha": [fecha], "prediccion": [val * 0.9]})
    # sin real -> no aplica
    out = apply_known_month_overrides(base.assign(valor_real=[np.nan]), require_matching_real=True)
    assert not out["override_aplicado"].iloc[0] and out["prediccion"].iloc[0] == val * 0.9
    # real de otra empresa/base -> no aplica
    out = apply_known_month_overrides(base.assign(valor_real=[val * 0.5]), require_matching_real=True)
    assert not out["override_aplicado"].iloc[0]
    # real coincidente -> aplica
    out = apply_known_month_overrides(base.assign(valor_real=[val]), require_matching_real=True)
    assert out["override_aplicado"].iloc[0] and out["prediccion"].iloc[0] == val
    # API antigua (sin guard) sigue igual
    out = apply_known_month_overrides(base)
    assert out["prediccion"].iloc[0] == val


@pytest.mark.parametrize(
    "fecha_limite,primer_post,post_sep",
    [(date(2026, 8, 31), "2026-09-01", True), (date(2026, 9, 1), "2026-09-01", True),
     (date(2026, 9, 30), "2026-10-01", False)],
)
def test_division_semantica_adjust_period_value(fecha_limite, primer_post, post_sep):
    d = DivisionMercado(fecha_limite=fecha_limite, porcentaje=70.0)
    assert d.factor == pytest.approx(0.7)
    assert d.first_post_month == pd.Timestamp(primer_post)
    assert d.is_post(pd.Timestamp("2026-09-01")) is post_sep
    assert d.is_post(pd.Timestamp("2026-08-01")) is False


def test_clean_monthly_from_raw_dedup_y_parciales():
    dias = pd.date_range("2020-01-01", "2020-04-15", freq="D")
    raw = pd.DataFrame({"id": range(len(dias)), "fecha": dias, "total": 100.0, "file_type": "txf"})
    dup = raw[raw["fecha"].dt.month == 2].copy()
    dup["total"] = 10.0  # fila duplicada a ~1/10
    dup["id"] = dup["id"] + 10_000
    raw = pd.concat([raw, dup], ignore_index=True)
    m = clean_monthly_from_raw(raw)
    assert m.loc["2020-02-01", "demanda_kwh"] == pytest.approx(100.0 * 29)
    assert np.isnan(m.loc["2020-04-01", "demanda_kwh"])  # mes parcial final no se usa ni se imputa
