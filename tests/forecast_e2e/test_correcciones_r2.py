"""Tests de las correcciones de la ronda 2 de revisión (docs/work/07_correcciones_r2/REPORTE.md).

Los tests sin BD corren siempre; los marcados ``integration_db`` se omiten si no hay BD local.
"""
from __future__ import annotations

import contextlib
import io
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from api.services import calendar_co
from api.services import forecast_rfxgb as F
from api.services import forecast_sarimax as fs
from api.services.forecast import ForecastService, contract_month_labels


# --------------------------------------------------------------------------------------------------
# 1. Contrato de fechas: fin de mes ids 1-3, inicio de mes id 4 (como main)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mid,esperado", [
    (1, ["2026-01-31", "2026-02-28", "2028-02-29", "2026-12-31"]),
    (2, ["2026-01-31", "2026-02-28", "2028-02-29", "2026-12-31"]),
    (3, ["2026-01-31", "2026-02-28", "2028-02-29", "2026-12-31"]),
    (4, ["2026-01-01", "2026-02-01", "2028-02-01", "2026-12-01"]),
])
def test_contract_month_labels_formato_main(mid, esperado):
    inicio = pd.Series(pd.to_datetime(["2026-01-01", "2026-02-01", "2028-02-01", "2026-12-01"]))
    assert contract_month_labels(inicio, mid) == esperado
    if mid != 4:  # idempotente con fechas que ya son fin de mes (rama legacy)
        fin = pd.Series(pd.to_datetime(esperado))
        assert contract_month_labels(fin, mid) == esperado


# --------------------------------------------------------------------------------------------------
# 2-3. Caché de re-entrenos y historia mínima RF/XGB
# --------------------------------------------------------------------------------------------------
def test_cache_reentrenos_rfxgb_32():
    assert F.ONLINE_REFIT_CACHE_MAX == 32 and F._ONLINE_REFIT_CACHE.maxsize == 32


def _serie(n, start="2011-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    t = np.arange(n)
    return pd.Series(np.exp(np.log(4e8) + 0.004 * t + 0.05 * np.sin(2 * np.pi * t / 12)), index=idx)


def test_min_training_months_coincide_con_filas_reales():
    req = F.min_training_months(36)
    assert req == 58
    for n in (req - 1, req):
        filas = len(F.training_rows(F.build_design(_serie(n), 36)))
        assert (filas >= F.MIN_TRAIN_ROWS) == (n >= req)


def test_fit_model_historia_insuficiente_mensaje_claro():
    cfg = {"offset": "naive_drift", "params": {"n_estimators": 5}, "h_train": 36}
    with pytest.raises(ValueError, match=r"se requieren >= 58 meses"):
        F.fit_model(_serie(57), 1, cfg)
    F.fit_model(_serie(58), 1, cfg)  # con el mínimo entrena
    art = {"offset": "naive_drift", "params": {"n_estimators": 5}, "h_train": 36, "exclude_targets": []}
    with pytest.raises(ValueError, match=r"se requieren >= 58 meses para re-entrenar"):
        F.refit_from_artifact_cached(1, _serie(53), art, fingerprint="t")


def test_fit_model_pocas_filas_por_exclusion_mensaje_claro():
    cfg = {"offset": "naive_drift", "params": {"n_estimators": 5}, "h_train": 36,
           "exclude_targets": [("2015-01", "2015-12")]}
    with pytest.raises(ValueError, match="historia insuficiente: .* filas de entrenamiento"):
        F.fit_model(_serie(60), 1, cfg)


# --------------------------------------------------------------------------------------------------
# 6. Calendario único: valores idénticos a las implementaciones anteriores (los artefactos dependen de ellos)
# --------------------------------------------------------------------------------------------------
def _ref_rfxgb(year, month):
    from holidays_co import get_colombia_holidays_by_year

    hol = frozenset(h.date for h in get_colombia_holidays_by_year(year))
    days = pd.date_range(date(year, month, 1), periods=pd.Timestamp(year, month, 1).days_in_month, freq="D")
    lab = sab = domfest = 0
    for d in days:
        if d.weekday() == 6 or d.date() in hol:
            domfest += 1
        elif d.weekday() == 5:
            sab += 1
        else:
            lab += 1
    return len(days), lab, sab, domfest


def _ref_sarimax(m):
    from holidays_co import get_colombia_holidays_by_year

    hol = frozenset(pd.Timestamp(h.date) for h in get_colombia_holidays_by_year(m.year))
    days = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
    is_hol = np.array([d in hol for d in days])
    wd = days.weekday
    return (float(((wd < 5) & ~is_hol).sum()) / len(days), float(((wd == 6) | is_hol).sum()) / len(days))


def test_calendario_unico_identico_2010_2030():
    idx = pd.date_range("2010-01-01", "2030-12-01", freq="MS")
    feats = fs.calendar_features(idx, ["frac_habiles", "frac_domfest"])
    svc = ForecastService()
    for i, m in enumerate(idx):
        ref = _ref_rfxgb(m.year, m.month)
        assert calendar_co.calendar_counts(m.year, m.month) == ref
        assert F.calendar_counts is calendar_co.calendar_counts
        hab, domf = _ref_sarimax(m)
        assert feats["frac_habiles"].iloc[i] == hab and feats["frac_domfest"].iloc[i] == domf
        assert svc.get_number_of_days(m.year, m.month) == ref[1:]
    assert calendar_co.calendar_counts(2026, 1) == (31, 20, 5, 6)


# --------------------------------------------------------------------------------------------------
# 7. Clave de caché SARIMAX sin start
# --------------------------------------------------------------------------------------------------
def test_sarimax_cache_no_depende_de_start(monkeypatch):
    hist = _serie(156)  # 2011-01 .. 2023-12
    spec = fs.ModelSpec(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), transform="log_daily", covid="nan")
    fs._REFIT_PARAMS_CACHE.clear()
    n_fit = []
    real_fit = fs.fit_spec
    monkeypatch.setattr(fs, "fit_spec", lambda h, sp: (n_fit.append(1), real_fit(h, sp))[1])
    a = fs.refit_params_cached(hist, spec)
    b = fs.refit_params_cached(hist, spec)
    assert len(n_fit) == 1 and np.array_equal(a, b)


# --------------------------------------------------------------------------------------------------
# Humo de la rama legacy (ML_FORECAST_LEGACY=1) para ids 1-3 — requiere BD
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def session_factory():
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from training.common.config import database_url

    try:
        eng = create_engine(database_url())
        with eng.connect() as c:
            c.execute(text('SELECT 1 FROM "PRONOSTICO_demands" LIMIT 1'))
    except Exception as exc:
        pytest.skip(f"BD no disponible: {type(exc).__name__}")
    return sessionmaker(bind=eng)


@pytest.mark.integration_db
@pytest.mark.parametrize("mid", [1, 2, 3])
def test_legacy_humo_ids_1_3(session_factory, monkeypatch, mid):
    from api.config.schemas import ForecastSchema

    monkeypatch.setenv("ML_FORECAST_LEGACY", "1")
    db = session_factory()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = ForecastService().predict(
                ForecastSchema(id=mid, user_id=1, session_id=1, start_date="2026-01-01", end_date="2026-12-31"), db)
    finally:
        db.close()
    assert "message" not in r, r.get("message")
    assert r["modelo"].get("version") != "v2"
    mp = r["monthly_predictions"]
    esperado = pd.period_range("2026-01", "2026-12", freq="M")
    assert mp["fechas"] == [p.to_timestamp(how="end").strftime("%Y-%m-%d") for p in esperado]
    v = np.asarray(mp["valores_pronostico"], dtype=float)
    assert len(v) == 12 and np.isfinite(v).all() and (v > 0).all()
    json.dumps(r, allow_nan=False)
