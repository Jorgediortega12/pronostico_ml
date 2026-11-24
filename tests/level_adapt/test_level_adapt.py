"""Tests de la correccion adaptativa de nivel (``training/common/level_adapt.py``, ``api/services/forecast_level.py``).

Los de nucleo son puros (sin BD). Los de integracion con ``ForecastService`` usan la BD local y se omiten si no
esta disponible (``integration_db``). Ver ``docs/work/11_nivel_adaptativo/REPORTE.md``.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.services import forecast_level  # noqa: E402
from training.common import level_adapt as la  # noqa: E402
from training.common.level_adapt import LevelSpec, adjust, apply_level, covid_exclusions  # noqa: E402

SPEC = LevelSpec(k=6, hl_w=2.0, est="median", phi=0.5, hl_h=3.0, cap=0.06, min_obs=3)


def _hist(n=72, start="2018-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series(1e8 * (1 + 0.004 * np.arange(n)) * (1 + 0.05 * np.sin(2 * np.pi * idx.month / 12)), index=idx)


def _fc(hist, n=12):
    idx = pd.date_range(hist.index.max() + pd.offsets.MonthBegin(1), periods=n, freq="MS")
    return pd.Series(np.full(n, 1.3e8), index=idx)


def _ref_with_ratio(hist, ratio):
    """Pronostico de referencia tal que real/pron = ``ratio`` en los meses posteriores a su origen (stub de prueba)."""
    def ref_fn(hist_ref, h):
        o = hist_ref.index.max()
        idx = pd.date_range(o + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        return pd.Series(hist.reindex(idx).to_numpy() / ratio, index=idx)
    return ref_fn


# ---------------------------------------------------------------- nucleo
def test_sin_fuga_ref_fn_solo_ve_historia_hasta_el_origen_de_referencia():
    hist = _hist()
    seen = []

    def ref_fn(hist_ref, h):
        seen.append((hist_ref.index.max(), len(hist_ref), h))
        return _ref_with_ratio(hist, 1.0)(hist_ref, h)

    res = adjust(hist, _fc(hist), ref_fn, SPEC, ())
    last = hist.index.max()
    assert seen == [(last - pd.DateOffset(months=SPEC.k), len(hist) - SPEC.k, SPEC.k)]
    assert res.origen_ref == last - pd.DateOffset(months=SPEC.k)


def test_el_factor_no_depende_del_pronostico_base():
    hist = _hist()
    f1 = adjust(hist, _fc(hist), _ref_with_ratio(hist, 1.04), SPEC, ()).factor
    f2 = adjust(hist, _fc(hist) * 3.0, _ref_with_ratio(hist, 1.04), SPEC, ()).factor
    np.testing.assert_allclose(f1.to_numpy(), f2.to_numpy())


def test_sin_desvio_factor_exactamente_1():
    hist = _hist()
    res = adjust(hist, _fc(hist), _ref_with_ratio(hist, 1.0), SPEC, ())
    assert res.aplicado and np.allclose(res.factor.to_numpy(), 1.0, atol=1e-12)
    pd.testing.assert_series_equal(apply_level(_fc(hist), res), _fc(hist))


def test_subestimacion_sube_y_sobreestimacion_baja():
    hist = _hist()
    up = adjust(hist, _fc(hist), _ref_with_ratio(hist, 1.03), SPEC, ()).factor
    dn = adjust(hist, _fc(hist), _ref_with_ratio(hist, 0.97), SPEC, ()).factor
    assert (up > 1).all() and (dn < 1).all()


@pytest.mark.parametrize("hist_len", [0, 3, 6])
def test_sin_meses_observados_suficientes_factor_1(hist_len):
    hist = _hist(n=hist_len) if hist_len else pd.Series(dtype=float)
    fc = pd.Series(1e8, index=pd.date_range("2024-01-01", periods=6, freq="MS"))
    called = []
    res = adjust(hist, fc, lambda h, k: called.append(1) or h, SPEC, ())
    assert not res.aplicado and (res.factor == 1.0).all() and not called


def test_ref_fn_con_error_deja_factor_1():
    hist = _hist()

    def boom(hist_ref, h):
        raise ValueError("historia insuficiente")

    res = adjust(hist, _fc(hist), boom, SPEC, ())
    assert not res.aplicado and (res.factor == 1.0).all() and "ValueError" in res.motivo


def test_meses_covid_no_son_residuos():
    hist = _hist(n=84, start="2017-01-01")
    hist = hist[hist.index <= "2020-08-01"]  # ultimo mes en pleno COVID
    fc = _fc(hist, 6)
    res = adjust(hist, fc, _ref_with_ratio(pd.concat([hist, fc]), 1.5), SPEC, covid_exclusions())
    # de la ventana de 6 meses (2020-03..08) todos son COVID: sin residuos validos
    assert not res.aplicado and res.motivo == "residuos_insuficientes" and (res.factor == 1.0).all()


def test_limite_cap_y_amortiguacion_phi():
    hist = _hist()
    big = adjust(hist, _fc(hist), _ref_with_ratio(hist, 1.5), SPEC, ())
    assert abs(big.ajuste_log) <= SPEC.cap + 1e-12 and abs(np.log(big.factor.max())) <= SPEC.cap + 1e-12
    no = adjust(hist, _fc(hist), _ref_with_ratio(hist, 1.5), LevelSpec(**{**SPEC.to_dict(), "phi": 0.0}), ())
    assert (no.factor == 1.0).all()


def test_decaimiento_por_horizonte_y_semivida():
    hist = _hist()
    res = adjust(hist, _fc(hist, 24), _ref_with_ratio(hist, 1.04), SPEC, ())
    c = np.log(res.factor.to_numpy())
    assert (np.diff(c) < 0).all() and c[0] == pytest.approx(res.ajuste_log)
    assert c[int(SPEC.hl_h)] == pytest.approx(c[0] / 2)  # h = 1 + hl_h -> mitad


def test_sin_decaimiento_con_hl_h_none():
    hist = _hist()
    res = adjust(hist, _fc(hist, 12), _ref_with_ratio(hist, 1.04), LevelSpec(**{**SPEC.to_dict(), "hl_h": None}), ())
    assert np.allclose(res.factor.to_numpy(), res.factor.iloc[0])


def test_pronostico_debe_ser_posterior_al_ultimo_mes():
    hist = _hist()
    with pytest.raises(ValueError):
        adjust(hist, hist.iloc[-3:], lambda h, k: h, SPEC, ())


def test_compuerta_de_persistencia_y_zona_muerta():
    idx = pd.date_range("2024-01-01", periods=6, freq="MS")
    mixto = pd.Series([0.04, -0.04, 0.04, -0.04, 0.04, 0.01], index=idx)
    last = idx.max()
    base = LevelSpec(k=6, hl_w=None, est="mean", phi=1.0, hl_h=None, cap=1.0)
    assert la.estimate_level(mixto, last, base)[0] != 0.0
    assert la.estimate_level(mixto, last, LevelSpec(**{**base.to_dict(), "agree": 0.9}))[0] == 0.0
    chico = pd.Series(0.01, index=idx)
    assert la.estimate_level(chico, last, LevelSpec(**{**base.to_dict(), "tau": 0.02}))[0] == 0.0
    assert la.estimate_level(chico, last, LevelSpec(**{**base.to_dict(), "tau": 0.004}))[0] == pytest.approx(0.006)


def test_validacion_de_spec():
    for bad in ({"k": 0}, {"phi": 1.5}, {"est": "x"}, {"hl_h": 0.0}, {"min_obs": 9}, {"cap": -1}):
        with pytest.raises(ValueError):
            LevelSpec(**{**SPEC.to_dict(), **bad}).validate()
    assert LevelSpec.from_dict({**SPEC.to_dict(), "extra": 1}) == SPEC


def test_paridad_con_la_evaluacion_vectorizada():
    from training.level_adapt import evaluate as ev

    rng = np.random.default_rng(3)
    R = rng.normal(0.01, 0.03, size=(40, 6))
    R[rng.random(R.shape) < 0.15] = np.nan
    for est in ("mean", "median"):
        for kw in ({}, {"agree": 0.8}, {"tau": 0.01}, {"hl_w": None}):
            sp = LevelSpec(**{**SPEC.to_dict(), "est": est, **kw})
            vec = ev.level_from_R(R, sp)
            last = pd.Timestamp("2024-06-01")
            idx = pd.date_range(last - pd.DateOffset(months=5), last, freq="MS")[::-1]  # j=0 el mas reciente
            for i in range(len(R)):
                one, _ = la.estimate_level(pd.Series(R[i], index=idx), last, sp)
                assert vec[i] == pytest.approx(one, abs=1e-12), (est, kw, i)


def test_config_json_coincide_con_la_spec_evaluada():
    from training.level_adapt import evaluate as ev

    spec, models = forecast_level.load_config()
    assert spec == ev.CHOSEN and models == frozenset({1, 2, 4})


# ---------------------------------------------------------------- flag
def test_flag_por_defecto_apagado(monkeypatch):
    monkeypatch.delenv("LEVEL_ADAPT", raising=False)
    assert not forecast_level.enabled() and not forecast_level.applies_to(1)
    for v in ("0", "", "false"):
        monkeypatch.setenv("LEVEL_ADAPT", v)
        assert not forecast_level.applies_to(1)
    monkeypatch.setenv("LEVEL_ADAPT", "1")
    assert forecast_level.applies_to(1) and forecast_level.applies_to(4) and not forecast_level.applies_to(3)


def test_adjust_forecast_flag_apagado_devuelve_el_mismo_objeto(monkeypatch):
    monkeypatch.setenv("LEVEL_ADAPT", "0")
    hist = _hist()
    fc = _fc(hist)
    out, info = forecast_level.adjust_forecast(1, hist, fc, _ref_with_ratio(hist, 1.1))
    assert out is fc and info is None


def test_adjust_forecast_flag_encendido_reporta_info(monkeypatch):
    monkeypatch.setenv("LEVEL_ADAPT", "1")
    hist = _hist()
    fc = _fc(hist)
    out, info = forecast_level.adjust_forecast(1, hist, fc, _ref_with_ratio(hist, 1.04))
    assert info["aplicado"] and info["meses_residuo"] == 6 and info["factor_primer_mes"] > 1
    assert (out > fc).all() and json.dumps(info, allow_nan=False)


# ---------------------------------------------------------------- id 4 (sin BD)
ARTIFACTS = BACKEND_ROOT / "training" / "forecast_climate" / "artifacts"
needs_clima = pytest.mark.skipif(not (ARTIFACTS / "dataset_mensual.csv").exists(), reason="dataset climatico no disponible")


@needs_clima
def test_clima_level_spec_none_identico_y_sin_fuga():
    from training.forecast_climate import clima_v3 as v3
    from training.forecast_climate import walkforward_clima_v3 as wf3

    full = wf3.load_hist()
    kw = dict(cutoff_year=2025, forecast_years=1, spec=wf3.CHOSEN, start_month=5)
    base = v3.predict_years(full, **kw)
    assert "ajuste_nivel" not in base.attrs
    adj = v3.predict_years(full, level_spec=SPEC, **kw)
    assert adj.attrs["ajuste_nivel"]["aplicado"] and adj.attrs["ajuste_nivel"]["origen_referencia"] == "2025-10"
    obs = adj["observado"].to_numpy()
    np.testing.assert_allclose(adj.loc[obs, "prediccion"], base.loc[obs, "prediccion"])
    r = adj.loc[~obs, "prediccion"].to_numpy() / base.loc[~obs, "prediccion"].to_numpy()
    assert (np.abs(np.log(r)) <= SPEC.cap + 1e-12).all()
    # cero fuga: alterar la demanda de meses >= start (2026-05 en adelante) no cambia nada
    mod = full.copy()
    mod.loc[(mod["year"] == 2026) & (mod["month"] >= 5), "demand_value"] *= 1.5
    adj2 = v3.predict_years(mod, level_spec=SPEC, **kw)
    np.testing.assert_allclose(adj2["prediccion"], adj["prediccion"])


# ---------------------------------------------------------------- integracion con la BD
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


def _predict(session_factory, **kw):
    from api.config.schemas import ForecastSchema
    from api.services.forecast import ForecastService

    db = session_factory()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return ForecastService().predict(ForecastSchema(user_id=1, session_id=1, **kw), db)
    finally:
        db.close()


@pytest.mark.integration_db
@pytest.mark.parametrize("mid", [1, 2, 3, 4])
def test_flag_apagado_respuesta_identica_y_sin_campo(session_factory, monkeypatch, mid):
    kw = dict(id=mid, start_date="2026-05-01", end_date="2026-12-31")
    monkeypatch.delenv("LEVEL_ADAPT", raising=False)
    a = _predict(session_factory, **kw)
    monkeypatch.setenv("LEVEL_ADAPT", "0")
    b = _predict(session_factory, **kw)
    assert "ajuste_nivel" not in a and a == b


@pytest.mark.integration_db
def test_flag_apagado_ids_1_2_3_igual_a_la_inferencia_directa(session_factory, monkeypatch):
    from api.services import forecast as fcm
    from api.services import forecast_rfxgb, forecast_sarimax
    from api.services.forecast_history import load_clean_history_session

    monkeypatch.delenv("LEVEL_ADAPT", raising=False)
    db = session_factory()
    try:
        hist = load_clean_history_session(db, None).serie
    finally:
        db.close()
    start, end = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-12-01")
    for mid in (1, 2):
        fc, _, info = fcm.predict_v2_full_scale(mid, hist, start, end)
        art = forecast_rfxgb.load_artifact(mid)
        if pd.Timestamp(art["train_last_month"] + "-01") >= pd.Timestamp("2026-05-01"):
            art = forecast_rfxgb.refit_from_artifact_cached(mid, hist[hist.index < start], art)
        ref = forecast_rfxgb.predict_monthly(mid, hist, start, end, artifact=art)
        np.testing.assert_allclose(fc.to_numpy(), ref["demanda_mercado_completo_kwh"].to_numpy())
        assert "ajuste_nivel" not in info
    fc3, _, info3 = fcm.predict_v2_full_scale(3, hist, start, end)
    np.testing.assert_allclose(fc3.to_numpy(), forecast_sarimax.predict_monthly(hist, start, end)["prediccion"].to_numpy())
    assert "ajuste_nivel" not in info3


@pytest.mark.integration_db
@pytest.mark.parametrize("mid", [1, 2, 4])
def test_flag_encendido_ajusta_y_expone_ajuste_nivel(session_factory, monkeypatch, mid):
    kw = dict(id=mid, start_date="2026-05-01", end_date="2026-12-31")
    monkeypatch.setenv("LEVEL_ADAPT", "0")
    off = _predict(session_factory, **kw)
    monkeypatch.setenv("LEVEL_ADAPT", "1")
    on = _predict(session_factory, **kw)
    assert "message" not in on, on.get("message")
    aj = on["ajuste_nivel"]
    assert aj["aplicado"] and aj["meses_residuo"] >= 3 and aj["origen_referencia"] == "2025-10"
    # contrato intacto: mismas claves (+ ajuste_nivel) y fechas; solo cambian los valores pronosticados
    assert set(off) | {"ajuste_nivel"} == set(on) and on["monthly_predictions"]["fechas"] == off["monthly_predictions"]["fechas"]
    r = np.asarray(on["monthly_predictions"]["valores_pronostico_crudos"]) / np.asarray(off["monthly_predictions"]["valores_pronostico_crudos"])
    assert (np.abs(np.log(r)) <= forecast_level.load_config()[0].cap + 1e-9).all()
    assert r[0] == pytest.approx(aj["factor_primer_mes"], rel=1e-6)
    assert abs(np.log(r[-1])) <= abs(np.log(r[0])) + 1e-12
    json.dumps(on, allow_nan=False)


@pytest.mark.integration_db
def test_flag_encendido_id3_no_cambia(session_factory, monkeypatch):
    kw = dict(id=3, start_date="2026-05-01", end_date="2026-12-31")
    monkeypatch.setenv("LEVEL_ADAPT", "0")
    off = _predict(session_factory, **kw)
    monkeypatch.setenv("LEVEL_ADAPT", "1")
    assert _predict(session_factory, **kw) == off
