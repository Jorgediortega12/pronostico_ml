"""Tests de las correcciones de la ronda 1 de revisión (docs/work/06_correcciones/REPORTE.md). Sin BD."""
from __future__ import annotations

import threading
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from api.services import forecast_rfxgb as F
from api.services import forecast_sarimax as fs
from api.services.forecast import ForecastService, run_v2_forecast, v2_annual_totals, v2_internal_gaps
from api.services.forecast_adjustments import apply_ipc_adjustment
from api.services.forecast_cache import SingleFlightLRU, series_digest
from api.services.forecast_history import DivisionMercado
from training.common.data import MarketBreak


def _synthetic(start="2011-01-01", end="2025-06-01", seed=0) -> pd.Series:
    idx = pd.date_range(start, end, freq="MS")
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    season = 0.06 * np.sin(2 * np.pi * (idx.month - 3) / 12)
    return pd.Series(np.exp(np.log(4e8) + 0.004 * t + season + rng.normal(0, 0.015, len(idx))), index=idx)


# --------------------------------------------------------------------------------------------------
# 1. IPC con start a mitad de año: solo escala la parte pronosticada
# --------------------------------------------------------------------------------------------------
def test_ipc_start_mitad_de_anio_suma_mensual_cuadra_con_anual():
    hist = _synthetic(end="2025-05-01")
    start_m = pd.Timestamp("2025-06-01")
    fc = pd.Series(4.5e8, index=pd.date_range("2025-06-01", "2026-12-01", freq="MS"))
    years, est, fixed = v2_annual_totals({2024: 1.0}, hist, fc, [2025, 2026], start_m)
    reales_2025 = float(hist.loc["2025"].sum())
    assert fixed[2025] == pytest.approx(reales_2025)
    assert fixed[2026] == 0.0
    mp = pd.DataFrame({"fecha": fc.index, "prediccion": fc.to_numpy()})
    y2, mp2, growth = apply_ipc_adjustment(years, mp, {2025: 10.0, 2026: 5.0}, years_to_predict=[2025, 2026],
                                           fixed_by_year=fixed, adjust_from=start_m)
    m25 = mp2[mp2["fecha"].dt.year == 2025]["prediccion"].sum()
    m26 = mp2[mp2["fecha"].dt.year == 2026]["prediccion"].sum()
    # Los reales previos al start NO se escalan; el total cuadra con reales + pronóstico ajustado.
    assert y2[2025] == pytest.approx(reales_2025 + 7 * 4.5e8 * 1.10, abs=2)
    assert y2[2025] == pytest.approx(reales_2025 + m25, abs=2)
    assert y2[2026] == pytest.approx(m26, abs=2)
    assert growth[2026] == pytest.approx((y2[2026] - y2[2025]) / y2[2025])


def test_ipc_legacy_sin_fijos_conserva_comportamiento():
    mp = pd.DataFrame({"fecha": pd.date_range("2027-01-01", periods=12, freq="MS"), "prediccion": 100.0})
    y2, mp2, _ = apply_ipc_adjustment({2026: 1000, 2027: 1200}, mp, {2027: 3.0}, years_to_predict=[2027])
    assert y2[2027] == int(1200 * 1.03)
    assert mp2["prediccion"].tolist() == pytest.approx([103.0] * 12)


# --------------------------------------------------------------------------------------------------
# 3. Caché LRU + single-flight
# --------------------------------------------------------------------------------------------------
def test_lru_real_move_to_end_y_tope():
    c = SingleFlightLRU(2)
    c.get_or_compute("a", lambda: 1)
    c.get_or_compute("b", lambda: 2)
    c.get_or_compute("a", lambda: 99)  # acierto: "a" pasa a ser la más reciente
    c.get_or_compute("c", lambda: 3)  # expulsa "b" (la menos usada), no "a"
    # Aciertos devuelven el valor cacheado; "b" expulsada se recalcula.
    assert c.get_or_compute("a", lambda: -1) == 1
    assert c.get_or_compute("c", lambda: -1) == 3
    assert c.get_or_compute("b", lambda: 22) == 22


def test_single_flight_calcula_una_vez_y_errores_no_se_cachean():
    c = SingleFlightLRU(4)
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return 42

    out = []
    ths = [threading.Thread(target=lambda: out.append(c.get_or_compute("k", slow))) for _ in range(5)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert out == [42] * 5 and len(calls) == 1

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        c.get_or_compute("e", boom)
    assert c.get_or_compute("e", lambda: 7) == 7  # el error no quedó cacheado


def test_series_digest_cambia_con_valores_e_indice():
    s = _synthetic(end="2015-01-01")
    assert series_digest(s) == series_digest(s.copy())
    assert series_digest(s) != series_digest(s * 1.0001)
    assert series_digest(s) != series_digest(s.shift(1, freq="MS"))
    assert len(series_digest(s)) == 16


def test_sarimax_refit_cacheado(monkeypatch):
    hist = _synthetic(end="2024-12-01")
    spec = fs.ModelSpec(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), transform="log_daily", covid="nan")
    art = fs.Artifact(components=[{"spec": spec.to_dict(), "params": [0.0, 0.0, 0.01]}], weights=[1.0],
                      train_end="2026-08")
    fs._REFIT_PARAMS_CACHE.clear()
    n_fit = []
    real_fit = fs.fit_spec
    monkeypatch.setattr(fs, "fit_spec", lambda h, sp: (n_fit.append(1), real_fit(h, sp))[1])
    a = fs.predict_monthly(hist, "2025-01-01", "2025-12-31", artifact=art)
    b = fs.predict_monthly(hist, "2025-01-01", "2025-12-31", artifact=art)
    assert a.attrs["refit"] and len(n_fit) == 1
    assert a["prediccion"].tolist() == b["prediccion"].tolist()


def test_rfxgb_refit_cacheado_por_historia_y_huella(monkeypatch):
    hist = _synthetic(end="2020-12-01")
    art = {"offset": "naive_drift", "params": {"n_estimators": 10}, "h_train": 12, "exclude_targets": []}
    n_fit = []
    real_fit = F.fit_model

    def counting(h, mid, cfg):
        n_fit.append(cfg["params"]["n_jobs"])
        return real_fit(h, mid, cfg)

    monkeypatch.setattr(F, "fit_model", counting)
    F._ONLINE_REFIT_CACHE.clear()
    a1 = F.refit_from_artifact_cached(1, hist, art, fingerprint="v1")
    a2 = F.refit_from_artifact_cached(1, hist, art, fingerprint="v1")
    assert a1 is a2 and n_fit == [F.ONLINE_REFIT_N_JOBS]
    F.refit_from_artifact_cached(1, hist, art, fingerprint="v2")  # artefacto distinto -> nuevo re-entreno
    assert len(n_fit) == 2


# --------------------------------------------------------------------------------------------------
# 5. Hueco interno en el año del último real
# --------------------------------------------------------------------------------------------------
def test_hueco_interno_se_rellena_con_pronostico(monkeypatch):
    hist = _synthetic(end="2025-06-01").drop(pd.Timestamp("2025-03-01"))
    gaps = v2_internal_gaps(hist[hist.index < "2025-07-01"])
    assert list(gaps) == [pd.Timestamp("2025-03-01")]

    calls = []

    def fake_predict(model_id, hist_serie, start_m, end_m, corte=None):
        h = hist_serie[hist_serie.index < start_m]
        idx = pd.date_range(h.index.max() + pd.offsets.MonthBegin(1), end_m, freq="MS")
        calls.append((h.index.max(), end_m))
        return pd.Series(4e8, index=idx), h, {"version": "v2"}

    import api.services.forecast as fmod

    monkeypatch.setattr(fmod, "predict_v2_full_scale", fake_predict)
    out = run_v2_forecast(3, hist, None, pd.Timestamp("2025-07-01"), [2025, 2026], {})
    assert out["info"]["meses_hueco_rellenados"] == ["2025-03"]
    # Se pronostica desde el primer hueco (historia hasta 2025-02).
    assert calls[1] == (pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01"))
    esperado_2025 = hist.loc["2025"].sum() + 4e8 + 6 * 4e8
    assert out["years"][2025] == pytest.approx(esperado_2025, abs=1)


def test_hueco_interno_sarimax_real_sin_error():
    hist = _synthetic(end="2025-06-01").drop(pd.Timestamp("2025-03-01"))
    out = run_v2_forecast(3, hist, None, pd.Timestamp("2025-07-01"), [2025], {})
    assert 2025 in out["years"] and np.isfinite(out["years"][2025])
    assert out["info"]["meses_hueco_rellenados"] == ["2025-03"]


# --------------------------------------------------------------------------------------------------
# 6. RF/XGB con historia corta: error claro antes de entrenar
# --------------------------------------------------------------------------------------------------
def test_rfxgb_historia_corta_error_claro():
    hist = _synthetic(start="2022-01-01", end="2024-06-01")
    cfg = {"offset": "naive_drift", "params": {"n_estimators": 5}, "h_train": 12}
    with pytest.raises(ValueError, match="historia insuficiente"):
        F.fit_model(hist, 1, cfg)
    with pytest.raises(ValueError, match="historia insuficiente"):
        F.refit_from_artifact_cached(2, hist, {**cfg, "exclude_targets": []}, fingerprint="x")


# --------------------------------------------------------------------------------------------------
# 8. Calendario heredado (get_number_of_days) cuenta festivos
# --------------------------------------------------------------------------------------------------
def test_get_number_of_days_cuenta_festivos_enero_2026():
    # 1-ene (jueves) y 12-ene (lunes, Reyes) son festivos; 4 domingos y 5 sábados.
    assert ForecastService().get_number_of_days(2026, 1) == (20, 5, 6)
    assert F.calendar_counts(2026, 1) == (31, 20, 5, 6)


# --------------------------------------------------------------------------------------------------
# 11. DivisionMercado = MarketBreak
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fl", [date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 30), date(2026, 2, 28)])
def test_division_mercado_equivale_a_market_break(fl):
    d = DivisionMercado(fecha_limite=fl, porcentaje=70.0)
    mb = MarketBreak(fecha_limite=fl, porcentaje=70.0)
    assert isinstance(d, MarketBreak)
    months = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    assert d.first_post_month == mb.first_post_month
    assert (d.is_post(months) == mb.is_post(months)).all()
    assert [d.is_post(m) for m in months] == list(mb.is_post(months))
    # Semántica de adjust_period_value: post si el último día del mes es > fecha_limite.
    esperado = [m + pd.offsets.MonthEnd(0) > pd.Timestamp(fl) for m in months]
    assert list(d.is_post(months)) == esperado
    assert d.as_dict()["factor"] == pytest.approx(0.7)
