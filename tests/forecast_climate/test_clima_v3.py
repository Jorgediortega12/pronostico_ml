"""Tests del modelo id=4 v3 (clima+ONI, base mensual con nivel reciente).

No requieren la BD: usan ``training/forecast_climate/artifacts/dataset_mensual.csv`` y
``api/models/climate_model/climate_v3.json``. Los tests con datos sinteticos fijan las propiedades del metodo
(recuperacion de beta/deriva, cero fuga, semantica de ``start_month``, orden de escenarios ENSO).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from training.forecast_climate import clima_v3 as v3  # noqa: E402

ARTIFACTS = BACKEND_ROOT / "training" / "forecast_climate" / "artifacts"
MODEL_DIR = BACKEND_ROOT / "api" / "models" / "climate_model"

needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "dataset_mensual.csv").exists() or not (MODEL_DIR / "climate_v3.json").exists(),
    reason="dataset/artefacto climate_v3.json no disponibles",
)


# --------------------------------------------------------------------------------------------------
# Datos sinteticos: y = a + g*t + estacionalidad + beta*ONI (sin ruido)
# --------------------------------------------------------------------------------------------------
def _synthetic(n_years=14, g=0.05, beta=0.02, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2011-01-01", periods=12 * n_years, freq="MS")
    t = np.arange(len(idx)) / 12.0
    season = 0.08 * np.sin(2 * np.pi * (idx.month - 1) / 12) + 0.03 * np.cos(4 * np.pi * (idx.month - 1) / 12)
    # ONI "lento" con oscilacion de ~4 anios (varianza suficiente para identificar beta)
    oni = 1.2 * np.sin(2 * np.pi * t / 4.0) + 0.3 * np.sin(2 * np.pi * t / 1.7)
    y = 20.0 + g * t + season + beta * oni + rng.normal(0, noise, len(idx))
    return pd.DataFrame(
        {"year": idx.year, "month": idx.month, "demand_value": np.exp(y), "oni_actual": oni}
    )


SPEC = v3.V3Spec(family="B", window=60, slope="fixed", k_recent=72, w_recent=0.5, beta_shrink=1.0, oni_lag=0)


def test_estimate_beta_recupera_sensibilidad_sintetica():
    df = _synthetic(beta=0.02)
    y = pd.Series(np.log(df["demand_value"].to_numpy()), index=pd.date_range("2011-01-01", periods=len(df), freq="MS"))
    oni = pd.Series(df["oni_actual"].to_numpy(), index=y.index)
    assert v3.estimate_beta(y, oni, SPEC) == pytest.approx(0.02, abs=1e-6)
    # beta_mode none -> 0; signo fisico: beta negativo se trunca a 0
    assert v3.estimate_beta(y, oni, v3.V3Spec(beta_mode="none")) == 0.0
    y_neg = pd.Series(np.log(_synthetic(beta=-0.02)["demand_value"].to_numpy()), index=y.index)
    assert v3.estimate_beta(y_neg, oni, SPEC) == 0.0


def test_prediccion_sintetica_exacta_sin_ruido():
    """Con datos sin ruido que cumplen el modelo, v3 pronostica exactamente (nivel+estacionalidad+deriva+ONI)."""
    cut = 12 * 11  # origen = dic del anio 11
    # El ONI "proyectado" no coincide con el ONI sintetico futuro: se verifica con beta=0 (sin dependencia del ONI).
    spec = v3.V3Spec(family="B", window=48, slope="fixed", k_recent=36, w_recent=1.0, beta_mode="none")
    flat = _synthetic(n_years=14, beta=0.0)
    pred = v3.predict_v3(flat.iloc[:cut], 24, spec)
    assert np.allclose(pred.to_numpy(), flat.iloc[cut:cut + 24]["demand_value"].to_numpy(), rtol=1e-9)


def test_cero_fuga_el_futuro_no_cambia_el_pronostico():
    """Pronosticar con el historico completo recortado en el origen == con un historico mas largo recortado."""
    full = _synthetic(n_years=14, noise=0.01, seed=3)
    origin_rows = 12 * 10
    base = v3.predict_v3(full.iloc[:origin_rows], 12, SPEC)
    # Alterar TODO lo posterior al origen no puede mover el pronostico (predict_v3 solo recibe <= origen).
    tampered = full.copy()
    tampered.loc[origin_rows:, "demand_value"] *= 5.0
    tampered.loc[origin_rows:, "oni_actual"] += 3.0
    again = v3.predict_v3(tampered.iloc[:origin_rows], 12, SPEC)
    pd.testing.assert_series_equal(base, again)
    # predict_years recorta por (cutoff_year, start_month): tampered completo da lo mismo que full completo
    cy = int(full["year"].iloc[origin_rows - 1])
    a = v3.predict_years(full, cy, 1, SPEC, start_month=1)
    b = v3.predict_years(tampered, cy, 1, SPEC, start_month=1)
    pd.testing.assert_frame_equal(a, b)


def test_start_month_usa_solo_meses_previos_y_devuelve_observados():
    full = _synthetic(n_years=14, noise=0.01, seed=5)
    cy = 2020
    r1 = v3.predict_years(full, cy, 2, SPEC, start_month=1)
    r6 = v3.predict_years(full, cy, 2, SPEC, start_month=6)
    assert len(r1) == len(r6) == 24
    assert not r1["observado"].any()
    assert r6["observado"].tolist()[:5] == [True] * 5 and not any(r6["observado"].tolist()[5:])
    obs = full[(full["year"] == cy + 1) & (full["month"] <= 5)]["demand_value"].to_numpy()
    assert np.allclose(r6["prediccion"].to_numpy()[:5], obs)
    # el pronostico cambia al conocer 5 meses mas (el nivel se re-ancla)
    assert not np.allclose(r1["prediccion"].to_numpy()[5:], r6["prediccion"].to_numpy()[5:])
    # fechas: inicio de mes, consecutivas
    assert (r6["fecha"].dt.day == 1).all() and r6["fecha"].is_monotonic_increasing


def test_historico_insuficiente_y_horizonte_cubierto_dan_error_claro():
    full = _synthetic(n_years=14)
    with pytest.raises(ValueError, match="histórico suficiente"):
        v3.predict_years(full, 2012, 1, SPEC)
    with pytest.raises(ValueError):
        v3.predict_years(full, 2020, 1, SPEC, start_month=13)


def test_covid_no_contamina_beta_ni_deriva():
    """Los pares yoy de COVID (2020-03..2022-02) se excluyen: un shock de demanda en esa ventana no mueve beta/g."""
    df = _synthetic(n_years=14, beta=0.02)
    idx = pd.date_range("2011-01-01", periods=len(df), freq="MS")
    shock = df.copy()
    m = (idx >= "2020-03-01") & (idx <= "2021-02-01")
    shock.loc[m, "demand_value"] *= 0.6
    y0 = pd.Series(np.log(df["demand_value"].to_numpy()), index=idx)
    y1 = pd.Series(np.log(shock["demand_value"].to_numpy()), index=idx)
    oni = pd.Series(df["oni_actual"].to_numpy(), index=idx)
    assert v3.estimate_beta(y1, oni, SPEC) == pytest.approx(v3.estimate_beta(y0, oni, SPEC), abs=1e-9)
    b = v3.estimate_beta(y0, oni, SPEC)
    assert v3.estimate_drift(y1, oni, b, SPEC) == pytest.approx(v3.estimate_drift(y0, oni, b, SPEC), abs=1e-9)


# --------------------------------------------------------------------------------------------------
# Con datos y artefactos reales
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hist():
    return pd.read_csv(ARTIFACTS / "dataset_mensual.csv")


@pytest.fixture(scope="module")
def predictor():
    from api.services.forecast import DemandPredictor

    return DemandPredictor(model_path="api/models/climate_model/")


@needs_artifacts
def test_artefacto_v3_y_carga_en_predictor(predictor):
    info = json.loads((MODEL_DIR / "climate_v3.json").read_text(encoding="utf-8"))
    assert info["version"] == "v3"
    assert predictor.climate_v3_spec == v3.spec_from_dict(info["spec"])
    assert predictor.get_mape()["monthly"] == pytest.approx(info["metricas_walkforward"]["monthly"]["mape"])
    # respaldo v2 presente (rollback) y artefactos v2 intactos
    for f in ("annual_model.pkl", "annual_scaler_x.pkl", "annual_scaler_y.pkl", "model_info.pkl", "metrics.pkl"):
        assert (MODEL_DIR / "v2_backup" / f).exists()
    # beta fisico y plausible (0 = ENSO sin efecto; >5 % por unidad ONI seria implausible)
    assert 0.005 < info["beta_oni_por_unidad_estimado_al_ultimo_mes"] < 0.05


@needs_artifacts
def test_enso_phase_oni_coincide_con_forecast_py():
    from api.services.forecast import ENSO_PHASE_ONI

    assert v3.ENSO_PHASE_ONI == ENSO_PHASE_ONI


@needs_artifacts
def test_contrato_de_salida_v3(predictor):
    ann, res = predictor.predict_future_climate(forecast_years=2, cutoff_year=2025, model_version="v3")
    assert isinstance(ann, np.ndarray) and ann.shape == (2, 1)
    assert {"fecha", "prediccion", "prediccion_cruda", "prediccion_ajustada"} <= set(res.columns)
    assert len(res) == 24 and (res["fecha"].dt.day == 1).all() and res["fecha"].is_monotonic_increasing
    assert list(res["fecha"].dt.year.unique()) == [2026, 2027]
    for i, y in enumerate((2026, 2027)):
        assert res.loc[res["fecha"].dt.year == y, "prediccion"].sum() == pytest.approx(ann[i, 0], rel=1e-9)
    assert (res["prediccion"] > 0).all()
    with pytest.raises(ValueError):
        predictor.predict_future_climate(forecast_years=1, cutoff_year=2025, model_version="v3", atypical_years=[2020])


@needs_artifacts
def test_paridad_api_vs_modelo_y_default_sigue_siendo_v2(predictor, hist):
    """La API (predict_future_climate v3) usa exactamente clima_v3.predict_v3 sobre el historico < start."""
    ann, res = predictor.predict_future_climate(forecast_years=1, cutoff_year=2024, model_version="v3")
    h = hist[hist["year"] <= 2024]
    direct = v3.predict_v3(h, 12, predictor.climate_v3_spec)
    assert np.allclose(res["prediccion"].to_numpy(), direct.to_numpy(), rtol=1e-12)
    # sin model_version el comportamiento es el v2 (anual + reparto): total != v3
    ann_v2, _ = predictor.predict_future_climate(forecast_years=1, cutoff_year=2024)
    assert not np.isclose(ann_v2[0, 0], ann[0, 0], rtol=1e-6)


@needs_artifacts
def test_start_month_en_la_api_usa_meses_observados(predictor, hist):
    ann, res = predictor.predict_future_climate(forecast_years=1, cutoff_year=2024, model_version="v3", start_month=7)
    obs = hist[(hist["year"] == 2025) & (hist["month"] <= 6)]["demand_value"].to_numpy()
    assert np.allclose(res["prediccion"].to_numpy()[:6], obs)
    assert res["prediccion"].sum() == pytest.approx(ann[0, 0])


@needs_artifacts
def test_escenarios_enso_ordenados_y_efecto_anual_no_trivial(predictor):
    tot = {}
    for fase in ("NIÑA", "NORMAL", "NIÑO"):
        ann, _ = predictor.predict_future_climate(
            forecast_years=1, cutoff_year=2025, model_version="v3",
            enso_phase_by_month={f"2026-{m:02d}": fase for m in range(1, 13)},
        )
        tot[fase] = float(ann[0, 0])
    auto, _ = predictor.predict_future_climate(forecast_years=1, cutoff_year=2025, model_version="v3")
    assert tot["NIÑA"] < tot["NORMAL"] < tot["NIÑO"]
    spread = tot["NIÑO"] / tot["NIÑA"] - 1
    # Umbral del plan clima v3: efecto Niño-Niña anual >= ~3 % (v2 daba 3.1 %).
    assert 0.03 <= spread < 0.08, f"spread Niño/Niña anual {spread:.2%}"
    assert tot["NIÑA"] <= float(auto[0, 0]) <= tot["NIÑO"]


@needs_artifacts
def test_un_mes_forzado_mueve_ese_mes_con_orden_fisico_y_no_los_demas(predictor):
    _, base = predictor.predict_future_climate(forecast_years=1, cutoff_year=2025, model_version="v3")
    for month in (2, 5, 8, 11):
        key = f"2026-{month:02d}"
        vals = {}
        for fase in ("NIÑA", "NORMAL", "NIÑO"):
            _, r = predictor.predict_future_climate(
                forecast_years=1, cutoff_year=2025, model_version="v3", enso_phase_by_month={key: fase}
            )
            vals[fase] = r.set_index("fecha")["prediccion"]
            others = r["fecha"].dt.month != month
            assert np.allclose(r.loc[others, "prediccion"].to_numpy(), base.loc[others, "prediccion"].to_numpy())
        ts = pd.Timestamp(key + "-01")
        assert vals["NIÑA"][ts] < vals["NORMAL"][ts] < vals["NIÑO"][ts]
        spread = vals["NIÑO"][ts] / vals["NIÑA"][ts] - 1
        assert spread >= 0.03, f"{key}: spread de un mes forzado {spread:.2%}"


@needs_artifacts
def test_forma_y_nivel_razonables(predictor, hist):
    ann, res = predictor.predict_future_climate(forecast_years=3, cutoff_year=2025, model_version="v3")
    last_real = float(hist[hist["year"] == 2025]["demand_value"].sum())
    g = [ann[0, 0] / last_real - 1] + [ann[i, 0] / ann[i - 1, 0] - 1 for i in (1, 2)]
    assert all(0.0 < x < 0.10 for x in g), g
    # forma estacional: correlacion con el perfil de 2023-2025 y mismo pico/valle
    prof = hist[hist["year"].between(2023, 2025)].groupby("month")["demand_value"].mean()
    prof = prof / prof.sum()
    for y in (2027, 2028):
        sub = res[res["fecha"].dt.year == y]
        m = pd.Series(sub["prediccion"].to_numpy(), index=sub["fecha"].dt.month.to_numpy())
        sh = m / m.sum()
        assert np.corrcoef(sh.to_numpy(), prof.to_numpy())[0, 1] > 0.95
        assert int(sh.idxmax()) == int(prof.idxmax()) and int(sh.idxmin()) == int(prof.idxmin())
    # sin saltos mensuales extremos fuera de lo historico
    r = res["prediccion"].pct_change().dropna()
    assert r.max() < 0.20 and r.min() > -0.14


@needs_artifacts
def test_walkforward_v3_usa_el_mismo_codigo_que_la_inferencia(hist):
    """El fit_predict del walk-forward y predict_years (API) dan el mismo pronostico en el mismo origen."""
    from training.forecast_climate import walkforward_clima_v3 as w

    origin = pd.Timestamp("2024-12-01")
    s = w.series_of(hist)
    train = s[s.index <= origin].copy()
    train.attrs["origen"] = origin
    wf = w.make_fit_predict(hist, SPEC)(train, 12)
    api = v3.predict_years(hist, 2024, 1, SPEC)["prediccion"].to_numpy()
    assert np.allclose(wf.to_numpy(), api, rtol=1e-12)
    # y el spec congelado de la evaluacion es el del artefacto productivo
    assert w.CHOSEN == v3.spec_from_dict(json.loads((MODEL_DIR / "climate_v3.json").read_text(encoding="utf-8"))["spec"])
