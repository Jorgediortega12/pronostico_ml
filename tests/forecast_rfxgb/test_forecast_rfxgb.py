"""Tests del módulo de inferencia v2 RF/XGB (``api/services/forecast_rfxgb.py``).

No requieren BD: usan una serie sintética (crecimiento + estacionalidad + calendario + ruido). Los tests de
los artefactos reales se omiten si no existen.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from api.services import forecast_rfxgb as F


def _synthetic(start="2011-01-01", end="2026-10-01", seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS")
    t = np.arange(len(idx))
    season = 0.06 * np.sin(2 * np.pi * (idx.month - 3) / 12)
    lab = np.array([F.calendar_counts(d.year, d.month)[1] for d in idx])
    y = 5.0e8 * np.exp(0.004 * t + season + 0.004 * (lab - 20) + rng.normal(0, 0.01, len(idx)))
    return pd.Series(y, index=idx)


FAST = {
    1: dict(n_estimators=40, min_samples_leaf=10, max_features=0.5),
    2: dict(n_estimators=60, max_depth=2, min_child_weight=10),
}


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return _synthetic()


@pytest.fixture(scope="module", params=[1, 2])
def artifact(request, series):
    mid = request.param
    return F.fit_model(series[series.index <= "2025-12-01"], mid,
                       dict(offset="naive_drift", params=FAST[mid], h_train=24))


# ---------------------------------------------------------------------------------------------- fuga
def test_features_al_origen_no_dependen_del_futuro(series):
    o = pd.Timestamp("2022-06-01")
    trunc = F.build_design(series[series.index <= o], 24, origins=[o])
    full = F.build_design(series, 24, origins=[o])
    cols = F.FEATURES + ["off_naive_drift", "off_nivel_estacional"]
    pd.testing.assert_frame_equal(trunc[cols], full[cols])
    # y del objetivo no se filtra a las features: en la versión truncada es NaN.
    assert trunc["y"].isna().all() and full["y"].notna().all()


def test_futuro_alterado_no_cambia_el_pronostico(series, artifact):
    start = "2024-01-01"
    base = F.predict_monthly(artifact["model_id"], series, start, "2025-06-30", artifact=artifact)
    corrupt = series.copy()
    corrupt[corrupt.index >= start] *= 3.0  # cualquier dato >= start debe ignorarse
    alt = F.predict_monthly(artifact["model_id"], corrupt, start, "2025-06-30", artifact=artifact)
    pd.testing.assert_frame_equal(base, alt)


def test_training_rows_solo_targets_observados(series):
    o = pd.Timestamp("2020-12-01")
    d = F.training_rows(F.build_design(series[series.index <= o], 36))
    assert (d["objetivo"] <= o).all()
    assert (d["origen"] < d["objetivo"]).all()


def test_offset_igual_al_naive_drift_del_harness(series):
    from training.common.walkforward import seasonal_naive_drift

    o = pd.Timestamp("2023-12-01")
    tr = series[series.index <= o].copy()
    tr.attrs = {"origen": o}
    d = F.build_design(tr, 24, origins=[o])
    np.testing.assert_allclose(np.exp(d["off_naive_drift"].to_numpy()), seasonal_naive_drift(tr, 24), rtol=1e-12)


# ---------------------------------------------------------------------------------------------- fechas
def test_fechas_rotuladas_desde_start(series, artifact):
    out = F.predict_monthly(artifact["model_id"], series, "2026-03-15", "2026-12-31", artifact=artifact)
    assert list(out["fecha"]) == list(pd.date_range("2026-03-01", "2026-12-01", freq="MS"))
    # último dato < start = 2026-02 -> h empieza en 1
    assert out["h"].iloc[0] == 1


def test_start_2027_con_anio_parcial(series, artifact):
    hist = series[series.index <= "2026-10-01"]  # 2026 incompleto (sin nov-dic)
    out = F.predict_monthly(artifact["model_id"], hist, "2027-01-01", "2027-12-31", artifact=artifact)
    assert list(out["fecha"]) == list(pd.date_range("2027-01-01", periods=12, freq="MS"))
    assert out["h"].tolist() == list(range(3, 15))
    ref = hist.iloc[-12:].mean()
    assert out["demanda_kwh"].between(0.85 * ref, 1.25 * ref).all()
    assert out["demanda_kwh"].notna().all()


def test_quiebre_de_mercado_se_corta_y_factor(series, artifact):
    brk = pd.Timestamp("2026-09-01")
    hist = series.copy()
    hist[hist.index >= brk] *= 0.7
    a = F.predict_monthly(artifact["model_id"], hist, "2027-01-01", "2027-06-30", market_break_month=brk,
                          artifact=artifact)
    b = F.predict_monthly(artifact["model_id"], series[series.index < brk], "2027-01-01", "2027-06-30",
                          artifact=artifact)
    pd.testing.assert_frame_equal(a, b)
    c = F.predict_monthly(artifact["model_id"], hist, "2026-09-01", "2027-06-30", market_break_month=brk,
                          post_break_factor=0.7, artifact=artifact)
    np.testing.assert_allclose(c["demanda_kwh"], c["demanda_mercado_completo_kwh"] * 0.7)
    assert c.attrs["escala"] == "bd_post_quiebre"


def test_horizonte_mayor_al_entrenado(series, artifact):
    out = F.predict_monthly(artifact["model_id"], series[series.index <= "2025-12-01"], "2026-01-01",
                            "2029-12-31", artifact=artifact)
    assert len(out) == 48 and np.isfinite(out["demanda_kwh"]).all()
    anual = out.groupby(out["fecha"].dt.year)["demanda_kwh"].sum()
    assert (anual.diff().dropna() > 0).all()  # serie sintética creciente


# ---------------------------------------------------------------------------------------------- valores
def test_rango_plausible_y_precision_sintetica(series, artifact):
    hist = series[series.index <= "2024-12-01"]
    out = F.predict_monthly(artifact["model_id"], hist, "2025-01-01", "2025-12-31", artifact=artifact)
    real = series.reindex(out["fecha"]).to_numpy()
    mape = np.mean(np.abs(out["demanda_kwh"].to_numpy() / real - 1)) * 100
    assert mape < 5.0


def test_entradas_invalidas(series, artifact):
    mid = artifact["model_id"]
    with pytest.raises(ValueError):
        F.predict_monthly(mid, series, "2026-05-01", "2026-01-01", artifact=artifact)
    with pytest.raises(ValueError):
        F.predict_monthly(mid, series.iloc[:30], "2014-01-01", "2014-12-01", artifact=artifact)
    with pytest.raises(ValueError):
        F.predict_monthly(mid, series, "2026-01-01", "2040-01-01", artifact=artifact)
    with pytest.raises(ValueError):
        F.predict_monthly(mid, series, "2026-01-01", "2026-12-01", post_break_factor=1.5, artifact=artifact)
    with pytest.raises(ValueError):
        F.predict_monthly(9, series, "2026-01-01", "2026-12-01")


def test_acepta_dataframe(series, artifact):
    df = pd.DataFrame({"fecha": series.index, "total": series.to_numpy()})
    a = F.predict_monthly(artifact["model_id"], df, "2026-01-01", "2026-06-01", artifact=artifact)
    b = F.predict_monthly(artifact["model_id"], series, "2026-01-01", "2026-06-01", artifact=artifact)
    pd.testing.assert_frame_equal(a, b)


def test_calendario_colombia():
    # Enero 2026: 31 días, 4 domingos + festivos 1 y 12 (lunes) -> 6 domfest
    dias, lab, sab, domfest = F.calendar_counts(2026, 1)
    assert dias == 31 and dias == lab + sab + domfest
    assert domfest == 6


# ---------------------------------------------------------------------------------------------- artefactos reales
@pytest.mark.parametrize("mid", [1, 2])
def test_artefactos_reales(mid, series):
    try:
        art = F.load_artifact(mid)
    except FileNotFoundError:
        pytest.skip("artefactos v2 no entrenados")
    assert art["model_id"] == mid and art["features"] == F.FEATURES
    meta = json.loads((F.MODEL_DIRS[mid] / F.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta.get("model_name") == F.MODEL_NAMES[mid]
    # Escala real ~6e8-9e8 kWh/mes: la serie sintética es otra escala, pero el modelo trabaja en log
    # relativo, así que el pronóstico debe seguir a la serie que recibe.
    out = F.predict_monthly(mid, series, "2027-01-01", "2027-12-31")
    ref = series.iloc[-12:].mean()
    assert out["demanda_kwh"].between(0.8 * ref, 1.3 * ref).all()


def test_rf_y_xgb_son_algoritmos_distintos():
    try:
        a1, a2 = F.load_artifact(1), F.load_artifact(2)
    except FileNotFoundError:
        pytest.skip("artefactos v2 no entrenados")
    assert type(a1["estimator"]).__name__ == "RandomForestRegressor"
    assert type(a2["estimator"]).__name__ == "XGBRegressor"
