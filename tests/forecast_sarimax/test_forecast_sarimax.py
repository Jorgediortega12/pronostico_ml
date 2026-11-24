"""Tests del módulo de inferencia id=3 (api/services/forecast_sarimax.py).

Cubren: fechas rotuladas desde start (2025/2026/2027), cero fuga (datos >= start no influyen), re-anclaje del
estado (start distinto ⇒ valores distintos, no la misma curva desplazada), refit automático en hindcast,
exclusión post-quiebre, y plausibilidad del artefacto real.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.services import forecast_sarimax as fs
from api.services.forecast_sarimax import Artifact, ModelSpec, predict_monthly

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------------
# Fixtures sintéticas
# --------------------------------------------------------------------------------------------------
def _synthetic(end="2026-08", seed=0) -> pd.Series:
    idx = pd.date_range("2011-01-01", end, freq="MS")
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    season = 0.06 * np.sin(2 * np.pi * (idx.month - 3) / 12)
    log_y = np.log(4e8) + 0.004 * t + season + rng.normal(0, 0.015, len(idx))
    return pd.Series(np.exp(log_y), index=idx)


@pytest.fixture(scope="module")
def hist():
    return _synthetic()


@pytest.fixture(scope="module")
def artifact(hist):
    spec = ModelSpec(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), transform="log_daily", covid="nan")
    train = hist.loc[:"2026-08"]
    res = fs.fit_spec(train, spec)
    return Artifact(
        components=[{"spec": spec.to_dict(), "params": res.params.tolist(), "param_names": list(res.model.param_names)}],
        weights=[1.0],
        train_end="2026-08",
    )


# --------------------------------------------------------------------------------------------------
# start_date respetado
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("start,end,n", [("2025-01-01", "2025-12-31", 12), ("2026-01-01", "2027-12-31", 24),
                                          ("2027-01-01", "2027-12-31", 12), ("2026-03-15", "2026-05-02", 3)])
def test_fechas_desde_start(hist, artifact, start, end, n):
    out = predict_monthly(hist, start, end, artifact=artifact)
    assert len(out) == n
    assert out["fecha"].iloc[0] == pd.Timestamp(start).to_period("M").to_timestamp()
    assert out["fecha"].iloc[-1] == pd.Timestamp(end).to_period("M").to_timestamp()
    assert out["fecha"].is_monotonic_increasing
    assert np.isfinite(out["prediccion"]).all() and (out["prediccion"] > 0).all()


def test_start_distinto_no_es_misma_curva_desplazada(hist, artifact):
    """Regresión del bug original: start=2025 devolvía los valores de 2026 rotulados como 2025."""
    a = predict_monthly(hist, "2025-01-01", "2025-12-31", artifact=artifact)["prediccion"].to_numpy()
    b = predict_monthly(hist, "2026-01-01", "2026-12-31", artifact=artifact)["prediccion"].to_numpy()
    assert not np.allclose(a, b)
    # el hindcast 2025 debe estar cerca del real 2025 (serie sintética suave), no del nivel 2026
    real25 = hist.loc["2025"].to_numpy()
    assert np.mean(np.abs(a / real25 - 1)) < 0.05


def test_ultimo_real_y_refit_auto(hist, artifact):
    o25 = predict_monthly(hist, "2025-01-01", "2025-06-30", artifact=artifact)
    assert o25.attrs["ultimo_real"] == "2024-12"
    assert o25.attrs["refit"] is True  # artefacto vio hasta 2026-08 >= start → re-estima (sin fuga de parámetros)
    o27 = predict_monthly(hist, "2027-01-01", "2027-06-30", artifact=artifact)
    assert o27.attrs["ultimo_real"] == "2026-08"
    assert o27.attrs["refit"] is False


# --------------------------------------------------------------------------------------------------
# Cero fuga
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("refit", ["auto", False, True])
@pytest.mark.parametrize("start", ["2024-07-01", "2026-01-01"])
def test_datos_desde_start_no_influyen(hist, artifact, refit, start):
    base = predict_monthly(hist, start, "2027-12-31", artifact=artifact, refit=refit)
    tampered = hist.copy()
    tampered[tampered.index >= pd.Timestamp(start)] *= 3.0  # cualquier cosa >= start
    alt = predict_monthly(tampered, start, "2027-12-31", artifact=artifact, refit=refit)
    pd.testing.assert_frame_equal(base, alt)


def test_exclude_from_corta_post_quiebre(hist, artifact):
    h = hist.copy()
    extra = pd.Series([0.7 * h.iloc[-12], 0.7 * h.iloc[-11]],
                      index=pd.to_datetime(["2026-09-01", "2026-10-01"]))
    h2 = pd.concat([h, extra])
    a = predict_monthly(h, "2027-01-01", "2027-12-31", artifact=artifact)
    b = predict_monthly(h2, "2027-01-01", "2027-12-31", artifact=artifact, exclude_from="2026-09-01")
    pd.testing.assert_frame_equal(a, b)
    c = predict_monthly(h2, "2027-01-01", "2027-12-31", artifact=artifact)  # sin exclusión sí cambia
    assert not np.allclose(a["prediccion"], c["prediccion"])


def test_historia_insuficiente(hist, artifact):
    with pytest.raises(ValueError):
        predict_monthly(hist, "2012-06-01", "2012-12-31", artifact=artifact)
    with pytest.raises(ValueError):
        predict_monthly(hist, "2026-05-01", "2026-01-01", artifact=artifact)


def test_apply_params_igual_a_filtrar_con_parametros_del_artefacto(hist, artifact):
    """refit=False con historia = train del artefacto reproduce el forecast del ajuste original."""
    spec = artifact.specs()[0]
    res = fs.fit_spec(hist.loc[:"2026-08"], spec)
    direct = fs.forecast_spec(res, spec, pd.Timestamp("2026-08-01"), 16).loc["2026-09":"2027-12"]
    out = predict_monthly(hist, "2026-09-01", "2027-12-31", artifact=artifact, refit=False)
    np.testing.assert_allclose(out["prediccion"].to_numpy(), direct.to_numpy(), rtol=1e-6)


# --------------------------------------------------------------------------------------------------
# Calendario
# --------------------------------------------------------------------------------------------------
def test_calendario_determinista_y_acotado():
    idx = pd.date_range("2024-01-01", "2027-12-01", freq="MS")
    X = fs.calendar_features(idx, ["frac_habiles", "frac_domfest", "bisiesto_feb"])
    assert X.notna().all().all()
    assert ((X[["frac_habiles", "frac_domfest"]] > 0) & (X[["frac_habiles", "frac_domfest"]] < 1)).all().all()
    assert X.loc["2024-02-01", "bisiesto_feb"] == 1 and X.loc["2025-02-01", "bisiesto_feb"] == 0
    pd.testing.assert_frame_equal(X, fs.calendar_features(idx, ["frac_habiles", "frac_domfest", "bisiesto_feb"]))


def test_spec_roundtrip():
    s = ModelSpec(order=(2, 1, 0), seasonal_order=(0, 1, 1, 12), exog=("frac_habiles",), covid="dummy")
    assert ModelSpec.from_dict(s.to_dict()) == s


# --------------------------------------------------------------------------------------------------
# Artefacto real
# --------------------------------------------------------------------------------------------------
def test_artefacto_real_carga_y_es_plausible(hist):
    path = fs.MODEL_DIR / fs.ARTIFACT_FILE
    if not path.exists():
        pytest.skip("artefacto no generado")
    art = fs.load_artifact()
    assert art.components and len(art.weights) == len(art.components)
    for spec in art.specs():
        assert spec.kind in {"sarima", "ets"}
    # nivel de la serie sintética ≠ real, pero la forma (estacionalidad/crecimiento) debe ser sensata
    out = predict_monthly(hist, "2026-09-01", "2027-12-31", artifact=art, refit=False)
    ratio = out.set_index("fecha")["prediccion"] / hist.iloc[-12:].mean()
    assert ratio.between(0.8, 1.3).all()


@pytest.mark.integration_db
def test_bd_real_plausible_2026_2027():
    try:
        from training.common.data import load_monthly_demand, target_series

        monthly = load_monthly_demand(None, impute_incomplete=True)
        mb = monthly.attrs.get("market_break")
        history, first_post = target_series(monthly), (mb.first_post_month if mb is not None else None)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BD no disponible: {type(exc).__name__}")
    if not (fs.MODEL_DIR / fs.ARTIFACT_FILE).exists():
        pytest.skip("artefacto no generado")
    for start, end in [("2025-01-01", "2025-12-31"), ("2026-01-01", "2027-12-31"), ("2027-01-01", "2027-12-31")]:
        out = predict_monthly(history, start, end, exclude_from=first_post).set_index("fecha")["prediccion"]
        assert out.index[0] == pd.Timestamp(start)
        # cada mes dentro de ±25 % del mismo mes del último año pre-quiebre observado
        ref = history[history.index < (first_post or pd.Timestamp(start))]
        for m, v in out.items():
            same = ref[ref.index.month == m.month].iloc[-1]
            assert 0.75 < v / same < 1.30, (m, v, same)
    # hindcast 2025 sin fuga: error medio razonable (< 8 %) contra el real 2025
    h25 = predict_monthly(history, "2025-01-01", "2025-12-31").set_index("fecha")["prediccion"]
    real = history.loc["2025"]
    assert float(np.mean(np.abs(h25 / real - 1))) < 0.08
