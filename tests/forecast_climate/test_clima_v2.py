"""Tests del modelo id=4 v2 (clima+ONI): datos limpios, cero fuga, paridad train/serve y escenarios ONI.

No requieren la BD: usan los CSV regenerados en ``training/forecast_climate/artifacts/`` y los artefactos de
``api/models/climate_model/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

ARTIFACTS = BACKEND_ROOT / "training" / "forecast_climate" / "artifacts"
MODEL_DIR = BACKEND_ROOT / "api" / "models" / "climate_model"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "dataset_mensual.csv").exists() or not (MODEL_DIR / "annual_model.pkl").exists(),
    reason="datasets/artefactos del modelo climático no disponibles",
)


@pytest.fixture(scope="module")
def datasets():
    m = pd.read_csv(ARTIFACTS / "dataset_mensual.csv")
    a = pd.read_csv(ARTIFACTS / "dataset_anual.csv")
    return m, a


@pytest.fixture(scope="module")
def clima_data(datasets):
    from training.forecast_climate.walkforward_clima import ClimaData

    return ClimaData(*datasets)


@pytest.fixture(scope="module")
def prod_predictor():
    from api.services.forecast import DemandPredictor

    return DemandPredictor(model_path="api/models/climate_model/")


# --------------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------------
def test_dataset_2020q4_no_deflactado(datasets):
    m, _ = datasets
    v = m.set_index(["year", "month"])["demand_value"]
    for mes in (10, 11, 12):
        ratio_prev = v[(2020, mes)] / v[(2019, mes)]
        ratio_next = v[(2021, mes)] / v[(2020, mes)]
        assert 0.85 < ratio_prev < 1.20, (mes, ratio_prev)
        assert 0.85 < ratio_next < 1.25, (mes, ratio_next)


def test_dataset_sin_huecos_ni_nulos_en_demanda_y_oni(datasets):
    m, a = datasets
    idx = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
    assert (idx.diff().dropna().dt.days.between(28, 31)).all(), "meses no consecutivos"
    assert m["demand_value"].notna().all() and (m["demand_value"] > 0).all()
    assert m["oni_actual"].notna().all()
    assert set(m["oni_fuente"]) <= {"observado", "proyectado"}
    # La proyección solo puede estar en la cola (meses aún sin ONI publicado).
    proj = m["oni_fuente"].eq("proyectado").to_numpy()
    if proj.any():
        assert proj[np.argmax(proj):].all()
    assert (a["demand_total"] == m.groupby("year")["demand_value"].sum().reindex(a["year"]).to_numpy()).all()


def test_dataset_2021_09_imputado_en_escala(datasets):
    m, _ = datasets
    v = m.set_index(["year", "month"])["demand_value"]
    # 2021-09 tenía 28/30 días (−7 %): imputado por promedio diario, debe quedar entre ago y oct.
    assert min(v[(2021, 8)], v[(2021, 10)]) * 0.95 < v[(2021, 9)] < max(v[(2021, 8)], v[(2021, 10)]) * 1.05


# --------------------------------------------------------------------------------------------------
# Cero fuga / paridad train-serve
# --------------------------------------------------------------------------------------------------
def test_features_de_inferencia_no_ven_el_futuro(datasets):
    from training.forecast_climate.clima_oni_model import capture_inference_features

    m, a = datasets
    base = capture_inference_features(m, a, 2018)
    m2, a2 = m.copy(), a.copy()
    fut_m, fut_a = m2["year"] > 2018, a2["year"] > 2018
    m2.loc[fut_m, "demand_value"] *= 3
    m2.loc[fut_m, "oni_actual"] = 5.0
    m2.loc[fut_m, "temp_mean_mes"] += 10
    a2.loc[fut_a, "demand_total"] *= 3
    a2.loc[fut_a, "oni_mean_anio"] = 5.0
    pert = capture_inference_features(m2, a2, 2018)
    pd.testing.assert_series_equal(base, pert)


def test_training_table_usa_features_de_inferencia_con_corte_t_menos_1(datasets):
    from training.forecast_climate.clima_oni_model import build_training_table, capture_inference_features

    m, a = datasets
    tab = build_training_table(m, a, [2016], "proyectado")
    feat = capture_inference_features(m, a, 2015)
    for k in ("oni_mean_anio", "lag_1", "growth_rate_lag1", "temp_mean_anio"):
        assert tab.loc[2016, k] == pytest.approx(feat[k], rel=1e-12)
    assert tab.loc[2016, "growth"] == pytest.approx(
        np.log1p(a.set_index("year").loc[2016, "demand_total"]) - feat["lag_1"], rel=1e-12
    )
    obs = build_training_table(m, a, [2016], "observado")
    assert obs.loc[2016, "oni_mean_anio"] == pytest.approx(a.set_index("year").loc[2016, "oni_mean_anio"])


def test_replica_de_trayectoria_igual_a_inferencia_real(clima_data):
    """El evaluador walk-forward (``forecast_path``) reproduce EXACTAMENTE ``predict_future_climate``."""
    from training.forecast_climate import clima_oni_model as com
    from training.forecast_climate import walkforward_clima as wf

    spec = com.GrowthModelSpec(extra_features=("oni_mean_anio",), oni_source="observado", window_years=5)
    tab = clima_data.training_table(2017, "observado")
    model = com.fit_growth_model(tab, spec)
    arts = com.to_sklearn_artifacts(model, model.train_X, tab.loc[model.train_years, "target_log"])
    mine = wf.forecast_path(model, clima_data, 2017, 3, 2)
    real = wf.real_inference_path(clima_data, 2017, 3, arts, 2)
    np.testing.assert_allclose(mine.to_numpy(), real.reindex(mine.index).to_numpy(), rtol=1e-10)


def test_conversion_a_sklearn_es_exacta():
    from training.forecast_climate import clima_oni_model as com

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"lag_1": rng.normal(22, 0.3, 12), "oni_mean_anio": rng.normal(0, 0.8, 12)})
    model = com.GrowthModel(com.GrowthModelSpec(), mu=0.05, coefs={"oni_mean_anio": 0.02}, train_X=X)
    sx, lr, sy = com.to_sklearn_artifacts(model, X, X["lag_1"] + 0.05)
    got = sy.inverse_transform(lr.predict(sx.transform(X[list(sx.feature_names_in_)])).reshape(-1, 1)).ravel()
    np.testing.assert_allclose(got, model.predict_log(X), rtol=1e-12)


def test_signo_oni_no_negativo():
    from training.forecast_climate import clima_oni_model as com

    years = list(range(2010, 2022))
    oni = np.linspace(-1, 1, len(years))
    tab = pd.DataFrame({"oni_mean_anio": oni, "growth": 0.05 - 0.03 * oni, "lag_1": 22.0}, index=years)
    model = com.fit_growth_model(tab, com.GrowthModelSpec(extra_features=("oni_mean_anio",)))
    assert model.coefs["oni_mean_anio"] == 0.0
    assert model.mu == pytest.approx(0.05, abs=1e-9)


def test_corrector_honesto_no_usa_objetivos_futuros():
    from training.forecast_climate.walkforward_clima import honest_corrector_eval

    origins = [pd.Timestamp(f"{y}-12-01") for y in range(2015, 2019)]
    rows = []
    for o in origins:
        for h in range(1, 25):
            t = o + pd.offsets.MonthBegin(h)
            rows.append({"origen": o, "objetivo": t, "h": h, "pred": 100.0, "real": 100.0})
    res = pd.DataFrame(rows)
    res["err_pct"] = 0.0
    bad = res.copy()
    # Reales absurdos en objetivos posteriores al último origen: no deben afectar a ninguna corrección.
    bad.loc[bad["objetivo"] > origins[-1], "real"] = 1e6
    a, _ = honest_corrector_eval(res, origins)
    b, _ = honest_corrector_eval(bad, origins)
    np.testing.assert_allclose(a["pred_corr"], b["pred_corr"])


# --------------------------------------------------------------------------------------------------
# Artefactos productivos v2 + escenarios ONI
# --------------------------------------------------------------------------------------------------
def test_artefactos_v2_compatibles_y_parsimoniosos(prod_predictor):
    import joblib

    info = joblib.load(MODEL_DIR / "model_info.pkl")
    assert info.get("version") == "v2"
    feats = prod_predictor.climate_annual_features
    assert "lag_1" in feats and "oni_mean_anio" in feats
    assert len(feats) <= 4, "el modelo anual debe ser parsimonioso"
    # Elasticidad respecto a lag_1 exactamente 1 (paseo aleatorio con deriva) y ONI con signo físico.
    sx, mdl, sy = prod_predictor.climate_annual_scaler_x, prod_predictor.climate_annual_model, prod_predictor.climate_annual_scaler_y
    base = pd.DataFrame([{f: float(sx.mean_[i]) for i, f in enumerate(feats)}])

    def log_pred(df):
        return float(sy.inverse_transform(mdl.predict(sx.transform(df[feats])).reshape(-1, 1))[0, 0])

    up = base.copy()
    up["lag_1"] += 0.1
    assert log_pred(up) - log_pred(base) == pytest.approx(0.1, rel=1e-9)
    hot = base.copy()
    hot["oni_mean_anio"] += 1.0
    beta = log_pred(hot) - log_pred(base)
    assert 0.0 < beta < 0.08, f"sensibilidad ONI fuera de rango plausible: {beta}"


def test_escenarios_oni_coherentes_y_magnitud_plausible(prod_predictor):
    last = int(prod_predictor.climate_hist_annual.loc[prod_predictor.climate_hist_annual["n_meses"] >= 12, "year"].max())
    year = last + 1
    tot = {}
    for fase in ("NIÑA", "NORMAL", "NIÑO"):
        ann, _ = prod_predictor.predict_future_climate(
            forecast_years=1, cutoff_year=last, enso_phase_by_month={f"{year}-{m:02d}": fase for m in range(1, 13)}
        )
        tot[fase] = float(ann[0, 0])
    auto, _ = prod_predictor.predict_future_climate(forecast_years=1, cutoff_year=last)
    assert tot["NIÑA"] < tot["NORMAL"] < tot["NIÑO"]
    spread = tot["NIÑO"] / tot["NIÑA"] - 1
    assert 0.005 < spread < 0.10, f"spread Niño/Niña anual implausible: {spread:.3%}"
    # Auto (ONI proyectado) queda dentro del abanico de escenarios.
    assert tot["NIÑA"] <= float(auto[0, 0]) <= tot["NIÑO"]


def test_pronostico_v2_crece_y_sin_saltos(prod_predictor):
    last = int(prod_predictor.climate_hist_annual.loc[prod_predictor.climate_hist_annual["n_meses"] >= 12, "year"].max())
    ann, res = prod_predictor.predict_future_climate(forecast_years=2, cutoff_year=last)
    real_last = float(prod_predictor.climate_hist_annual.set_index("year").loc[last, "demand_total"])
    g1 = ann[0, 0] / real_last - 1
    g2 = ann[1, 0] / ann[0, 0] - 1
    assert 0.0 < g1 < 0.12 and 0.0 < g2 < 0.12
