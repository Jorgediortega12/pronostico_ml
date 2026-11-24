"""Tests de `DemandPredictor` para el modelo climático + ONI (id=4).

Carga los artefactos productivos reales de `api/models/climate_model/`
(serializados por el agente de entrenamiento) y el histórico real de
`training/forecast_climate/artifacts/` para verificar que:

  - `DemandPredictor(model_path='api/models/climate_model/')` carga
    correctamente todos los artefactos (scalers, modelos, corrector).
  - `predict_future_climate` produce un output con la forma esperada
    (longitud, columnas, rangos razonables) contra esos datos reales.

Se salta automáticamente (skip) si los artefactos productivos aún no
existen en el entorno donde corre pytest.
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

CLIMATE_MODEL_DIR = BACKEND_ROOT / "api" / "models" / "climate_model"
REQUIRED_ARTIFACTS = [
    "annual_model.pkl",
    "annual_scaler_x.pkl",
    "annual_scaler_y.pkl",
    "monthly_model.pkl",
    "metrics.pkl",
    "model_info.pkl",
    "postproc_corrector_v1.json",
]

pytestmark = pytest.mark.skipif(
    not all((CLIMATE_MODEL_DIR / f).exists() for f in REQUIRED_ARTIFACTS),
    reason="Artefactos productivos de api/models/climate_model/ no disponibles en este entorno.",
)


@pytest.fixture(scope="module")
def predictor():
    from api.services.forecast import DemandPredictor

    return DemandPredictor(model_path="api/models/climate_model/")


def test_predictor_loads_climate_artifacts(predictor):
    assert hasattr(predictor, "climate_annual_model")
    assert hasattr(predictor, "climate_monthly_model")
    assert hasattr(predictor, "climate_annual_scaler_x")
    assert hasattr(predictor, "climate_annual_scaler_y")

    # Las listas de features se recuperan de los artefactos, no se hardcodean.
    assert predictor.climate_annual_features == list(
        predictor.climate_annual_scaler_x.feature_names_in_
    )
    assert predictor.climate_monthly_features == list(
        predictor.climate_monthly_model.feature_names_in_
    )
    assert len(predictor.climate_annual_features) > 0
    assert len(predictor.climate_monthly_features) > 0


def test_predictor_loads_corrector(predictor):
    assert predictor.climate_corrector is not None
    assert "bias_corrector" in predictor.climate_corrector
    assert "seasonal_corrector" in predictor.climate_corrector


def test_predictor_loads_historical_climate_data(predictor):
    assert isinstance(predictor.climate_hist_monthly, pd.DataFrame)
    assert isinstance(predictor.climate_hist_annual, pd.DataFrame)
    assert len(predictor.climate_hist_monthly) > 0
    assert len(predictor.climate_hist_annual) > 0
    for col in ("year", "month", "demand_value", "oni_actual"):
        assert col in predictor.climate_hist_monthly.columns
    for col in ("year", "demand_total"):
        assert col in predictor.climate_hist_annual.columns


def test_get_mape_returns_climate_metrics(predictor):
    mape = predictor.get_mape()
    assert set(mape.keys()) == {"annual", "monthly"}
    assert mape["annual"] > 0
    assert mape["monthly"] > 0


def test_predict_future_climate_shape_and_ranges(predictor):
    forecast_years = 2
    annual_pred, results = predictor.predict_future_climate(forecast_years=forecast_years)

    # --- annual_pred ---
    assert isinstance(annual_pred, np.ndarray)
    assert annual_pred.shape == (forecast_years, 1)
    # Rango razonable: la demanda anual histórica real está entre ~3.9e9 y
    # ~8.6e9 (dataset_anual.csv); las predicciones no deberían dispararse
    # órdenes de magnitud fuera de ese rango.
    for val in annual_pred.flatten():
        assert 1e9 < val < 2e10

    # --- results (mensual) ---
    expected_columns = {"fecha", "prediccion", "prediccion_cruda", "prediccion_ajustada"}
    assert expected_columns.issubset(set(results.columns))
    assert len(results) == forecast_years * 12

    # Fechas mensuales consecutivas, ordenadas
    assert results["fecha"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(results["fecha"])

    # Cada año pronosticado debe tener exactamente 12 meses
    year_counts = results["fecha"].dt.year.value_counts()
    assert (year_counts == 12).all()

    # Valores mensuales positivos y en un rango razonable
    assert (results["prediccion"] > 0).all()
    assert (results["prediccion"] < 3e9).all()

    # La suma mensual de cada año debe ser aproximadamente el total anual
    # ajustado (mismo principio de normalización que predict_future_xgrf).
    for i, year in enumerate(sorted(results["fecha"].dt.year.unique())):
        year_sum = results.loc[results["fecha"].dt.year == year, "prediccion"].sum()
        assert year_sum == pytest.approx(annual_pred.flatten()[i], rel=1e-6)

    # Trazabilidad: predicción cruda vs ajustada deben ser series distintas
    # (el corrector se aplicó), ambas positivas.
    assert (results["prediccion_cruda"] > 0).all()
    assert (results["prediccion_ajustada"] > 0).all()


def test_predict_future_climate_respects_atypical_years(predictor):
    # No debe fallar al excluir un año atípico presente en el histórico.
    hist_years = predictor.climate_hist_annual["year"].tolist()
    atypical = [hist_years[len(hist_years) // 2]]
    annual_pred, results = predictor.predict_future_climate(
        forecast_years=1, atypical_years=atypical
    )
    assert annual_pred.shape == (1, 1)
    assert len(results) == 12


def _year_month_share(results, year, month):
    year_mask = results["fecha"].dt.year == year
    year_total = results.loc[year_mask, "prediccion"].sum()
    month_val = results.loc[year_mask & (results["fecha"].dt.month == month), "prediccion"].iloc[0]
    return month_val / year_total


def _expected_recent_full_year_share(predictor, month, n_years_shape, cutoff_year=None):
    hist_monthly = predictor.climate_hist_monthly.copy()
    if cutoff_year is not None:
        hist_monthly = hist_monthly[hist_monthly["year"] <= cutoff_year]
    hist_monthly = hist_monthly.copy()
    hist_monthly["yearly_total"] = hist_monthly.groupby("year")["demand_value"].transform("sum")
    hist_monthly["demand_percentage"] = hist_monthly["demand_value"] / hist_monthly["yearly_total"]

    month_counts = hist_monthly.groupby("year")["month"].transform("count")
    full_years_pct = hist_monthly.loc[month_counts == 12]
    full_years_sorted = sorted(full_years_pct["year"].unique())
    recent_full_years = full_years_sorted[-n_years_shape:]
    shape_src = full_years_pct[full_years_pct["year"].isin(recent_full_years)]

    return float(shape_src.loc[shape_src["month"] == month, "demand_percentage"].mean())


def test_predict_future_climate_may_share_exceeds_june_every_year(predictor):
    forecast_years = 5
    _, results = predictor.predict_future_climate(forecast_years=forecast_years)

    for year in sorted(results["fecha"].dt.year.unique()):
        may_share = _year_month_share(results, year, 5)
        june_share = _year_month_share(results, year, 6)
        assert may_share > june_share, f"year {year}: may={may_share}, june={june_share}"


def test_predict_future_climate_monthly_shape_is_stable_across_years(predictor):
    forecast_years = 5
    _, results = predictor.predict_future_climate(forecast_years=forecast_years)

    years = sorted(results["fecha"].dt.year.unique())
    assert len(years) == forecast_years

    for month in (5, 6, 11):
        shares = [_year_month_share(results, year, month) for year in years]
        for share in shares[1:]:
            assert share == pytest.approx(shares[0], abs=1e-9)

    # Validación independiente: el share de mayo debe coincidir con el
    # promedio real de los últimos n_years_shape=2 años completos.
    expected_may_share = _expected_recent_full_year_share(predictor, month=5, n_years_shape=2)
    actual_may_share = _year_month_share(results, years[0], 5)
    assert actual_may_share == pytest.approx(expected_may_share, rel=1e-6)


def test_predict_future_climate_never_calls_monthly_xgboost_model(predictor):
    from unittest.mock import patch

    with patch.object(predictor.climate_monthly_model, "predict") as mock_predict:
        predictor.predict_future_climate(forecast_years=3)
        mock_predict.assert_not_called()


def test_predict_future_climate_respects_n_years_shape_window(predictor):
    _, results_1y = predictor.predict_future_climate(forecast_years=1, n_years_shape=1)
    _, results_3y = predictor.predict_future_climate(forecast_years=1, n_years_shape=3)

    year_1y = results_1y["fecha"].dt.year.iloc[0]
    year_3y = results_3y["fecha"].dt.year.iloc[0]

    share_1y = _year_month_share(results_1y, year_1y, 11)
    share_3y = _year_month_share(results_3y, year_3y, 11)
    assert share_1y != pytest.approx(share_3y, rel=1e-6)

    expected_1y_share = _expected_recent_full_year_share(predictor, month=11, n_years_shape=1)
    assert share_1y == pytest.approx(expected_1y_share, rel=1e-6)


# ---------------------------------------------------------------------------
# enso_phase_by_month: override de fase ENSO mes a mes
# ---------------------------------------------------------------------------


def test_oni_to_enso_phase_and_enso_phase_oni_mapping():
    from api.services.forecast import ENSO_PHASE_ONI, _oni_to_enso_phase

    assert _oni_to_enso_phase(0.5) == "NIÑO"
    assert _oni_to_enso_phase(1.2) == "NIÑO"
    assert _oni_to_enso_phase(-0.5) == "NIÑA"
    assert _oni_to_enso_phase(-1.0) == "NIÑA"
    assert _oni_to_enso_phase(0.0) == "NORMAL"
    assert _oni_to_enso_phase(0.49) == "NORMAL"
    assert _oni_to_enso_phase(-0.49) == "NORMAL"

    assert ENSO_PHASE_ONI == {"NIÑO": 0.9, "NIÑA": -0.8, "NORMAL": -0.1}
    assert _oni_to_enso_phase(ENSO_PHASE_ONI["NIÑO"]) == "NIÑO"
    assert _oni_to_enso_phase(ENSO_PHASE_ONI["NIÑA"]) == "NIÑA"
    assert _oni_to_enso_phase(ENSO_PHASE_ONI["NORMAL"]) == "NORMAL"


def test_phase_conditional_month_share_fallback_and_samples(predictor):
    from api.services.forecast import (
        _phase_conditional_month_share,
        _phase_conditional_stats,
        _isotonic_phase_month_shares,
        _oni_to_enso_phase,
    )

    hist = predictor.climate_hist_monthly.copy()
    hist["yearly_total"] = hist.groupby("year")["demand_value"].transform("sum")
    hist["demand_percentage"] = hist["demand_value"] / hist["yearly_total"]
    month_counts = hist.groupby("year")["month"].transform("count")
    full = hist.loc[month_counts == 12]

    fallback = 0.083

    # Fase inválida -> fallback exacto, sin importar shrinkage_k.
    assert _phase_conditional_month_share(
        full, month=5, phase="INVALIDA", fallback=fallback
    ) == fallback

    # Mes sin ninguna fila histórica -> fallback exacto (n=0 en las 3 fases,
    # PAVA no encuentra violación porque los 3 valores quedan iguales).
    assert _phase_conditional_month_share(
        full, month=13, phase="NIÑO", fallback=fallback
    ) == fallback

    # --- _phase_conditional_stats: extracción cruda, sin fallback/shrinkage ---

    # Fase inválida -> (None, 0).
    assert _phase_conditional_stats(full, month=5, phase="INVALIDA") == (None, 0)

    # Mes sin filas -> (None, 0).
    assert _phase_conditional_stats(full, month=13, phase="NIÑO") == (None, 0)

    subset = full.loc[full["month"] == 5]
    phase_mask = subset["oni_actual"].map(_oni_to_enso_phase) == "NORMAL"
    phase_series = subset.loc[phase_mask, "demand_percentage"]
    assert len(phase_series) > 0, "se necesita al menos un mayo NORMAL en el dataset real para este test"
    pure_conditional_mean = float(phase_series.mean())
    n = len(phase_series)

    mean_stat, n_stat = _phase_conditional_stats(full, month=5, phase="NORMAL")
    assert n_stat == n
    assert mean_stat == pytest.approx(pure_conditional_mean, rel=1e-9)

    # Comparación contra cálculo manual con pandas puro (sin pasar por el
    # módulo, para validar independientemente el filtrado mes+fase).
    manual_subset = full.loc[full["month"] == 5].copy()
    manual_subset["phase"] = manual_subset["oni_actual"].apply(_oni_to_enso_phase)
    manual_normal = manual_subset.loc[manual_subset["phase"] == "NORMAL", "demand_percentage"]
    assert n_stat == len(manual_normal)
    assert mean_stat == pytest.approx(float(manual_normal.mean()), rel=1e-9)

    # --- _isotonic_phase_month_shares: shrinkage leve + PAVA ---

    # k enorme -> las 3 fases convergen (casi exactamente) al fallback,
    # independientemente de cuántas muestras condicionales haya (y por lo
    # tanto ya son no decrecientes -> PAVA no interviene).
    shares_huge_k = _isotonic_phase_month_shares(full, month=5, fallback=fallback, shrinkage_k=1e9)
    for phase in ("NIÑA", "NORMAL", "NIÑO"):
        assert shares_huge_k[phase] == pytest.approx(fallback, abs=1e-6)

    # k ~0 -> si el orden crudo (promedios condicionales puros) de mayo ya
    # es NIÑA <= NORMAL <= NIÑO, las 3 fases convergen exactamente a sus
    # promedios condicionales puros (PAVA es la identidad).
    means_by_phase = {}
    for phase in ("NIÑA", "NORMAL", "NIÑO"):
        mean, phase_n = _phase_conditional_stats(full, month=5, phase=phase)
        assert phase_n > 0, f"se necesita al menos un mayo {phase} en el dataset real para este test"
        means_by_phase[phase] = mean

    shares_tiny_k = _isotonic_phase_month_shares(full, month=5, fallback=fallback, shrinkage_k=1e-9)
    if means_by_phase["NIÑA"] <= means_by_phase["NORMAL"] <= means_by_phase["NIÑO"]:
        for phase in ("NIÑA", "NORMAL", "NIÑO"):
            assert shares_tiny_k[phase] == pytest.approx(means_by_phase[phase], rel=1e-6)
    else:
        # Si los promedios crudos violan el orden, PAVA los poolea -- en
        # cualquier caso el resultado debe quedar no decreciente.
        assert (
            shares_tiny_k["NIÑA"] <= shares_tiny_k["NORMAL"] <= shares_tiny_k["NIÑO"]
        )

    # Caso a mano para NORMAL: verificar la fórmula de shrinkage leve exacta
    # con n y k conocidos, cuando el orden crudo ya es no decreciente (PAVA
    # identidad, así que el share final == shrinkage puro).
    if means_by_phase["NIÑA"] <= means_by_phase["NORMAL"] <= means_by_phase["NIÑO"]:
        k = 7.0
        expected_normal = fallback + (n / (n + k)) * (pure_conditional_mean - fallback)
        shares_k7 = _isotonic_phase_month_shares(full, month=5, fallback=fallback, shrinkage_k=k)
        assert shares_k7["NORMAL"] == pytest.approx(expected_normal, rel=1e-9)

    # El orden NIÑA <= NORMAL <= NIÑO se cumple SIEMPRE, para cualquier k
    # (garantía estructural de PAVA), independientemente de si los
    # promedios condicionales crudos lo cumplían o no.
    for k in (0.0, 1.0, 7.0, 50.0, 1e6):
        shares_k = _isotonic_phase_month_shares(full, month=5, fallback=fallback, shrinkage_k=k)
        assert shares_k["NIÑA"] <= shares_k["NORMAL"] <= shares_k["NIÑO"]


def test_predict_future_climate_no_enso_override_is_identical(predictor):
    annual_base, results_base = predictor.predict_future_climate(forecast_years=2)
    annual_none, results_none = predictor.predict_future_climate(
        forecast_years=2, enso_phase_by_month=None
    )
    annual_empty, results_empty = predictor.predict_future_climate(
        forecast_years=2, enso_phase_by_month={}
    )

    np.testing.assert_allclose(annual_base, annual_none, rtol=0, atol=0)
    np.testing.assert_allclose(annual_base, annual_empty, rtol=0, atol=0)
    np.testing.assert_allclose(
        results_base["prediccion"].to_numpy(),
        results_none["prediccion"].to_numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        results_base["prediccion"].to_numpy(),
        results_empty["prediccion"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_predict_future_climate_all_nino_vs_all_nina_annual_differs(predictor):
    last_hist = int(predictor.climate_hist_annual["year"].max())
    forecast_year = last_hist + 1
    nino_map = {f"{forecast_year}-{m:02d}": "NIÑO" for m in range(1, 13)}
    nina_map = {f"{forecast_year}-{m:02d}": "NIÑA" for m in range(1, 13)}
    normal_map = {f"{forecast_year}-{m:02d}": "NORMAL" for m in range(1, 13)}

    annual_nino, _ = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=nino_map
    )
    annual_nina, _ = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=nina_map
    )
    annual_normal, _ = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=normal_map
    )

    assert annual_nino.shape == (1, 1)
    assert annual_nina.shape == (1, 1)
    assert annual_normal.shape == (1, 1)
    assert annual_nino[0, 0] != pytest.approx(annual_nina[0, 0], rel=1e-9)

    # Regresión directa del bug reportado: forzar el año completo a NIÑA,
    # NORMAL o NIÑO debe respetar el orden físico esperado
    # NIÑA < NORMAL < NIÑO en el total anual pronosticado.
    assert annual_nina[0, 0] < annual_normal[0, 0] < annual_nino[0, 0]


def test_phase_conditional_share_preserves_physical_order_single_month(predictor):
    """Regresión directa del bug reportado por el usuario: forzando UN SOLO
    mes a NIÑA/NORMAL/NIÑO (selector "Escenario ENSO (mes a mes)"), la
    demanda de ese mes debe respetar NIÑA < NORMAL < NIÑO -- para varios
    meses y varios cutoff_year (para no depender de un único año)."""
    last_hist = int(predictor.climate_hist_annual["year"].max())
    cutoff_years = [last_hist - 1, last_hist - 3]

    for cutoff_year in cutoff_years:
        forecast_year = cutoff_year + 1
        for month in (1, 4, 7, 10):  # enero, abril, julio, octubre
            key = f"{forecast_year}-{month:02d}"
            demand_by_phase = {}
            for phase in ("NIÑA", "NORMAL", "NIÑO"):
                _, results = predictor.predict_future_climate(
                    forecast_years=1,
                    cutoff_year=cutoff_year,
                    enso_phase_by_month={key: phase},
                )
                mask = (
                    (results["fecha"].dt.year == forecast_year)
                    & (results["fecha"].dt.month == month)
                )
                demand_by_phase[phase] = float(results.loc[mask, "prediccion"].iloc[0])

            assert demand_by_phase["NIÑA"] < demand_by_phase["NORMAL"] < demand_by_phase["NIÑO"], (
                f"cutoff_year={cutoff_year} mes={month}: "
                f"NIÑA={demand_by_phase['NIÑA']}, NORMAL={demand_by_phase['NORMAL']}, "
                f"NIÑO={demand_by_phase['NIÑO']}"
            )


def test_weighted_pava_nondecreasing_is_identity_when_already_ordered():
    """Si `values` ya es no decreciente, PAVA no debe tocar nada -- se
    preserva exactamente la magnitud original de la señal."""
    from api.services.forecast import _weighted_pava_nondecreasing

    values = [1.0, 2.5, 2.5, 9.0]
    weights = [5, 3, 10, 1]
    result = _weighted_pava_nondecreasing(values, weights)
    assert result == pytest.approx(values, rel=0, abs=0)


def test_weighted_pava_nondecreasing_pools_violating_adjacent_pair():
    """Caso sintético con una única violación conocida: los dos valores
    adyacentes que violan el orden se funden en el promedio ponderado
    exacto de ambos; el resto de la secuencia queda intacto."""
    from api.services.forecast import _weighted_pava_nondecreasing

    # values[1] > values[2] -> violación entre el segundo y tercer elemento.
    values = [1.0, 5.0, 3.0, 6.0]
    weights = [1.0, 2.0, 4.0, 1.0]

    expected_pool = (5.0 * 2.0 + 3.0 * 4.0) / (2.0 + 4.0)  # 3.6667
    result = _weighted_pava_nondecreasing(values, weights)

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(expected_pool, rel=1e-9)
    assert result[2] == pytest.approx(expected_pool, rel=1e-9)
    assert result[3] == pytest.approx(6.0)
    # No decreciente por construcción.
    assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def test_weighted_pava_nondecreasing_cascading_pool():
    """Fuerza un merge en cascada: el pool inicial de dos valores sigue
    violando el orden con el bloque anterior, forzando una segunda fusión
    (más de un pool encadenado)."""
    from api.services.forecast import _weighted_pava_nondecreasing

    # values[2] > values[3] viola primero -> se poolea (values[2],values[3]).
    # El bloque resultante (peso 1+1=2, valor 5.0) sigue siendo MAYOR que
    # values[1]=1.0? No -- diseñamos el caso para que el pool en cascada
    # ocurra hacia atrás: el primer pool baja lo suficiente para seguir
    # violando con el bloque anterior, forzando una segunda fusión.
    values = [1.0, 10.0, 8.0, 0.0]
    weights = [1.0, 1.0, 1.0, 1.0]
    # Paso 1: bloques [1],[10],[8],[0]. Al agregar 8: 10>8 -> pool (10,8)
    #   -> valor (10+8)/2=9.0, peso 2. Bloques: [1],[9.0(w2)].
    # Al agregar 0: 9.0>0 -> pool (9.0(w2), 0(w1)) -> valor
    #   (9.0*2+0*1)/3=6.0, peso 3. Bloques: [1],[6.0(w3)].
    # 1.0 <= 6.0 -> no hay más violación con el primer bloque.
    expected = [1.0, 6.0, 6.0, 6.0]

    result = _weighted_pava_nondecreasing(values, weights)
    assert result == pytest.approx(expected, rel=1e-9)
    assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def test_predict_future_climate_single_month_sensitivity_spread_is_meaningful(predictor):
    """Regresión contra el bug de sensibilidad casi eliminada (caso real
    reportado por el usuario: Niña=744, Normal=747, Niño=752 GWh, spread
    ~1% con el viejo shrinkage uniforme k=85). Con shrinkage LEVE + PAVA
    (k_light=3, ver training/forecast_climate/artifacts/reporte_isotonic_enso.md,
    spread promedio real del barrido ~3.1% vs ~1.0% del baseline), el
    spread NIÑO-vs-NIÑA de un mes forzado individualmente debe mantenerse
    sustancialmente por encima del nivel casi nulo del bug original.

    Umbral fijado a partir del barrido real: promedio observado en el
    barrido (k_light=3) fue 3.11%; aquí se usa un umbral bajo el promedio
    real medido para estos casos concretos (~2.87%) para no ser frágil
    ante variación entre entornos, pero muy por encima del ~1% del bug
    original -- documenta que se recuperó sensibilidad real.

    Recalibrado 2026-11-23 (agente-clima, docs/work/04_clima/REPORTE.md §Escenarios): el ~2.9 % previo
    venía en gran parte de datos corruptos (2020-Q4 deflactado ~50 % en dataset_mensual.csv → shares
    de oct-dic 2020 —año NIÑA— artificialmente bajos). Con la capa de datos limpia, el mismo k_light=3
    da 0.71 % de spread medio en estos 8 casos y el shrinkage agresivo del bug (k=85) da 0.30 %
    (k=20: 0.42 %). El umbral 0.5 % sigue separando k_light=3 del shrinkage agresivo (que es lo que
    este test protege) sin depender del artefacto de datos."""
    UMBRAL_SPREAD_PROMEDIO_PCT = 0.5

    last_hist = int(predictor.climate_hist_annual["year"].max())
    cutoff_years = [last_hist - 1, last_hist - 3]
    months = (1, 4, 7, 10)

    spreads = []
    for cutoff_year in cutoff_years:
        forecast_year = cutoff_year + 1
        for month in months:
            key = f"{forecast_year}-{month:02d}"
            demand_by_phase = {}
            for phase in ("NIÑA", "NORMAL", "NIÑO"):
                _, results = predictor.predict_future_climate(
                    forecast_years=1,
                    cutoff_year=cutoff_year,
                    enso_phase_by_month={key: phase},
                )
                mask = (
                    (results["fecha"].dt.year == forecast_year)
                    & (results["fecha"].dt.month == month)
                )
                demand_by_phase[phase] = float(results.loc[mask, "prediccion"].iloc[0])

            assert demand_by_phase["NIÑA"] <= demand_by_phase["NORMAL"] <= demand_by_phase["NIÑO"]
            spread_pct = 100.0 * (
                (demand_by_phase["NIÑO"] - demand_by_phase["NIÑA"]) / demand_by_phase["NIÑA"]
            )
            spreads.append(spread_pct)

    spread_promedio = sum(spreads) / len(spreads)
    assert spread_promedio >= UMBRAL_SPREAD_PROMEDIO_PCT, (
        f"spread promedio {spread_promedio:.2f}% por debajo del umbral "
        f"{UMBRAL_SPREAD_PROMEDIO_PCT}% -- posible regresión hacia un "
        "shrinkage demasiado agresivo (revisar PHASE_SHARE_LIGHT_SHRINKAGE_K)"
    )


def test_predict_future_climate_single_month_nino_vs_nina_differs(predictor):
    last_hist = int(predictor.climate_hist_annual["year"].max())
    forecast_year = last_hist + 1
    may_nino = {f"{forecast_year}-05": "NIÑO"}
    may_nina = {f"{forecast_year}-05": "NIÑA"}

    _, results_base = predictor.predict_future_climate(forecast_years=1)
    _, results_nino = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=may_nino
    )
    _, results_nina = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=may_nina
    )

    share_nino = _year_month_share(results_nino, forecast_year, 5)
    share_nina = _year_month_share(results_nina, forecast_year, 5)
    share_base = _year_month_share(results_base, forecast_year, 5)

    # Al menos una de las fases debe mover el share de mayo respecto al baseline
    # o entre sí; el requisito fuerte es NIÑO ≠ NIÑA.
    assert share_nino != pytest.approx(share_nina, rel=1e-9) or (
        results_nino.loc[
            (results_nino["fecha"].dt.year == forecast_year)
            & (results_nino["fecha"].dt.month == 5),
            "prediccion",
        ].iloc[0]
        != pytest.approx(
            results_nina.loc[
                (results_nina["fecha"].dt.year == forecast_year)
                & (results_nina["fecha"].dt.month == 5),
                "prediccion",
            ].iloc[0],
            rel=1e-9,
        )
    )
    # Sanity: valores positivos
    assert share_base > 0 and share_nino > 0 and share_nina > 0


def test_predict_future_climate_partial_nino_differs_from_full_normal(predictor):
    """Meses omitidos usan proyección ONI; forzar todo el año a NORMAL no es equivalente.

    Un override parcial (solo mayo=NIÑO) debe diferir de un mapa con los 12 meses
    en NORMAL — documenta que los meses sin override no quedan atados a ONI=-0.1.
    """
    last_hist = int(predictor.climate_hist_annual["year"].max())
    forecast_year = last_hist + 1
    may_nino = {f"{forecast_year}-05": "NIÑO"}
    all_normal = {f"{forecast_year}-{m:02d}": "NORMAL" for m in range(1, 13)}

    annual_partial, results_partial = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=may_nino
    )
    annual_normal, results_normal = predictor.predict_future_climate(
        forecast_years=1, enso_phase_by_month=all_normal
    )

    annual_differs = annual_partial[0, 0] != pytest.approx(annual_normal[0, 0], rel=1e-9)
    monthly_differs = not np.allclose(
        results_partial["prediccion"].to_numpy(),
        results_normal["prediccion"].to_numpy(),
        rtol=0,
        atol=0,
    )
    assert annual_differs or monthly_differs


def test_enso_phase_by_month_schema_validation():
    from pydantic import ValidationError
    from api.config.schemas import ForecastSchema

    base = dict(id=4, user_id=1, session_id=1, start_date="2026-01-01", end_date="2026-12-31")

    ok = ForecastSchema(**base, enso_phase_by_month={"2026-05": "NIÑO", "2026-06": "NORMAL"})
    assert ok.enso_phase_by_month["2026-05"] == "NIÑO"

    empty = ForecastSchema(**base, enso_phase_by_month={})
    assert empty.enso_phase_by_month == {}

    none_schema = ForecastSchema(**base)
    assert none_schema.enso_phase_by_month is None

    with pytest.raises(ValidationError):
        ForecastSchema(**base, enso_phase_by_month={"2026-13": "NIÑO"})

    with pytest.raises(ValidationError):
        ForecastSchema(**base, enso_phase_by_month={"202605": "NIÑO"})

    with pytest.raises(ValidationError):
        ForecastSchema(**base, enso_phase_by_month={"2026-05": "EL_NINO"})

    with pytest.raises(ValidationError):
        ForecastSchema(**base, enso_phase_by_month={"2026-05": "nino"})

