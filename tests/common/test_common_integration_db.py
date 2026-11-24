"""Integración con la BD local (se omite si no está disponible)."""
from __future__ import annotations

import pandas as pd
import pytest

from training.common.data import get_engine, load_monthly_demand, target_series
from training.common.walkforward import default_exclusions, evaluate_with_baselines, make_origins

pytestmark = pytest.mark.integration_db


@pytest.fixture(scope="module")
def monthly():
    try:
        eng = get_engine()
        with eng.connect():
            pass
    except Exception as exc:  # BD caída / sin driver
        pytest.skip(f"BD no disponible: {type(exc).__name__}")
    try:
        return load_monthly_demand(eng, impute_incomplete=True)
    except Exception as exc:  # tabla ausente en otra BD
        pytest.skip(f"tabla de demanda no disponible: {type(exc).__name__}")


def test_bd_sin_duplicados_meses_marcados(monthly):
    assert monthly.index.is_unique
    assert monthly.attrs["n_total_vs_horas"] == 0
    log = monthly.attrs["dedup_log"]
    if len(log):
        # Cada elección debe estar en la escala de los vecinos (±50 %).
        assert ((log["elegido"] / log["referencia"]).between(0.5, 1.5)).all()
    incompletos = monthly[~monthly["completo"]]
    assert monthly.loc[incompletos.index, "demanda_kwh"].isna().sum() + incompletos["imputado"].sum() == len(incompletos)
    # Ningún mes preliminar es target.
    assert monthly.loc[monthly["preliminar"], "demanda_kwh"].isna().all()


def test_bd_harness_baselines_corre(monthly):
    ts = target_series(monthly)
    act = target_series(monthly, include_imputed=False)
    mb = monthly.attrs["market_break"]
    _, summ = evaluate_with_baselines(
        None, ts, make_origins("2022-12", "2024-12", 12), 12, actuals=act, exclusions=default_exclusions(mb)
    )
    assert (summ["n"] > 0).all()
    assert summ["mape"].between(0, 50).all()
    assert isinstance(ts.index, pd.DatetimeIndex)
