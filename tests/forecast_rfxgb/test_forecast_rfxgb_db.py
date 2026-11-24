"""Integración v2 RF/XGB con la BD local y los artefactos entrenados (se omite si no hay BD/artefactos)."""
from __future__ import annotations

import pandas as pd
import pytest

from api.services import forecast_rfxgb as F

pytestmark = pytest.mark.integration_db


@pytest.fixture(scope="module")
def monthly():
    from training.common.data import get_engine, load_monthly_demand

    try:
        eng = get_engine()
        with eng.connect():
            pass
        return load_monthly_demand(eng, impute_incomplete=True)
    except Exception as exc:  # BD caída / tabla ausente
        pytest.skip(f"BD no disponible: {type(exc).__name__}")


@pytest.mark.parametrize("mid", [1, 2])
@pytest.mark.parametrize("start,end", [("2026-01-01", "2027-12-31"), ("2027-01-01", "2027-12-31"),
                                       ("2026-12-01", "2027-05-31")])
def test_bd_rangos_y_fechas(monthly, mid, start, end):
    from training.common.data import target_series

    try:
        F.load_artifact(mid)
    except FileNotFoundError:
        pytest.skip("artefactos v2 no entrenados")
    mb = monthly.attrs["market_break"]
    hist = target_series(monthly)  # incluye meses post-quiebre: predict_monthly debe cortarlos
    out = F.predict_monthly(mid, hist, start, end, market_break_month=mb.first_post_month if mb else None)
    assert list(out["fecha"]) == list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS"))
    pre = target_series(monthly, market_break=mb)
    ref = pre[pre.index < pd.Timestamp(start)].iloc[-12:].mean()
    assert out["demanda_kwh"].between(0.85 * ref, 1.35 * ref).all()
    # Sin colapso por año parcial: total 2027 >= 2025 completo
    if pd.Timestamp(start).year <= 2027 <= pd.Timestamp(end).year and len(out[out["fecha"].dt.year == 2027]) == 12:
        assert out[out["fecha"].dt.year == 2027]["demanda_kwh"].sum() > pre.loc["2025"].sum()
