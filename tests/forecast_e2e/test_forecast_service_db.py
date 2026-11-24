"""Integración de ``ForecastService.predict`` con la BD local (se omite si no hay BD).

Cubre los 4 modelos, rangos arbitrarios, IPC/escenarios ENSO, contrato JSON (campos previos intactos +
opcionales nuevos), ausencia de NaN/inf y que el tope nunca actúe en meses post-división.
"""
from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration_db

CONTRATO_MENSUAL = {"fechas", "valores_pronostico", "valores_reales", "errores_porcentuales"}
CONTRATO_RAIZ = {"annual_predictions", "monthly_predictions", "growth", "sarimax_macro_fallback",
                 "metricas_error", "demand_values_for_table", "date_values_for_table"}
NUEVOS_MENSUAL = {"valores_pronostico_crudos", "errores_porcentuales_crudos", "tope_aplicado",
                  "override_aplicado", "post_division", "factor_division"}


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


def _check_contract(r, start, end, mid):
    assert "message" not in r, r.get("message")
    assert CONTRATO_RAIZ <= set(r)
    mp = r["monthly_predictions"]
    assert CONTRATO_MENSUAL <= set(mp) and NUEVOS_MENSUAL <= set(mp)
    esperado = pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M")
    # Contrato de main: fin de mes para ids 1-3, inicio de mes para id 4.
    how = "end" if mid in (1, 2, 3) else "start"
    assert mp["fechas"] == [p.to_timestamp(how=how).strftime("%Y-%m-%d") for p in esperado]
    n = len(esperado)
    for k in CONTRATO_MENSUAL | NUEVOS_MENSUAL:
        assert len(mp[k]) == n, k
    json.dumps(r, allow_nan=False)  # sin NaN/inf
    v = np.asarray(mp["valores_pronostico"], dtype=float)
    assert np.isfinite(v).all() and (v > 0).all()
    assert all(isinstance(x, float) and np.isfinite(x) for x in r["annual_predictions"])
    return mp


RANGOS = [("2026-01-01", "2027-12-31"), ("2025-01-01", "2025-12-31"), ("2027-01-01", "2027-12-31"),
          ("2026-06-15", "2026-06-20"), ("2026-07-01", "2028-12-31")]


@pytest.mark.parametrize("mid", [1, 2, 3, 4])
@pytest.mark.parametrize("start,end", RANGOS)
def test_rangos_contrato(session_factory, mid, start, end):
    r = _predict(session_factory, id=mid, start_date=start, end_date=end)
    mp = _check_contract(r, start, end, mid)
    # El tope nunca actúa en modelos 3/4 ni en meses post-división con factor != 1 (con factor 1.0 la escala no
    # cambia y sí puede actuar).
    for tope, post, f in zip(mp["tope_aplicado"], mp["post_division"], mp["factor_division"]):
        assert not (tope and post and f != 1.0)
    if mid in (3, 4):
        assert not any(mp["tope_aplicado"])
    # Sin tope ni override, el pronóstico es el crudo.
    for v, c, t, o in zip(mp["valores_pronostico"], mp["valores_pronostico_crudos"], mp["tope_aplicado"],
                          mp["override_aplicado"]):
        if not (t or o):
            assert v == pytest.approx(c)


@pytest.mark.parametrize("mid", [1, 2, 3, 4])
def test_curva_2027_plausible(session_factory, mid):
    r = _predict(session_factory, id=mid, start_date="2026-01-01", end_date="2027-12-31")
    mp = _check_contract(r, "2026-01-01", "2027-12-31", mid)
    v = np.asarray(mp["valores_pronostico_crudos"], dtype=float)
    reales_2025 = [x for x in r["annual_predictions"][:-2]]
    total_2025 = reales_2025[-1]
    # Escala de mercado completo: 2026 y 2027 crecen respecto a 2025 y no colapsan (sin x0.7).
    assert v[:12].sum() > 0.97 * total_2025
    assert v[12:].sum() > v[:12].sum() * 0.97
    assert v.min() > 0.85 * total_2025 / 12


def test_ipc_solo_rf_xgb_y_escala(session_factory):
    base = _predict(session_factory, id=1, start_date="2027-01-01", end_date="2027-12-31")
    ipc = _predict(session_factory, id=1, start_date="2027-01-01", end_date="2027-12-31",
                   ipc_growth_by_year={2027: 3.0})
    a = np.asarray(base["monthly_predictions"]["valores_pronostico"])
    b = np.asarray(ipc["monthly_predictions"]["valores_pronostico"])
    assert b == pytest.approx(a * 1.03)
    assert ipc["annual_predictions"][-1] == pytest.approx(base["annual_predictions"][-1] * 1.03, rel=1e-6)
    s3 = _predict(session_factory, id=3, start_date="2027-01-01", end_date="2027-12-31",
                  ipc_growth_by_year={2027: 3.0})
    s3b = _predict(session_factory, id=3, start_date="2027-01-01", end_date="2027-12-31")
    assert s3["monthly_predictions"]["valores_pronostico"] == s3b["monthly_predictions"]["valores_pronostico"]


def test_escenarios_enso_clima(session_factory):
    meses = {f"2027-{m:02d}": None for m in range(1, 13)}
    tot = {}
    for fase in ("NIÑA", "NORMAL", "NIÑO"):
        r = _predict(session_factory, id=4, start_date="2027-01-01", end_date="2027-12-31",
                     enso_phase_by_month={k: fase for k in meses})
        _check_contract(r, "2027-01-01", "2027-12-31", 4)
        tot[fase] = sum(r["monthly_predictions"]["valores_pronostico"])
    assert tot["NIÑA"] < tot["NORMAL"] < tot["NIÑO"]


def test_start_respetado_sarimax(session_factory):
    a = _predict(session_factory, id=3, start_date="2025-01-01", end_date="2025-12-31")
    b = _predict(session_factory, id=3, start_date="2026-01-01", end_date="2026-12-31")
    assert a["monthly_predictions"]["valores_pronostico_crudos"] != b["monthly_predictions"]["valores_pronostico_crudos"]
    assert a["modelo"]["ultimo_mes_historia"] == "2024-12"


def test_rango_invertido_devuelve_error_controlado(session_factory):
    r = _predict(session_factory, id=1, start_date="2027-01-01", end_date="2026-01-01")
    assert r.get("status") == "error"


# --------------------------------------------------------------------------------------------------
# Correcciones ronda 1 (docs/work/06_correcciones/REPORTE.md)
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def split_config(monkeypatch):
    """Fuerza la configuración de división de mercado sin tocar la BD."""
    from api.services.demand_config_service import DemandConfigService, DemandSplitConfig

    def _set(fecha_limite, porcentaje):
        monkeypatch.setattr(DemandConfigService, "get_config",
                            lambda self, db: DemandSplitConfig(pd.Timestamp(fecha_limite).date(), porcentaje))
    return _set


def _post_rows(r):
    mp = r["monthly_predictions"]
    return [i for i, (p, real) in enumerate(zip(mp["post_division"], mp["valores_reales"])) if p and real]


@pytest.mark.parametrize("mid", [1, 2])
def test_division_70_sep_oct_2026_escala_front_sin_tope(session_factory, split_config, mid):
    split_config("2026-08-31", 70.0)
    r = _predict(session_factory, id=mid, start_date="2026-01-01", end_date="2026-12-31")
    mp = _check_contract(r, "2026-01-01", "2026-12-31", mid)
    assert r["division_mercado"]["factor"] == pytest.approx(0.7)
    post = _post_rows(r)
    assert [mp["fechas"][i][:7] for i in post][:2] == ["2026-09", "2026-10"]
    for i in post:
        assert not mp["tope_aplicado"][i]  # factor != 1: el tope no actúa post-división
        pred, real = mp["valores_pronostico"][i], mp["valores_reales"][i]
        assert mp["errores_porcentuales"][i] == pytest.approx((pred * 0.7 - real) / real * 100)
        assert -15.0 < mp["errores_porcentuales"][i] < 5.0  # ~ -5..-9 % (los datos implican ~74 %)


@pytest.mark.parametrize("mid", [1, 2])
def test_division_100_tope_actua_post_division(session_factory, split_config, mid):
    split_config("2026-08-31", 100.0)
    r = _predict(session_factory, id=mid, start_date="2026-01-01", end_date="2026-12-31")
    mp = _check_contract(r, "2026-01-01", "2026-12-31", mid)
    post = _post_rows(r)
    assert post
    for i in post:
        pred, real = mp["valores_pronostico"][i], mp["valores_reales"][i]
        assert mp["errores_porcentuales"][i] == pytest.approx((pred - real) / real * 100)
        assert abs(mp["errores_porcentuales"][i]) <= 4.0 + 1e-9  # misma escala: el tope sí actúa
    # Con real al ~70 % y factor 1.0 el error crudo es grande, así que el tope actuó en esos meses.
    assert any(mp["tope_aplicado"][i] for i in post)


def test_ipc_start_mitad_de_anio_no_escala_reales(session_factory):
    base = _predict(session_factory, id=2, start_date="2025-07-01", end_date="2025-12-31")
    ipc = _predict(session_factory, id=2, start_date="2025-07-01", end_date="2025-12-31",
                   ipc_growth_by_year={2025: 10.0})
    a = np.asarray(base["monthly_predictions"]["valores_pronostico_crudos"])
    b = np.asarray(ipc["monthly_predictions"]["valores_pronostico_crudos"])
    assert b == pytest.approx(a * 1.10)
    tot_a, tot_b = base["annual_predictions"][-1], ipc["annual_predictions"][-1]
    # Solo la parte pronosticada (jul-dic) se escala: delta anual == 10 % de la suma jul-dic.
    assert tot_b - tot_a == pytest.approx(0.10 * a.sum(), rel=1e-6, abs=2)


def test_clima_inicio_2028_error_claro(session_factory):
    r = _predict(session_factory, id=4, start_date="2028-01-01", end_date="2028-12-31")
    assert r.get("status") == "error"
    assert "solo puede pronosticar a partir de" in str(r.get("message"))


def test_clima_growth_anio_previo_todo_post_division(session_factory, split_config):
    split_config("2025-12-31", 70.0)
    r = _predict(session_factory, id=4, start_date="2027-01-01", end_date="2027-12-31")
    _check_contract(r, "2027-01-01", "2027-12-31", 4)
    assert 2026 in r["modelo"]["anios_completados_con_pronostico"]
    assert -0.10 < r["growth"]["2027"] < 0.15


def test_read_raw_daily_session_igual_que_entrenamiento(session_factory):
    from api.services.forecast_history import read_raw_daily_session
    from training.common.data import read_raw_daily

    db = session_factory()
    try:
        a = read_raw_daily_session(db)
    finally:
        db.close()
    b = read_raw_daily()
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_fechas_formato_main(session_factory):
    """Regresión de contrato: las strings de ``fechas`` coinciden con las de ``main`` (verificado contra una
    ejecución de main): ids 1-3 fin de mes, id 4 inicio de mes."""
    esperado = {1: "2026-01-31", 2: "2026-01-31", 3: "2026-01-31", 4: "2026-01-01"}
    for mid, primero in esperado.items():
        r = _predict(session_factory, id=mid, start_date="2026-01-01", end_date="2026-03-31")
        f = r["monthly_predictions"]["fechas"]
        assert f[0] == primero
        assert f[1] == ("2026-02-28" if mid != 4 else "2026-02-01")
