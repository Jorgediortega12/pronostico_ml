"""Tests de training.common.data con datos sintéticos (sin BD)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from training.common.data import (
    MarketBreak,
    aggregate_monthly,
    deduplicate_daily,
    from_full_market_scale,
    target_series,
    to_full_market_scale,
)


def _raw(start: str, end: str, level: float = 2e7, file_type: str = "txf") -> pd.DataFrame:
    fechas = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"id": np.arange(len(fechas)), "fecha": fechas, "total": level * (1 + 0.05 * rng.standard_normal(len(fechas))),
         "file_type": file_type}
    )


def test_dedupe_elige_fila_de_escala_correcta_aunque_tenga_id_mayor_o_menor():
    raw = _raw("2020-09-01", "2021-01-31")
    dup_dates = pd.date_range("2020-10-01", "2020-12-31", freq="D")
    dups = pd.DataFrame({"id": 10_000 + np.arange(len(dup_dates)), "fecha": dup_dates,
                         "total": 2e6, "file_type": "txf"})
    # En la mitad de las fechas la fila mala tiene id menor (no depende del orden).
    dups.loc[::2, "id"] = -1 - np.arange(len(dups.loc[::2]))
    daily, log = deduplicate_daily(pd.concat([raw, dups], ignore_index=True))
    assert daily.index.is_unique
    assert len(daily) == len(pd.date_range("2020-09-01", "2021-01-31"))
    assert (daily["demanda_kwh"] > 1e7).all()
    assert len(log) == len(dup_dates)
    assert (log["razon_max_min"] > 5).all()
    assert (log["elegido"] > 1e7).all()


def test_dedupe_prefiere_definitivo_sobre_preliminar():
    raw = _raw("2024-01-01", "2024-03-31")
    extra = pd.DataFrame({"id": [999], "fecha": [pd.Timestamp("2024-02-10")], "total": [2.0e7], "file_type": ["tx2"]})
    daily, log = deduplicate_daily(pd.concat([raw, extra], ignore_index=True))
    assert daily.loc["2024-02-10", "file_type"] == "txf"
    assert log.iloc[0]["criterio"] == "prefiere_definitivo"


def test_mes_incompleto_es_nan_y_se_imputa_solo_hueco_interno():
    raw = _raw("2021-07-01", "2021-11-17")
    raw = raw[~raw["fecha"].isin(pd.to_datetime(["2021-09-01", "2021-09-02"]))]
    daily, _ = deduplicate_daily(raw)
    m = aggregate_monthly(daily)
    assert not m.loc["2021-09-01", "completo"]
    assert np.isnan(m.loc["2021-09-01", "demanda_kwh"])
    assert np.isnan(m.loc["2021-11-01", "demanda_kwh"])  # último mes parcial
    mi = aggregate_monthly(daily, impute_incomplete=True)
    sep = mi.loc["2021-09-01"]
    assert sep["imputado"]
    assert sep["demanda_kwh"] == pytest.approx(sep["demanda_observada_kwh"] / 28 * 30)
    # Nunca se imputa el mes final parcial (no hay mes completo posterior) ni se crean meses futuros.
    assert not mi.loc["2021-11-01", "imputado"]
    assert np.isnan(mi.loc["2021-11-01", "demanda_kwh"])
    assert mi.index.max() == pd.Timestamp("2021-11-01")


def test_mes_preliminar_no_es_target_ni_se_imputa():
    raw = pd.concat([_raw("2026-08-01", "2026-10-31"), _raw("2026-11-01", "2026-11-17", file_type="tx2")])
    daily, _ = deduplicate_daily(raw)
    m = aggregate_monthly(daily, impute_incomplete=True)
    nov = m.loc["2026-11-01"]
    assert nov["preliminar"] and not nov["completo"] and not nov["imputado"]
    assert np.isnan(nov["demanda_kwh"])
    ts = target_series(m)
    assert ts.index.max() == pd.Timestamp("2026-10-01")


def test_target_series_excluye_imputados_para_metricas():
    raw = _raw("2021-07-01", "2021-10-31")
    raw = raw[raw["fecha"] != pd.Timestamp("2021-09-01")]
    m = aggregate_monthly(deduplicate_daily(raw)[0], impute_incomplete=True)
    assert pd.Timestamp("2021-09-01") in target_series(m).index
    assert pd.Timestamp("2021-09-01") not in target_series(m, include_imputed=False).index


def test_market_break_semantica_igual_a_api():
    mb = MarketBreak(date(2026, 8, 31), 70.0)
    assert mb.first_post_month == pd.Timestamp("2026-09-01")
    mb2 = MarketBreak(date(2026, 9, 1), 70.0)  # sep termina después de 09-01 -> post
    assert mb2.first_post_month == pd.Timestamp("2026-09-01")
    idx = pd.date_range("2026-07-01", periods=4, freq="MS")
    s = pd.Series([100.0, 100.0, 70.0, 70.0], index=idx)
    full = to_full_market_scale(s, mb)
    assert full.tolist() == pytest.approx([100, 100, 100, 100])
    assert from_full_market_scale(full, mb).tolist() == pytest.approx(s.tolist())
    assert target_series(pd.DataFrame({"demanda_kwh": s, "imputado": False}), market_break=mb).index.max() == pd.Timestamp(
        "2026-08-01"
    )


def test_to_full_market_scale_100_por_ciento_es_noop_con_aviso():
    s = pd.Series([1.0, 2.0], index=pd.date_range("2026-08-01", periods=2, freq="MS"))
    with pytest.warns(UserWarning):
        out = to_full_market_scale(s, MarketBreak(date(2026, 8, 31), 100.0))
    assert out.tolist() == [1.0, 2.0]
