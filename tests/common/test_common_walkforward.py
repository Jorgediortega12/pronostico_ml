"""Tests del harness walk-forward con series sintéticas (sin BD)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from training.common.data import MarketBreak
from training.common.walkforward import (
    BASELINE_DRIFT,
    BASELINE_NAIVE,
    annual_totals,
    default_exclusions,
    drift_rate,
    evaluate_with_baselines,
    make_origins,
    seasonal_naive,
    seasonal_naive_drift,
    summarize,
    walk_forward,
)


def _serie(start="2012-01-01", end="2026-10-01", growth=0.03) -> pd.Series:
    idx = pd.date_range(start, end, freq="MS")
    t = np.arange(len(idx))
    season = 1 + 0.1 * np.sin(2 * np.pi * idx.month / 12)
    return pd.Series(1e8 * (1 + growth) ** (t / 12) * season, index=idx)


def test_cero_fuga_el_train_nunca_ve_despues_del_origen():
    s = _serie()
    vistos = []

    def espia(train: pd.Series, h: int):
        vistos.append((train.attrs["origen"], train.index.max()))
        return np.repeat(train.iloc[-1], h)

    origins = make_origins("2017-12", "2025-12")
    out = walk_forward(espia, s, origins, 24)
    assert len(vistos) == len(origins)
    assert all(ultimo <= origen for origen, ultimo in vistos)
    # El pred "último valor" debe ser exactamente el valor del origen (no uno futuro).
    for origen, grp in out.groupby("origen"):
        assert grp["pred"].iloc[0] == pytest.approx(s.loc[origen])


def test_mutar_el_train_no_altera_la_serie_original():
    s = _serie()
    copia = s.copy()

    def vandalo(train, h):
        train.iloc[:] = 0.0
        return np.zeros(h)

    walk_forward(vandalo, s, ["2020-12"], 12)
    pd.testing.assert_series_equal(s, copia)


def test_objetivos_h_y_real_alineados():
    s = _serie(end="2026-08-01")
    out = walk_forward(seasonal_naive, s, ["2025-12"], 24)
    assert out["objetivo"].iloc[0] == pd.Timestamp("2026-01-01")
    assert out["h"].tolist() == list(range(1, 25))
    assert out.loc[out["objetivo"] > "2026-08-01", "real"].isna().all()


def test_prediccion_con_longitud_incorrecta_falla():
    with pytest.raises(ValueError):
        walk_forward(lambda tr, h: np.zeros(h - 1), _serie(), ["2020-12"], 12)


def test_naive_estacional_y_drift():
    s = _serie(growth=0.05)
    train = s.loc[:"2019-12-01"]
    train.attrs = {"origen": pd.Timestamp("2019-12-01")}
    naive = seasonal_naive(train, 24)
    assert naive[0] == pytest.approx(s.loc["2019-01-01"])
    assert naive[12] == pytest.approx(s.loc["2019-01-01"])
    assert drift_rate(train) == pytest.approx(0.05, rel=1e-6)
    drift = seasonal_naive_drift(train, 24)
    # Serie con crecimiento geométrico exacto: el naive+drift la reproduce.
    assert drift == pytest.approx(s.loc["2020-01-01":"2021-12-01"].to_numpy(), rel=1e-9)


def test_naive_usa_ultimo_mismo_mes_disponible_si_hay_hueco():
    s = _serie().drop(pd.Timestamp("2021-09-01"))
    train = s.loc[:"2021-12-01"]
    train.attrs = {"origen": pd.Timestamp("2021-12-01")}
    pred = seasonal_naive(train, 12)
    assert pred[8] == pytest.approx(s.loc["2020-09-01"])


def test_exclusiones_covid_y_quiebre():
    mb = MarketBreak(date(2026, 8, 31), 70.0)
    exc = default_exclusions(mb)
    s = _serie()
    res = walk_forward(seasonal_naive, s, ["2019-12", "2025-12"], 24)
    summ = summarize(res, exc)
    from training.common.walkforward import evaluable

    ev = evaluable(res, exc)
    assert not ((ev["objetivo"] >= "2020-03-01") & (ev["objetivo"] <= "2021-02-01")).any()
    assert not (ev["objetivo"] >= "2026-09-01").any()
    assert summ.loc[0, "n"] == len(ev)


def test_metricas_mape_sesgo_y_anual():
    idx = pd.date_range("2010-01-01", "2020-12-01", freq="MS")
    s = pd.Series(100.0, index=idx)
    res = walk_forward(lambda tr, h: np.full(h, 110.0), s, ["2017-12", "2018-12"], 24)
    summ = summarize(res).iloc[0]
    assert summ["mape"] == pytest.approx(10.0)
    assert summ["sesgo"] == pytest.approx(10.0)
    assert summ["mape_h1_12"] == pytest.approx(10.0)
    ann = annual_totals(res)
    # origen 2017-12: 2018 y 2019 completos; origen 2018-12: 2019 y 2020.
    assert len(ann) == 4
    assert ann["err_pct"].tolist() == pytest.approx([10.0] * 4)
    assert summ["mape_anual_h1_12"] == pytest.approx(10.0)


def test_evaluate_with_baselines_compara_en_mismo_soporte():
    s = _serie(growth=0.04)
    res, summ = evaluate_with_baselines(
        lambda tr, h: seasonal_naive_drift(tr, h), s, make_origins("2017-12", "2019-12", 12), 24, model_name="m"
    )
    assert set(summ["modelo"]) == {"m", BASELINE_NAIVE, BASELINE_DRIFT}
    fila = summ.set_index("modelo").loc["m"]
    assert fila[f"mape_vs_{BASELINE_DRIFT}"] == pytest.approx(1.0)
    assert not fila["supera_baselines"]  # empata con el baseline drift, no lo supera
