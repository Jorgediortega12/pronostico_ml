"""Utilidades compartidas de los scripts de id=3: datos, walk-forward en paralelo, original reevaluado.

Reglas: la selección solo mira orígenes <= SELECTION_LAST_ORIGIN y reales <= DEV_LAST_TARGET (2026 = test final).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.services.forecast_sarimax import ModelSpec, fit_predict  # noqa: E402
from training.common.data import load_monthly_demand, target_series  # noqa: E402
from training.common.walkforward import (  # noqa: E402
    BASELINES,
    default_exclusions,
    make_origins,
    summarize,
    walk_forward,
)

OUT = ROOT / "docs" / "work" / "03_sarimax"
SELECTION_ORIGINS = ("2017-12", "2023-12")  # selección (targets <= 2025-12)
DEV_ORIGINS = ("2017-12", "2025-12")  # reporte dev (targets <= 2025-12)
DEV_LAST_TARGET = "2025-12"
AIC_TRAIN_END = "2023-12"
HORIZON = 24
N_JOBS = -2


def load_data():
    """Devuelve ``(train_full, actuals_dev, actuals_all, market_break)``.

    train_full: serie limpia con imputados, cortada en el quiebre de mercado (escala completa).
    actuals_dev: reales observados (sin imputados) <= DEV_LAST_TARGET.
    actuals_all: reales observados pre-quiebre (incluye 2026; SOLO para el test final).
    """
    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    train = target_series(monthly, market_break=mb)
    act_all = target_series(monthly, include_imputed=False, market_break=mb)
    act_dev = act_all.loc[:DEV_LAST_TARGET]
    return train, act_dev, act_all, mb


def _one_origin(fn: Callable, series: pd.Series, origin, horizon: int, actuals: pd.Series, name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return walk_forward(fn, series, [origin], horizon, actuals, name)


def parallel_walk_forward(fn: Callable, series, origins, horizon, actuals, name, n_jobs=N_JOBS) -> pd.DataFrame:
    """``walk_forward`` paralelizado por origen (mismo resultado que la versión secuencial)."""
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_one_origin)(fn, series, o, horizon, actuals, name) for o in origins
    )
    return pd.concat(parts, ignore_index=True)


class SpecFn:
    """Callable picklable para joblib."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def __call__(self, train: pd.Series, h: int):
        return fit_predict(train, h, self.spec)


class EnsembleFn:
    """Media en log de varios specs (pesos iguales)."""

    def __init__(self, specs):
        self.specs = list(specs)

    def __call__(self, train: pd.Series, h: int):
        preds = [fit_predict(train, h, s) for s in self.specs]
        return pd.Series(np.exp(np.mean([np.log(p.to_numpy()) for p in preds], axis=0)), index=preds[0].index)


class OriginalFn:
    """Original reevaluado honestamente: SARIMAX(5,1,6)(1,1,1,12) sobre log1p con 6 macro anuales (RobustScaler
    ajustado en train) como exógenas; a futuro la macro del último año del train constante (como en producción).
    Se reajusta en cada origen. Usa la serie limpia (2020-Q4 deduplicado), igual que los candidatos."""

    MACRO_CSV = ROOT / "api" / "data" / "gmr_annual_es.csv"

    def __init__(self, use_macro: bool = True):
        self.use_macro = use_macro
        macro = pd.read_csv(self.MACRO_CSV).set_index("año")
        self.macro = macro.drop(columns=["total"])

    def __call__(self, train: pd.Series, h: int):
        from sklearn.preprocessing import RobustScaler
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        s = train.asfreq("MS")
        y = np.log1p(s)
        idx_f = pd.date_range(s.index.max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        exog = exog_f = None
        if self.use_macro:
            last_year = min(int(s.index.max().year), int(self.macro.index.max()))
            X = self.macro.reindex(s.index.year).ffill().to_numpy()
            Xf = np.repeat(self.macro.loc[[last_year]].to_numpy(), h, axis=0)
            sc = RobustScaler().fit(X)
            exog, exog_f = sc.transform(X), sc.transform(Xf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y, exog=exog, order=(5, 1, 6), seasonal_order=(1, 1, 1, 12)).fit(disp=False, maxiter=200)
            f = res.forecast(steps=h, exog=exog_f)
        return pd.Series(np.expm1(np.asarray(f)), index=idx_f)


def baseline_runs(series, origins, horizon, actuals) -> list[pd.DataFrame]:
    return [walk_forward(fn, series, origins, horizon, actuals, name) for name, fn in BASELINES.items()]


__all__ = [
    "ModelSpec",
    "load_data",
    "parallel_walk_forward",
    "SpecFn",
    "EnsembleFn",
    "OriginalFn",
    "baseline_runs",
    "summarize",
    "default_exclusions",
    "make_origins",
    "OUT",
    "SELECTION_ORIGINS",
    "DEV_ORIGINS",
    "AIC_TRAIN_END",
    "HORIZON",
]
