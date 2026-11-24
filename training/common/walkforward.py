"""Harness genérico de validación walk-forward (expanding window) para series mensuales.

Contrato del modelo::

    fit_predict_fn(train: pd.Series, horizon: int) -> array-like | pd.Series

* ``train`` es una **copia** de la serie truncada a meses ``<= origen`` (índice = inicio de mes, kWh). Es lo
  único que recibe: no hay forma de ver el futuro desde el harness (cero fuga). ``train.attrs["origen"]``
  trae el origen. Si el modelo usa exógenas, debe filtrarlas él mismo a ``<= train.index.max()`` (o a lo
  que se conocería en el origen).
* Debe devolver ``horizon`` valores para los meses ``origen+1 … origen+horizon`` (si devuelve una Series
  con índice de fechas, se alinea por fecha y debe cubrirlos todos).

Métricas (:func:`summarize`): MAPE y sesgo mensual, por tramo de horizonte, y MAPE del total anual, con
exclusión de meses (COVID, post-quiebre de mercado) solo en la evaluación.
"""
from __future__ import annotations

import warnings
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from training.common import config
from training.common.data import MarketBreak

FitPredictFn = Callable[[pd.Series, int], Union[Sequence[float], np.ndarray, pd.Series]]
Exclusion = tuple[pd.Timestamp, Optional[pd.Timestamp]]

BASELINE_NAIVE = "naive_estacional"
BASELINE_DRIFT = "naive_estacional_drift"


# --------------------------------------------------------------------------------------------------
# Utilidades de fechas
# --------------------------------------------------------------------------------------------------
def _month(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp()


def _as_monthly(series: pd.Series) -> pd.Series:
    s = series.astype(float).copy()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    if s.index.has_duplicates:
        raise ValueError("la serie tiene meses duplicados")
    return s.sort_index()


def make_origins(start: str, end: str, step_months: int = 1) -> list[pd.Timestamp]:
    """Orígenes mensuales (``step_months=1``) o anuales (``step_months=12``, p. ej. todos los diciembres)
    entre ``start`` y ``end`` inclusive. El origen es el **último mes observado** por el modelo."""
    rng = pd.date_range(_month(start), _month(end), freq="MS")
    return list(rng[::step_months])


# --------------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------------
def walk_forward(
    fit_predict_fn: FitPredictFn,
    series: pd.Series,
    origins: Iterable[Union[str, pd.Timestamp]],
    horizon_months: int,
    actuals: Optional[pd.Series] = None,
    model_name: str = "modelo",
    min_train_months: int = 24,
) -> pd.DataFrame:
    """Ejecuta el modelo en cada origen con ventana expansiva.

    Args:
        fit_predict_fn: ver contrato del módulo.
        series: serie de entrenamiento (puede incluir meses imputados o cualquier mes "válido para train").
        origins: últimos meses observados por el modelo en cada corrida.
        horizon_months: meses a pronosticar desde cada origen.
        actuals: serie contra la que se evalúa (por defecto ``series``). Útil para excluir imputados.
        model_name: etiqueta de la columna ``modelo``.
        min_train_months: orígenes con menos meses de train se omiten (con aviso).

    Returns:
        DataFrame ``modelo, origen, objetivo, h, real, pred, err_pct`` (``err_pct = (pred-real)/real·100``;
        ``real`` NaN si no hay dato observado).
    """
    s = _as_monthly(series)
    act = _as_monthly(actuals) if actuals is not None else s
    rows = []
    for origin in origins:
        origin = _month(origin)
        train = s.loc[s.index <= origin].copy()
        if len(train) < min_train_months:
            warnings.warn(f"origen {origin:%Y-%m}: solo {len(train)} meses de train; se omite.", stacklevel=2)
            continue
        train.attrs = {"origen": origin}
        targets = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
        pred = _coerce_prediction(fit_predict_fn(train, horizon_months), targets, model_name, origin)
        real = act.reindex(targets).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "modelo": model_name,
                    "origen": origin,
                    "objetivo": targets,
                    "h": np.arange(1, horizon_months + 1),
                    "real": real,
                    "pred": pred,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["modelo", "origen", "objetivo", "h", "real", "pred", "err_pct"])
    out = pd.concat(rows, ignore_index=True)
    out["err_pct"] = (out["pred"] - out["real"]) / out["real"] * 100
    return out


def _coerce_prediction(pred, targets: pd.DatetimeIndex, name: str, origin: pd.Timestamp) -> np.ndarray:
    if isinstance(pred, pd.Series) and isinstance(pred.index, pd.DatetimeIndex):
        p = _as_monthly(pred).reindex(targets)
        if p.isna().any():
            raise ValueError(f"{name} @ {origin:%Y-%m}: la predicción no cubre todos los meses objetivo")
        return p.to_numpy(dtype=float)
    arr = np.asarray(pred, dtype=float).ravel()
    if arr.shape[0] != len(targets):
        raise ValueError(f"{name} @ {origin:%Y-%m}: se esperaban {len(targets)} valores, llegaron {arr.shape[0]}")
    return arr


# --------------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------------
def _same_month_base(train: pd.Series, targets: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Para cada objetivo: último valor del mismo mes calendario en train y años transcurridos."""
    base, years = [], []
    for t in targets:
        same = train[train.index.month == t.month]
        if same.empty:
            raise ValueError(f"no hay historia para el mes {t.month}")
        m = same.index.max()
        base.append(same.loc[m])
        years.append(t.year - m.year)
    return np.asarray(base, dtype=float), np.asarray(years, dtype=float)


def _targets(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    origin = train.attrs.get("origen", train.index.max())
    return pd.date_range(_month(origin) + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Naive estacional: cada mes = último valor observado del mismo mes calendario."""
    base, _ = _same_month_base(train, _targets(train, horizon))
    return base


def drift_rate(train: pd.Series, years: int = config.DRIFT_YEARS) -> float:
    """Media de los crecimientos interanuales mensuales (y_t / y_{t-12} − 1) de los últimos ``years`` años
    del train (solo pares con ambos meses observados). 0 si no hay pares."""
    s = train.asfreq("MS")
    yoy = s / s.shift(12) - 1
    last = s.index.max()
    window = yoy.loc[yoy.index > last - pd.DateOffset(years=years)].dropna()
    return float(window.mean()) if len(window) else 0.0


def seasonal_naive_drift(train: pd.Series, horizon: int, years: int = config.DRIFT_YEARS) -> np.ndarray:
    """Naive estacional × (1 + drift)^años, con drift = :func:`drift_rate` de los últimos ``years`` años."""
    base, yrs = _same_month_base(train, _targets(train, horizon))
    return base * (1 + drift_rate(train, years)) ** yrs


BASELINES: dict[str, FitPredictFn] = {BASELINE_NAIVE: seasonal_naive, BASELINE_DRIFT: seasonal_naive_drift}


# --------------------------------------------------------------------------------------------------
# Exclusiones y métricas
# --------------------------------------------------------------------------------------------------
def default_exclusions(
    market_break: Optional[MarketBreak] = None, covid: Optional[tuple[str, str]] = config.COVID_EXCLUDED
) -> list[Exclusion]:
    """Periodos excluidos de las métricas: COVID (config) y todo mes desde el primer mes post-quiebre."""
    exc: list[Exclusion] = []
    if covid:
        exc.append((_month(covid[0]), _month(covid[1])))
    if market_break is not None:
        exc.append((market_break.first_post_month, None))
    return exc


def excluded_mask(months: Union[pd.Series, pd.DatetimeIndex], exclusions: Sequence[Exclusion]) -> np.ndarray:
    """``True`` si el mes cae en algún periodo excluido (fin ``None`` = abierto)."""
    idx = pd.DatetimeIndex(months)
    mask = np.zeros(len(idx), dtype=bool)
    for start, end in exclusions:
        m = idx >= start
        if end is not None:
            m &= idx <= end
        mask |= np.asarray(m)
    return mask


def _metrics(err_pct: pd.Series) -> dict[str, float]:
    return {
        "n": int(err_pct.size),
        "mape": float(err_pct.abs().mean()) if err_pct.size else np.nan,
        "sesgo": float(err_pct.mean()) if err_pct.size else np.nan,
        "max_ape": float(err_pct.abs().max()) if err_pct.size else np.nan,
    }


def evaluable(results: pd.DataFrame, exclusions: Sequence[Exclusion] = (), common_support: bool = True) -> pd.DataFrame:
    """Filas evaluables: real y pred presentes, objetivo no excluido y (si ``common_support``) presentes
    para todos los modelos, para comparar sobre exactamente los mismos puntos."""
    r = results.copy()
    ok = r["real"].notna() & r["pred"].notna() & ~excluded_mask(r["objetivo"], exclusions)
    r = r[ok]
    if common_support and r["modelo"].nunique() > 1:
        n_models = r["modelo"].nunique()
        cnt = r.groupby(["origen", "objetivo"])["modelo"].transform("nunique")
        r = r[cnt == n_models]
    return r


def annual_totals(results: pd.DataFrame, exclusions: Sequence[Exclusion] = ()) -> pd.DataFrame:
    """Totales anuales por (modelo, origen, año) cuando los 12 meses del año están en el horizonte, todos
    con real y ninguno excluido. Columnas: ``modelo, origen, anio, h_max, real, pred, err_pct``."""
    r = results.copy()
    r["anio"] = pd.DatetimeIndex(r["objetivo"]).year
    r["ok"] = r["real"].notna() & r["pred"].notna() & ~excluded_mask(r["objetivo"], exclusions)
    g = r.groupby(["modelo", "origen", "anio"])
    agg = g.agg(n=("ok", "size"), n_ok=("ok", "sum"), h_max=("h", "max"), real=("real", "sum"), pred=("pred", "sum"))
    agg = agg[(agg["n"] == 12) & (agg["n_ok"] == 12)].reset_index()
    agg["err_pct"] = (agg["pred"] - agg["real"]) / agg["real"] * 100
    return agg[["modelo", "origen", "anio", "h_max", "real", "pred", "err_pct"]]


def summarize(
    results: pd.DataFrame,
    exclusions: Sequence[Exclusion] = (),
    buckets: Sequence[tuple[int, int]] = config.HORIZON_BUCKETS,
    common_support: bool = True,
) -> pd.DataFrame:
    """Resumen por modelo: MAPE/sesgo/máx mensual global y por tramo de horizonte, y MAPE/sesgo del total
    anual (global y por tramo según el ``h`` máximo del año). Errores en %, sesgo = media de
    (pred−real)/real (negativo = subestima)."""
    ev = evaluable(results, exclusions, common_support)
    ann = annual_totals(results, exclusions)
    if common_support and ann["modelo"].nunique() > 1:
        cnt = ann.groupby(["origen", "anio"])["modelo"].transform("nunique")
        ann = ann[cnt == ann["modelo"].nunique()]
    rows = []
    for name in results["modelo"].unique():
        e = ev[ev["modelo"] == name]
        a = ann[ann["modelo"] == name]
        row: dict[str, object] = {"modelo": name, "n_origenes": int(e["origen"].nunique())}
        row.update({f"{k}": v for k, v in _metrics(e["err_pct"]).items()})
        for lo, hi in buckets:
            sub = e[(e["h"] >= lo) & (e["h"] <= hi)]["err_pct"]
            met = _metrics(sub)
            row[f"mape_h{lo}_{hi}"] = met["mape"]
            row[f"sesgo_h{lo}_{hi}"] = met["sesgo"]
            row[f"n_h{lo}_{hi}"] = met["n"]
        am = _metrics(a["err_pct"])
        row.update({"n_anual": am["n"], "mape_anual": am["mape"], "sesgo_anual": am["sesgo"]})
        for lo, hi in buckets:
            sub = a[(a["h_max"] >= lo) & (a["h_max"] <= hi)]["err_pct"]
            row[f"mape_anual_h{lo}_{hi}"] = _metrics(sub)["mape"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_with_baselines(
    fit_predict_fn: Optional[FitPredictFn],
    series: pd.Series,
    origins: Iterable[Union[str, pd.Timestamp]],
    horizon_months: int,
    actuals: Optional[pd.Series] = None,
    exclusions: Sequence[Exclusion] = (),
    model_name: str = "modelo",
    baselines: Optional[dict[str, FitPredictFn]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre el modelo y los baselines sobre los mismos orígenes y compara sobre el mismo soporte.

    Returns:
        ``(resultados, resumen)``. El resumen añade ``mape_vs_<baseline>`` (cociente MAPE modelo/baseline;
        < 1 = mejor) y ``supera_baselines`` (MAPE menor que todos los baselines). Con ``fit_predict_fn=None``
        solo corre los baselines.
    """
    origins = list(origins)
    baselines = BASELINES if baselines is None else baselines
    runs = [walk_forward(fn, series, origins, horizon_months, actuals, name) for name, fn in baselines.items()]
    if fit_predict_fn is not None:
        runs.insert(0, walk_forward(fit_predict_fn, series, origins, horizon_months, actuals, model_name))
    results = pd.concat(runs, ignore_index=True)
    summary = summarize(results, exclusions)
    base_mape = summary.set_index("modelo")["mape"]
    for b in baselines:
        summary[f"mape_vs_{b}"] = summary["mape"] / base_mape.get(b, np.nan)
    best_base = min((base_mape.get(b, np.inf) for b in baselines), default=np.inf)
    supera = (summary["mape"] < best_base).astype(object)
    supera[summary["modelo"].isin(list(baselines))] = None  # no aplica a los propios baselines
    summary["supera_baselines"] = supera
    return results, summary
