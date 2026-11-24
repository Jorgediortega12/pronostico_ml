"""Modelo id=4 (Climatico+ONI) v3: nivel reciente + deriva + sensibilidad ENSO, en base MENSUAL con origen.

Motivacion (ver ``docs/work/09_clima_v3/``): v2 era anual (deriva + beta*ONI_anual) con reparto mensual de
promedio de 2 anios: no usa los meses ya observados del anio en curso ni el nivel reciente. v3 trabaja sobre
el log de la demanda mensual ``y_t`` desde el ULTIMO MES OBSERVADO (``origen``):

* **Familia A (yoy recursivo)**: ``y_t = y_{t-12} + g + beta*(ONI_t - ONI_{t-12})`` (la base del mes ``t-12`` es
  el valor observado si ``t-12 <= origen`` y el pronostico recursivo si no).
* **Familia B (regresion de nivel por ventana)**: sobre los ultimos ``window`` meses
  ``y_t - beta*ONI_t = a + g*tau_t + estacionalidad_mes`` (12 efectos mensuales; ``tau`` en anios desde el
  origen); la pendiente ``g`` es la deriva (``slope='fixed'``) o la estimada por OLS (``slope='ols'``).

``beta`` (sensibilidad ENSO) se estima por OLS **expansivo** con las diferencias interanuales ya observadas
(``dy = Delta12 y`` contra ``dO = Delta12 ONI``, sin pares afectados por COVID), restringida a ``>= 0`` (signo
fisico) y opcionalmente encogida. La deriva ``g`` mezcla (``w_recent``) la media reciente de los ultimos
``k_recent`` meses de ``dy - beta*dO`` con la media larga. Todo se calcula solo con datos ``<= origen``; el ONI
futuro es ``project_oni_future`` (identico a la inferencia v2); los escenarios ENSO sustituyen el ONI de los
meses forzados por la mediana de su fase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from training.common import config
from training.forecast_climate import features as climate_features

#: Mediana historica de ONI por fase (misma que ``api.services.forecast.ENSO_PHASE_ONI``).
ENSO_PHASE_ONI = {"NIÑO": 0.9, "NIÑA": -0.8, "NORMAL": -0.1}


@dataclass(frozen=True)
class V3Spec:
    family: str = "A"  # "A" yoy recursivo | "B" regresion de nivel por ventana
    k_recent: int = 12  # meses para la deriva reciente
    w_recent: float = 1.0  # peso de la deriva reciente (1 - w = deriva larga)
    beta_mode: str = "ols"  # "ols" (expansivo) | "none" | "fixed"
    beta_shrink: float = 1.0  # multiplicador de beta_ols (encogimiento hacia 0)
    beta_fixed: float = 0.0
    beta_window: Optional[int] = None  # pares yoy mas recientes para estimar beta (None = todos)
    oni_lag: int = 0  # ONI usado = ONI_{t-oni_lag}
    window: int = 48  # familia B: meses de la regresion
    slope: str = "fixed"  # familia B: "fixed" (=g) | "ols"
    drift_cap: Optional[float] = None  # tope |g| anual (log) por seguridad; None = sin tope

    def label(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())


def _covid_pair_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Pares yoy cuyo mes objetivo o base cae en COVID (marzo 2020 .. febrero 2021 y el rebote 12 meses despues)."""
    lo = pd.Timestamp(config.COVID_EXCLUDED[0] + "-01")
    hi = pd.Timestamp(config.COVID_EXCLUDED[1] + "-01") + pd.DateOffset(years=1)
    return np.asarray((index >= lo) & (index <= hi))


def _oni_series(hist: pd.DataFrame, horizon: int, enso: Optional[dict]) -> pd.Series:
    """ONI observado (<= origen) + proyectado (``project_oni_future``) + override de fase, indexado por mes."""
    h = hist.sort_values(["year", "month"])
    idx = pd.DatetimeIndex(pd.to_datetime(dict(year=h["year"], month=h["month"], day=1)))
    obs = pd.Series(h["oni_actual"].to_numpy(float), index=idx)
    proj = climate_features.project_oni_future(h, meses_futuros=horizon, columna_oni="oni_actual", n_years_clim=10)
    pidx = pd.DatetimeIndex(pd.to_datetime(dict(year=proj["year"], month=proj["month"], day=1)))
    fut = pd.Series(proj["oni_proyectado"].to_numpy(float), index=pidx)
    oni = pd.concat([obs, fut])
    if enso:
        for key, phase in enso.items():
            ts = pd.Timestamp(f"{key}-01")
            if ts in oni.index and ts > obs.index.max():
                oni.loc[ts] = ENSO_PHASE_ONI[phase]
    return oni


def estimate_beta(y: pd.Series, oni: pd.Series, spec: V3Spec) -> float:
    """OLS de ``Delta12 y`` sobre ``Delta12 ONI`` (con intercepto) con datos observados; ``>= 0`` y encogido."""
    if spec.beta_mode == "none":
        return 0.0
    if spec.beta_mode == "fixed":
        return float(spec.beta_fixed)
    dy = y.diff(12)
    o = oni.shift(spec.oni_lag) if spec.oni_lag else oni
    do = (o - o.shift(12)).reindex(dy.index)
    d = pd.DataFrame({"dy": dy, "do": do}).dropna()
    d = d[~_covid_pair_mask(d.index)]
    if spec.beta_window:
        d = d.iloc[-spec.beta_window:]
    if len(d) < 24 or d["do"].std() == 0:
        return 0.0
    X = np.c_[np.ones(len(d)), d["do"].to_numpy()]
    b = np.linalg.lstsq(X, d["dy"].to_numpy(), rcond=None)[0][1]
    return float(max(b, 0.0) * spec.beta_shrink)


def estimate_drift(y: pd.Series, oni: pd.Series, beta: float, spec: V3Spec) -> float:
    """Deriva anual (log) = mezcla de media reciente y larga de ``dy - beta*dO`` (pares observados sin COVID)."""
    dy = y.diff(12)
    o = oni.shift(spec.oni_lag) if spec.oni_lag else oni
    do = (o - o.shift(12)).reindex(dy.index)
    r = (dy - beta * do).dropna()
    keep = ~_covid_pair_mask(r.index)
    r_ok = r[keep]
    long_ = float(r_ok.mean()) if len(r_ok) else 0.0
    rec = r_ok.iloc[-spec.k_recent:]
    recent = float(rec.mean()) if len(rec) else long_
    g = spec.w_recent * recent + (1 - spec.w_recent) * long_
    if spec.drift_cap is not None:
        g = float(np.clip(g, -spec.drift_cap, spec.drift_cap))
    return g


def predict_v3(
    hist: pd.DataFrame,
    horizon: int,
    spec: V3Spec,
    enso: Optional[dict] = None,
    return_params: bool = False,
):
    """Pronostico mensual (kWh) de ``horizon`` meses tras el ultimo mes de ``hist``.

    ``hist``: columnas ``year, month, demand_value, oni_actual`` con meses contiguos hasta el origen."""
    h = hist.sort_values(["year", "month"]).reset_index(drop=True)
    idx = pd.DatetimeIndex(pd.to_datetime(dict(year=h["year"], month=h["month"], day=1)))
    y = pd.Series(np.log(h["demand_value"].to_numpy(float)), index=idx)
    if not y.index.is_monotonic_increasing or (len(y) > 1 and (np.diff(y.index.year * 12 + y.index.month) != 1).any()):
        raise ValueError("el historico debe ser mensual contiguo")
    origin = y.index[-1]
    oni = _oni_series(h, horizon, enso)
    onil = oni.shift(spec.oni_lag) if spec.oni_lag else oni
    beta = estimate_beta(y, oni, spec)
    g = estimate_drift(y, oni, beta, spec)
    targets = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    if spec.family == "A":
        yy = y.copy()
        for t in targets:
            base = yy.loc[t - pd.DateOffset(years=1)]
            d_o = float(onil.loc[t] - onil.loc[t - pd.DateOffset(years=1)])
            yy.loc[t] = base + g + beta * d_o
        out = np.exp(yy.loc[targets])
    elif spec.family == "B":
        win = y.iloc[-spec.window:]
        covid_lo, covid_hi = (pd.Timestamp(m + "-01") for m in config.COVID_EXCLUDED)
        win = win[~((win.index >= covid_lo) & (win.index <= covid_hi))]
        tau = ((win.index.year - origin.year) * 12 + (win.index.month - origin.month)) / 12.0
        yadj = win.to_numpy() - beta * onil.reindex(win.index).to_numpy()
        months = win.index.month
        dummies = np.eye(12)[months - 1]
        if spec.slope == "fixed":
            slope = g
            coef = np.linalg.lstsq(dummies, yadj - slope * tau, rcond=None)[0]
        else:
            X = np.c_[dummies, tau]
            coef_all = np.linalg.lstsq(X, yadj, rcond=None)[0]
            coef, slope = coef_all[:12], float(coef_all[12])
            if spec.drift_cap is not None:
                slope = float(np.clip(slope, -spec.drift_cap, spec.drift_cap))
                coef = np.linalg.lstsq(dummies, yadj - slope * tau, rcond=None)[0]
        tf = ((targets.year - origin.year) * 12 + (targets.month - origin.month)) / 12.0
        yf = coef[targets.month - 1] + slope * tf + beta * onil.reindex(targets).to_numpy()
        out = pd.Series(np.exp(yf), index=targets)
        g = slope
    else:
        raise ValueError(f"familia desconocida: {spec.family}")
    res = pd.Series(np.asarray(out, float), index=targets, name="prediccion")
    if return_params:
        return res, {"beta": beta, "g": g}
    return res


# --------------------------------------------------------------------------------------------------
# Inferencia para la API (``DemandPredictor.predict_future_climate(model_version="v3")``)
# --------------------------------------------------------------------------------------------------
MIN_HISTORY_MONTHS = 36  # historia minima (>= 3 anios) para ajustar nivel + estacionalidad


def spec_from_dict(d: dict) -> V3Spec:
    fields = {f for f in V3Spec.__dataclass_fields__}
    return V3Spec(**{k: v for k, v in d.items() if k in fields})


def predict_years(
    hist_monthly: pd.DataFrame,
    cutoff_year: int,
    forecast_years: int,
    spec: V3Spec,
    start_month: int = 1,
    enso: Optional[dict] = None,
) -> pd.DataFrame:
    """Pronostico mensual de los anios ``cutoff_year+1 .. cutoff_year+forecast_years`` (12 meses cada uno).

    El **origen** es el ultimo mes observado anterior a ``(cutoff_year+1, start_month)``: con ``start_month=1`` el
    modelo usa solo anios completos ``<= cutoff_year`` (como v2) y con ``start_month=m`` ya usa los meses
    ``1..m-1`` observados de ``cutoff_year+1`` (misma semantica que el walk-forward con origen mensual y que los
    modelos 1-3: solo lo conocido antes de ``start_date``). Si el historico termina antes del origen pedido (p. ej.
    ``cutoff_year`` = ultimo anio parcial), el origen es el ultimo mes disponible y los meses faltantes se
    pronostican. Los meses ``<= origen`` que caen en el primer anio devuelto son los valores observados.

    Devuelve ``fecha, prediccion, observado`` (``observado`` = True si el mes es dato real, no pronostico).
    """
    if not 1 <= int(start_month) <= 12:
        raise ValueError(f"start_month fuera de rango: {start_month}")
    h = hist_monthly.sort_values(["year", "month"]).reset_index(drop=True)
    first_forecast_key = (cutoff_year + 1) * 12 + (int(start_month) - 1)  # year*12 + (month-1) del 1er mes a pronosticar
    h = h[(h["year"] * 12 + h["month"] - 1) < first_forecast_key]
    if len(h) < MIN_HISTORY_MONTHS:
        raise ValueError(
            f"No hay histórico suficiente para pronosticar desde {cutoff_year + 1}-{int(start_month):02d} "
            f"({len(h)} meses < {MIN_HISTORY_MONTHS})."
        )
    origin = pd.Timestamp(year=int(h["year"].iloc[-1]), month=int(h["month"].iloc[-1]), day=1)
    last_target_year = cutoff_year + forecast_years
    horizon = (last_target_year - origin.year) * 12 + (12 - origin.month)
    if horizon < 1:
        raise ValueError("el histórico ya cubre todo el horizonte pedido")
    fc = predict_v3(h, horizon, spec, enso=enso)
    obs = pd.Series(h["demand_value"].to_numpy(float),
                    index=pd.DatetimeIndex(pd.to_datetime(dict(year=h["year"], month=h["month"], day=1))))
    rows = []
    for yr in range(cutoff_year + 1, last_target_year + 1):
        for m in range(1, 13):
            ts = pd.Timestamp(year=yr, month=m, day=1)
            if ts in fc.index:
                rows.append((ts, float(fc.loc[ts]), False))
            elif ts in obs.index:
                rows.append((ts, float(obs.loc[ts]), True))
            else:  # hueco previo al historico disponible: no deberia ocurrir (historico contiguo)
                raise ValueError(f"mes {ts:%Y-%m} fuera del historico y del horizonte")
    return pd.DataFrame(rows, columns=["fecha", "prediccion", "observado"])
