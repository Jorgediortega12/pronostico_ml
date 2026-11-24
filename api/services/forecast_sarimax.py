"""Inferencia autocontenida del modelo id=3 (SARIMA/ETS mensual de demanda).

Reemplaza a ``DemandPredictor.predict_future_sarimax`` (``api/services/forecast.py``), que ignoraba
``start_date`` (siempre pronosticaba desde 2026-01 y rotulaba desde el último dato recibido).

Contrato
--------
``predict_monthly(history, start, end)``:

* ``history``: Serie mensual de demanda en kWh (índice fecha, cualquier día del mes; se normaliza a inicio de
  mes). Debe venir **limpia** (sin duplicados diarios, solo meses completos/definitivos); ver
  ``training.common.data`` (``load_monthly_demand`` + ``target_series``) o
  ``api.services.forecast_history.load_clean_history_session``.
* Solo se usan los meses **estrictamente anteriores** a ``start`` (y anteriores a ``exclude_from`` si se pasa,
  p. ej. el primer mes post-división de mercado). Nunca se usa un dato ``>= start``.
* El estado del modelo se re-ancla con esos reales (filtro de Kalman con los parámetros del artefacto,
  equivalente a ``res.apply``) y se pronostica desde el último mes real hasta ``end``; se devuelve solo
  ``[start, end]``, rotulado por fecha. Si los parámetros del artefacto se estimaron con datos ``>= start``
  (hindcast), se re-estiman con la historia disponible (``refit="auto"``) para que no haya fuga.
* Salida en **escala de mercado completo** (la del entrenamiento). Aplicar el factor de división de mercado
  es responsabilidad del llamador.

El mismo código (``ModelSpec``, ``fit_spec``, ``forecast_spec``) lo usan los scripts de entrenamiento en
``training/forecast_sarimax/`` → paridad exacta train/serve. Este módulo NO importa ``forecast.py``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from api.services.calendar_co import calendar_counts
from api.services.forecast_cache import RETRAIN_SEMAPHORE, SingleFlightLRU, series_digest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sarimax_model"
ARTIFACT_FILE = "sarimax_v2.json"

#: Periodo COVID (inclusive) tratado según ``ModelSpec.covid``.
COVID_PERIOD = ("2020-03", "2021-02")

DateLike = Union[str, pd.Timestamp, date]


# --------------------------------------------------------------------------------------------------
# Especificación
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Especificación de un modelo candidato.

    kind: ``"sarima"`` | ``"ets"``.
    transform: ``"log"`` (log y) | ``"log_daily"`` (log de la demanda media diaria: y / días del mes).
    covid: ``"none"`` | ``"nan"`` (meses COVID como faltantes: el filtro de Kalman los salta) |
        ``"dummy"`` (un regresor 0/1 en el periodo COVID; solo sarima).
    exog: nombres de regresores de calendario (ver :func:`calendar_features`); solo sarima.
    """

    kind: str = "sarima"
    order: tuple = (0, 1, 1)
    seasonal_order: tuple = (0, 1, 1, 12)
    trend: Optional[str] = None
    transform: str = "log"
    covid: str = "nan"
    exog: tuple = ()
    damped: bool = True  # solo ets

    def name(self) -> str:
        if self.kind == "ets":
            return f"ets_{'damped' if self.damped else 'add'}_{self.transform}_{self.covid}"
        ex = "+".join(self.exog) if self.exog else "noexog"
        tr = f"_t{self.trend}" if self.trend else ""
        return f"sarima{self.order}{self.seasonal_order[:3]}{tr}_{self.transform}_{self.covid}_{ex}".replace(" ", "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["order"] = list(self.order)
        d["seasonal_order"] = list(self.seasonal_order)
        d["exog"] = list(self.exog)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        d["order"] = tuple(d.get("order", (0, 1, 1)))
        d["seasonal_order"] = tuple(d.get("seasonal_order", (0, 1, 1, 12)))
        d["exog"] = tuple(d.get("exog", ()))
        return cls(**d)


# --------------------------------------------------------------------------------------------------
# Utilidades de fechas / calendario
# --------------------------------------------------------------------------------------------------
def _month(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp()


def _as_monthly(series: pd.Series) -> pd.Series:
    s = pd.Series(series, dtype=float).copy()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def calendar_features(index: pd.DatetimeIndex, names: Sequence[str]) -> pd.DataFrame:
    """Regresores de calendario deterministas (conocidos a cualquier horizonte → sin fuga).

    * ``frac_habiles``: fracción de días del mes que son lunes-viernes no festivos.
    * ``frac_domfest``: fracción de domingos + festivos.
    * ``bisiesto_feb``: 1 en febreros de 29 días (útil con transform ``log``).
    """
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)
    if not names:
        return out
    hab, domf = [], []
    for m in idx:
        # Mismo calendario que RF/XGB (api/services/calendar_co.py): laborable = lun-vie no festivo;
        # domingos + festivos (incluye festivos en sábado).
        dias, lab, _, domfest = calendar_counts(m.year, m.month)
        hab.append(float(lab) / dias)
        domf.append(float(domfest) / dias)
    feats = {
        "frac_habiles": np.asarray(hab),
        "frac_domfest": np.asarray(domf),
        "bisiesto_feb": np.asarray([(m.month == 2 and m.days_in_month == 29) * 1.0 for m in idx]),
    }
    for n in names:
        if n not in feats:
            raise ValueError(f"exógena desconocida: {n}")
        out[n] = feats[n]
    return out


def _covid_mask(index: pd.DatetimeIndex) -> np.ndarray:
    lo, hi = _month(COVID_PERIOD[0]), _month(COVID_PERIOD[1])
    return np.asarray((index >= lo) & (index <= hi))


def _exog_matrix(index: pd.DatetimeIndex, spec: ModelSpec) -> Optional[pd.DataFrame]:
    if spec.kind != "sarima":
        return None
    X = calendar_features(index, spec.exog)
    if spec.covid == "dummy":
        X["covid"] = _covid_mask(index).astype(float)
    return X if X.shape[1] else None


# --------------------------------------------------------------------------------------------------
# Transformación
# --------------------------------------------------------------------------------------------------
def _transform(s: pd.Series, spec: ModelSpec) -> pd.Series:
    if (s.dropna() <= 0).any():
        raise ValueError("la demanda debe ser > 0 para la transformación log")
    if spec.transform == "log":
        return np.log(s)
    if spec.transform == "log_daily":
        return np.log(s / s.index.days_in_month)
    raise ValueError(f"transformación desconocida: {spec.transform}")


def _inverse(z: pd.Series, spec: ModelSpec) -> pd.Series:
    if spec.transform == "log":
        return np.exp(z)
    return np.exp(z) * z.index.days_in_month


def prepare_endog(history: pd.Series, spec: ModelSpec) -> pd.Series:
    """Serie transformada con frecuencia mensual continua (huecos y COVID-``nan`` como NaN)."""
    s = _as_monthly(history).dropna()
    if s.empty:
        raise ValueError("historia vacía")
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full)
    z = _transform(s, spec)
    if spec.covid == "nan":
        z[_covid_mask(z.index)] = np.nan
    z.index.freq = "MS"
    return z


# --------------------------------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------------------------------
def _sarimax(z: pd.Series, spec: ModelSpec):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(
        z,
        exog=_exog_matrix(z.index, spec),
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        trend=spec.trend,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )


def _ets_endog(z: pd.Series) -> pd.Series:
    """ETS no admite NaN: los huecos se rellenan con el mismo mes del año previo + cambio medio anual."""
    z = z.copy()
    for t in z.index[z.isna()]:
        prev = t - pd.DateOffset(years=1)
        while prev in z.index and pd.isna(z.get(prev)):
            prev = prev - pd.DateOffset(years=1)
        if prev in z.index:
            z[t] = z[prev]
    return z.interpolate(limit_direction="both")


def _ets(z: pd.Series, spec: ModelSpec):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    return ETSModel(
        _ets_endog(z), error="add", trend="add", damped_trend=spec.damped, seasonal="add", seasonal_periods=12
    )


def build_model(z: pd.Series, spec: ModelSpec):
    if spec.kind == "sarima":
        return _sarimax(z, spec)
    if spec.kind == "ets":
        return _ets(z, spec)
    raise ValueError(f"kind desconocido: {spec.kind}")


def fit_spec(history: pd.Series, spec: ModelSpec):
    """Ajusta (MLE) el modelo con toda la ``history``. Devuelve el objeto results de statsmodels."""
    z = prepare_endog(history, spec)
    mod = build_model(z, spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.kind == "sarima":
            res = mod.fit(disp=False, maxiter=200)
        else:
            res = mod.fit(disp=False, maxiter=500)
    return res


def apply_params(history: pd.Series, spec: ModelSpec, params: Sequence[float]):
    """Re-ancla el estado: filtra ``history`` con parámetros fijos (equivalente a ``res.apply``)."""
    z = prepare_endog(history, spec)
    mod = build_model(z, spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.kind == "sarima":
            return mod.filter(np.asarray(params, dtype=float))
        return mod.smooth(np.asarray(params, dtype=float))


def forecast_spec(res, spec: ModelSpec, last_obs: pd.Timestamp, steps: int) -> pd.Series:
    """Pronóstico (kWh) de ``steps`` meses desde ``last_obs + 1``, rotulado por fecha."""
    idx = pd.date_range(_month(last_obs) + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.kind == "sarima":
            X = _exog_matrix(idx, spec)
            z = res.forecast(steps=steps, exog=X)
        else:
            z = res.forecast(steps=steps)
    z = pd.Series(np.asarray(z, dtype=float), index=idx)
    return _inverse(z, spec)


def fit_predict(history: pd.Series, horizon: int, spec: ModelSpec) -> pd.Series:
    """Ajusta con ``history`` y pronostica ``horizon`` meses (contrato del harness walk-forward)."""
    s = _as_monthly(history).dropna()
    res = fit_spec(s, spec)
    return forecast_spec(res, spec, s.index.max(), horizon)


# --------------------------------------------------------------------------------------------------
# Artefacto
# --------------------------------------------------------------------------------------------------
@dataclass
class Artifact:
    """Artefacto serializado en JSON (legible, sin pickles de statsmodels).

    components: lista de dicts ``{"spec": ModelSpec.to_dict(), "params": [...], "param_names": [...]}``;
    la predicción es la media geométrica (media en log) de los componentes con ``weights``.
    train_end: último mes usado para estimar los parámetros.
    """

    components: list
    weights: list
    train_end: str
    version: str = "2"
    metadata: dict = field(default_factory=dict)

    def specs(self) -> list[ModelSpec]:
        return [ModelSpec.from_dict(c["spec"]) for c in self.components]

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Artifact":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_artifact(model_dir: Union[str, Path, None] = None) -> Artifact:
    return Artifact.load(Path(model_dir or MODEL_DIR) / ARTIFACT_FILE)


#: Re-estimaciones en línea (hindcast) cacheadas: clave = hash corto de la historia + spec (los parámetros
#: MLE dependen solo de la historia usada, que ya refleja el ``start``).
REFIT_CACHE_MAX = 32
_REFIT_PARAMS_CACHE = SingleFlightLRU(REFIT_CACHE_MAX)


def refit_params_cached(hist: pd.Series, spec: ModelSpec) -> np.ndarray:
    """Parámetros MLE de ``spec`` con ``hist`` (cacheados; un solo cálculo por clave a la vez y como mucho
    :data:`~api.services.forecast_cache.MAX_CONCURRENT_RETRAINS` re-estimaciones simultáneas en el proceso)."""
    key = (series_digest(hist), json.dumps(spec.to_dict(), sort_keys=True))

    def _fit():
        with RETRAIN_SEMAPHORE:
            res = fit_spec(hist, spec)
        return np.asarray(res.params, dtype=float).copy()

    return _REFIT_PARAMS_CACHE.get_or_compute(key, _fit)


def combine(preds: Sequence[pd.Series], weights: Sequence[float]) -> pd.Series:
    """Media ponderada en log de varios pronósticos alineados."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    logs = np.vstack([np.log(p.to_numpy(dtype=float)) for p in preds])
    return pd.Series(np.exp(w @ logs), index=preds[0].index)


# --------------------------------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------------------------------
def predict_monthly(
    history_monthly_series: pd.Series,
    start: DateLike,
    end: DateLike,
    artifact: Optional[Artifact] = None,
    model_dir: Union[str, Path, None] = None,
    refit: Union[str, bool] = "auto",
    exclude_from: Optional[DateLike] = None,
    min_history_months: int = 36,
) -> pd.DataFrame:
    """Pronóstico mensual en ``[start, end]`` (meses inclusive) usando solo reales anteriores a ``start``.

    Args:
        history_monthly_series: demanda mensual limpia (kWh), índice fecha.
        start, end: primer y último mes a devolver (cualquier día del mes).
        artifact / model_dir: artefacto a usar (por defecto ``api/models/sarimax_model/sarimax_v2.json``).
        refit: ``"auto"`` re-estima parámetros solo si el artefacto vio meses ``>= start`` (hindcast);
            ``True`` siempre re-estima con la historia disponible; ``False`` nunca (solo re-ancla estado).
        exclude_from: primer mes que NO debe usarse aunque sea ``< start`` (p. ej. primer mes post-división de
            mercado, cuya escala es distinta).
        min_history_months: mínimo de meses reales requeridos.

    Returns:
        DataFrame con columnas ``fecha`` (inicio de mes), ``prediccion`` (kWh, escala mercado completo) y
        atributos ``attrs`` = {``ultimo_real``, ``refit``, ``modelo``}.
    """
    start_m, end_m = _month(start), _month(end)
    if end_m < start_m:
        raise ValueError("end debe ser >= start")
    art = artifact or load_artifact(model_dir)

    hist = _as_monthly(history_monthly_series).dropna()
    hist = hist[hist.index < start_m]
    if exclude_from is not None:
        hist = hist[hist.index < _month(exclude_from)]
    if len(hist) < min_history_months:
        raise ValueError(f"historia insuficiente antes de {start_m:%Y-%m}: {len(hist)} meses (se requieren >= {min_history_months})")
    last = hist.index.max()
    steps = (end_m.year - last.year) * 12 + (end_m.month - last.month)

    if refit == "auto":
        do_refit = _month(art.train_end) >= start_m
    else:
        do_refit = bool(refit)

    preds = []
    for comp in art.components:
        spec = ModelSpec.from_dict(comp["spec"])
        # Con re-estimación se cachean solo los parámetros; el estado se re-ancla siempre con apply_params
        # (mismo resultado en acierto o fallo de caché).
        params = refit_params_cached(hist, spec) if do_refit else comp["params"]
        res = apply_params(hist, spec, params)
        preds.append(forecast_spec(res, spec, last, steps))
    fc = combine(preds, art.weights) if len(preds) > 1 else preds[0]
    fc = fc.loc[start_m:end_m]
    if not np.isfinite(fc.to_numpy()).all():
        raise ValueError("pronóstico no finito")
    out = pd.DataFrame({"fecha": fc.index, "prediccion": fc.to_numpy()})
    out.attrs = {
        "ultimo_real": f"{last:%Y-%m}",
        "refit": do_refit,
        "modelo": art.metadata.get("nombre", "+".join(s.name() for s in art.specs())),
    }
    return out
