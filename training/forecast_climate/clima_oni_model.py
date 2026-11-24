"""Modelo anual id=4 (clima + ONI) v2: crecimiento parsimonioso con sensibilidad ONI, sin mismatch train/serve.

Forma del modelo (en log del total anual, kWh)::

    log1p(D_t) = log1p(D_{t-1}) + mu + beta_oni * ONI_t + sum_k gamma_k * x_k,t

* ``lag_1`` entra con coeficiente **fijo = 1** (paseo aleatorio con deriva): con ~13 años no se puede
  estimar un coeficiente de nivel sin sobreajustar (el Ridge original repartía peso entre lag_1, lag_2,
  year_index y year_index_norm).
* ``mu`` (deriva) es el crecimiento medio ajustado por ONI en la ventana de entrenamiento.
* ``beta_oni`` es la sensibilidad de la demanda al ONI medio del año (identidad del modelo: El Niño → más
  calor/sequía → más demanda; La Niña → menos). Opcionalmente restringida a ``>= 0`` (signo físico).
* ``x_k`` son features adicionales opcionales (clima en forma de climatología, ``growth_rate_lag1``…).

**Cero mismatch train/serve**: las features de entrenamiento NO se recalculan aparte; se capturan llamando
al código real de inferencia (``DemandPredictor.predict_future_climate``) con un escalador que solo graba
la fila que recibe (:func:`capture_inference_features`). Es decir, el año ``t`` se entrena con exactamente
lo que la API le pasaría al modelo con corte ``t-1``: climatología de 10 años hasta ``t-1``, ONI proyectado
desde el último mes observado (o el ONI observado, si ``oni_source="observado"``: variante evaluada por
walk-forward y documentada) y lags/crecimientos de los totales hasta ``t-1``.

El resultado se exporta como objetos sklearn estándar (``StandardScaler`` + ``LinearRegression`` +
``StandardScaler``) con el contrato que ``forecast.py`` ya espera (``feature_names_in_``, ``predict`` sobre X
escalado → y escalado → ``inverse_transform`` → ``log1p`` total). Los coeficientes en espacio escalado se
derivan algebraicamente del modelo en espacio original (ver :func:`to_sklearn_artifacts`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

#: Features anuales que ``predict_future_climate`` construye (en este orden) antes de seleccionar las del
#: escalador. Cualquier subconjunto es compatible con la inferencia actual sin tocar ``forecast.py``.
INFERENCE_ANNUAL_FEATURES = [
    "temp_mean_anio",
    "temp_max_anio",
    "temp_min_anio",
    "humidity_mean_anio",
    "wind_speed_mean_anio",
    "rain_sum_anio",
    "oni_mean_anio",
    "lag_1",
    "lag_2",
    "year_index",
    "year_index_norm",
    "growth_rate_lag1",
]


# --------------------------------------------------------------------------------------------------
# Captura de features EXACTAS de inferencia
# --------------------------------------------------------------------------------------------------
class _CaptureScaler:
    """Escalador falso: graba la fila de features que la inferencia arma y la devuelve sin cambios."""

    def __init__(self) -> None:
        self.rows: list[pd.DataFrame] = []
        self.feature_names_in_ = np.array(INFERENCE_ANNUAL_FEATURES, dtype=object)

    def transform(self, X):  # noqa: N803 (firma sklearn)
        self.rows.append(pd.DataFrame(X).copy())
        return np.zeros((len(X), 1))


class _ZeroModel:
    def predict(self, X):  # noqa: N803
        return np.zeros(len(X))


class _LagScaler:
    """``inverse_transform`` que devuelve un log-total plausible (lag_1 capturado) para que el año 2 tenga
    features bien definidas; solo se usan las features del año 1 para entrenar."""

    def __init__(self, capture: _CaptureScaler) -> None:
        self.capture = capture

    def inverse_transform(self, y):
        last = self.capture.rows[-1]
        return np.full((len(y), 1), float(last["lag_1"].iloc[-1]))


def make_predictor(hist_monthly: pd.DataFrame, hist_annual: pd.DataFrame, scaler_x=None, model=None, scaler_y=None):
    """Construye un ``DemandPredictor`` de id=4 en memoria (sin leer pkl/CSV de disco) para correr el código
    real de inferencia con datos y modelos arbitrarios (walk-forward, captura de features, tests)."""
    from api.services.forecast import DemandPredictor

    p = DemandPredictor.__new__(DemandPredictor)
    p.model_path = "api/models/climate_model/"
    p.climate_hist_monthly = hist_monthly.copy()
    p.climate_hist_annual = hist_annual.copy()
    p.climate_corrector = None
    if scaler_x is not None:
        p.climate_annual_scaler_x = scaler_x
        p.climate_annual_features = list(scaler_x.feature_names_in_)
        p.climate_annual_model = model
        p.climate_annual_scaler_y = scaler_y
    return p


def capture_inference_rows(
    hist_monthly: pd.DataFrame,
    hist_annual: pd.DataFrame,
    cutoff_year: int,
    forecast_years: int = 1,
    enso_phase_by_month: Optional[dict] = None,
) -> pd.DataFrame:
    """Filas de features que la inferencia arma para ``cutoff_year+1 … cutoff_year+forecast_years``.

    Las columnas exógenas (clima, ONI) son exactas para todos los años; las derivadas de la recursión
    (``lag_1``, ``lag_2``, ``growth_rate_lag1``) solo son exactas en el primer año (en los siguientes dependen
    de la predicción del modelo, que aquí es ficticia)."""
    import contextlib
    import io

    cap = _CaptureScaler()
    p = make_predictor(hist_monthly, hist_annual, cap, _ZeroModel(), _LagScaler(cap))
    with contextlib.redirect_stdout(io.StringIO()):
        p.predict_future_climate(
            forecast_years=forecast_years, cutoff_year=cutoff_year, enso_phase_by_month=enso_phase_by_month
        )
    if len(cap.rows) != forecast_years:
        raise RuntimeError("la inferencia no armó una fila por año (¿falló predict_future_climate?)")
    out = pd.concat(cap.rows, ignore_index=True)
    out.index = range(cutoff_year + 1, cutoff_year + 1 + forecast_years)
    return out


def capture_inference_features(
    hist_monthly: pd.DataFrame,
    hist_annual: pd.DataFrame,
    cutoff_year: int,
    enso_phase_by_month: Optional[dict] = None,
) -> pd.Series:
    """Features anuales EXACTAS que la inferencia usaría para el año ``cutoff_year + 1``."""
    rows = capture_inference_rows(hist_monthly, hist_annual, cutoff_year, 1, enso_phase_by_month)
    return rows.iloc[0].rename(cutoff_year + 1)


def build_training_table(
    hist_monthly: pd.DataFrame,
    hist_annual: pd.DataFrame,
    years: Sequence[int],
    oni_source: str = "proyectado",
) -> pd.DataFrame:
    """Una fila por año objetivo ``t`` (features de inferencia con corte ``t-1``) + ``target_log`` y
    ``growth`` (= target_log − lag_1). ``oni_source='observado'`` reemplaza ``oni_mean_anio`` por el ONI
    medio observado del año (variante de entrenamiento "condicional al escenario")."""
    tot = hist_annual.set_index("year")["demand_total"]
    rows = []
    for t in years:
        if t not in tot.index or (t - 1) not in tot.index:
            continue
        f = capture_inference_features(hist_monthly, hist_annual, t - 1)
        if oni_source == "observado":
            f["oni_mean_anio"] = float(hist_annual.set_index("year").loc[t, "oni_mean_anio"])
        elif oni_source != "proyectado":
            raise ValueError(f"oni_source inválido: {oni_source}")
        f["oni_mean_anio_lag1"] = float(hist_annual.set_index("year").loc[t - 1, "oni_mean_anio"])
        f["oni_delta"] = f["oni_mean_anio"] - f["oni_mean_anio_lag1"]
        f["target_log"] = float(np.log1p(tot.loc[t]))
        f["growth"] = f["target_log"] - f["lag_1"]
        rows.append(f)
    out = pd.DataFrame(rows)
    out.index.name = "year"
    return out


# --------------------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------------------
@dataclass
class GrowthModelSpec:
    """Hiperparámetros del modelo (todos elegidos por walk-forward con orígenes <= 2023-12)."""

    extra_features: tuple[str, ...] = ("oni_mean_anio",)
    oni_source: str = "proyectado"  # "proyectado" | "observado"
    window_years: Optional[int] = None  # None = toda la historia; N = últimos N años utilizables
    exclude_years: tuple[int, ...] = ()  # años objetivo excluidos del ajuste (p. ej. COVID)
    alpha: float = 0.0  # Ridge sobre features estandarizadas (0 = OLS)
    nonneg_oni: bool = True  # beta_oni >= 0 (signo físico)
    beta_all_history: bool = False  # True: coeficientes con toda la historia; mu con la ventana
    n_years_shape: int = 2  # reparto mensual (lo usa la inferencia; se documenta/evalúa aparte)

    def label(self) -> str:
        f = "+".join(self.extra_features) or "solo_deriva"
        return (
            f"{f}|oni={self.oni_source}|win={self.window_years or 'all'}|excl={','.join(map(str, self.exclude_years)) or '-'}"
            f"|a={self.alpha}|nn={int(self.nonneg_oni)}|bAll={int(self.beta_all_history)}"
        )


@dataclass
class GrowthModel:
    spec: GrowthModelSpec
    mu: float
    coefs: dict = field(default_factory=dict)  # feature -> coef (espacio original, sobre growth)
    train_years: list = field(default_factory=list)
    train_X: Optional[pd.DataFrame] = None  # features usadas (incl. lag_1) para los escaladores

    def predict_growth(self, X: pd.DataFrame) -> np.ndarray:
        g = np.full(len(X), self.mu)
        for k, c in self.coefs.items():
            g = g + c * X[k].to_numpy(float)
        return g

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_1"].to_numpy(float) + self.predict_growth(X)

    def to_dict(self) -> dict:
        return {"spec": asdict(self.spec), "mu": self.mu, "coefs": self.coefs, "train_years": self.train_years}


def fit_growth_model(table: pd.DataFrame, spec: GrowthModelSpec) -> GrowthModel:
    """Ajusta ``growth ~ mu + sum coef_k x_k`` sobre la tabla de :func:`build_training_table`.

    * Los coeficientes ``coef_k`` se estiman por OLS/Ridge sobre los años utilizables (sin ``exclude_years``):
      todos (``beta_all_history=True``) o solo la ventana ``window_years``.
    * La deriva ``mu`` es la media de ``growth − sum coef_k x_k`` en la ventana (adapta el nivel de
      crecimiento reciente sin re-estimar la sensibilidad climática con 5 puntos).
    * ``nonneg_oni``: si la sensibilidad ONI sale negativa (contra-física) se fija en 0 y se re-ajusta el resto.
    """
    t_all = table[~table.index.isin(spec.exclude_years)].sort_index()
    t = t_all.iloc[-spec.window_years:] if spec.window_years else t_all
    if len(t) < 3:
        raise ValueError(f"muy pocos años para ajustar ({len(t)})")
    src = t_all if spec.beta_all_history else t
    feats = list(spec.extra_features)
    coefs = _fit_coefs(src, feats, spec.alpha)
    oni_feat = next((f for f in feats if f in ("oni_mean_anio", ONI_DELTA)), None)
    if spec.nonneg_oni and oni_feat is not None and coefs[oni_feat] < 0:
        coefs = {**_fit_coefs(src, [f for f in feats if f != oni_feat], spec.alpha), oni_feat: 0.0}
        coefs = {f: coefs[f] for f in feats}
    resid = t["growth"].to_numpy(float) - sum(c * t[k].to_numpy(float) for k, c in coefs.items())
    mu = float(np.mean(resid))
    return GrowthModel(spec, mu, {k: float(v) for k, v in coefs.items()}, t.index.tolist(), t[_scaler_cols(feats)].copy())


def _fit_coefs(t: pd.DataFrame, feats: list, alpha: float) -> dict:
    if not feats:
        return {}
    X = t[feats].to_numpy(float)
    y = t["growth"].to_numpy(float)
    mx, sx = X.mean(axis=0), X.std(axis=0)
    sx[sx == 0] = 1.0
    reg = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
    reg.fit((X - mx) / sx, y)
    return dict(zip(feats, (reg.coef_ / sx).tolist()))


def _scaler_cols(feats) -> list:
    return ["lag_1", *expand_coefs({f: 1.0 for f in feats}).keys()]


#: Feature derivada que la inferencia actual NO arma: requiere el parche documentado en
#: ``docs/work/04_clima/forecast_clima_patch.md`` (añade ``oni_mean_anio_lag1`` a la fila anual).
ONI_DELTA = "oni_delta"
PATCH_FEATURES = ("oni_mean_anio_lag1",)


def expand_coefs(coefs: dict) -> dict:
    """Expresa ``oni_delta`` como ``+c·oni_mean_anio − c·oni_mean_anio_lag1`` (features de la fila anual)."""
    out: dict = {}
    for k, c in coefs.items():
        if k == ONI_DELTA:
            out["oni_mean_anio"] = out.get("oni_mean_anio", 0.0) + c
            out["oni_mean_anio_lag1"] = out.get("oni_mean_anio_lag1", 0.0) - c
        else:
            out[k] = out.get(k, 0.0) + c
    return out


def requires_patch(model: "GrowthModel") -> bool:
    return any(k in PATCH_FEATURES for k in expand_coefs(model.coefs))


def to_sklearn_artifacts(model: GrowthModel, scaler_fit_X: pd.DataFrame, target_log: np.ndarray):
    """Convierte el modelo a (scaler_x, LinearRegression, scaler_y) con el contrato de ``forecast.py``.

    log = lag_1 + mu + sum c_k x_k  →  con z_j = (x_j − m_j)/s_j  e  y_s = (log − m_y)/s_y:
    y_s = (mu + sum_j w_j m_j − m_y)/s_y + sum_j (w_j s_j / s_y) z_j, con w = {lag_1: 1, k: c_k}.
    El ``LinearRegression`` se ajusta por mínimos cuadrados sobre puntos generados por la propia fórmula
    (ajuste exacto, residuo ~0) para obtener un estimador sklearn genuino, sin escribir ``coef_`` a mano.
    """
    coefs = expand_coefs(model.coefs)
    feats = ["lag_1", *coefs.keys()]
    Xf = scaler_fit_X[feats].astype(float)
    sx = StandardScaler().fit(Xf)
    sy = StandardScaler().fit(np.asarray(target_log, float).reshape(-1, 1))
    # Puntos sintéticos en la caja de los datos (bien condicionados) para fijar la recta exacta.
    rng = np.random.default_rng(0)
    n = max(50, 10 * len(feats))
    Z = rng.normal(size=(n, len(feats)))
    Xs = pd.DataFrame(sx.inverse_transform(Z), columns=feats)
    log = Xs["lag_1"].to_numpy(float) + model.mu + sum(c * Xs[k].to_numpy(float) for k, c in coefs.items())
    ys = sy.transform(log.reshape(-1, 1)).ravel()
    lr = LinearRegression().fit(Z, ys)
    resid = np.abs(lr.predict(Z) - ys).max()
    if resid > 1e-8:
        raise RuntimeError(f"la conversión a sklearn no es exacta (residuo {resid})")
    return sx, lr, sy
