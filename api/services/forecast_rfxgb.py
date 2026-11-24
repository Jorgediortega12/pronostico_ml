"""Pronóstico mensual de demanda v2 para los modelos id=1 (Random Forest) e id=2 (XGBoost).

Módulo autocontenido (no depende de ``forecast.py``). Lo usan tanto la inferencia de la API como el
entrenamiento (``training/forecast_rfxgb``), así que las features son idénticas en train y en serve.

Enfoque: **pronóstico directo multi-horizonte** sobre ``log(demanda)`` con un *offset* estructural::

    log y[o+h] = offset(o, h) + f_ML(features(o, h))        h = 1 .. H

* ``o`` (origen) es el último mes completo del histórico; ``h`` el horizonte en meses.
* ``offset`` es el naive estacional con drift (mismo mes del último año observado × (1+drift)^k, drift =
  media del crecimiento interanual de los últimos 36 meses) o el nivel de 12 meses móviles × índice
  estacional medio (ver :data:`OFFSETS`). El modelo ML (RF o XGB) aprende **solo la corrección** en log.
* Features (todas con información ``<= o``, salvo el calendario del mes objetivo, que es determinista):
  horizonte, mes, días/laborables/sábados/domingos+festivos de Colombia del mes objetivo y su diferencia
  contra el mes base, crecimientos al origen y desviación estacional del mes base.
* Sin clima ni ONI: en inferencia solo habría climatología (≈ estacionalidad, ya capturada) y el ONI anual
  disponible es una clasificación ex-post del año (fuga). Ver ``docs/work/02_rfxgb/REPORTE.md``.

Unidad: kWh. Escala: la del histórico recibido (mercado completo si se corta en el quiebre de mercado).
"""
from __future__ import annotations

import logging
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from api.services.calendar_co import calendar_counts
from api.services.forecast_cache import RETRAIN_SEMAPHORE, SingleFlightLRU, series_digest

logger = logging.getLogger(__name__)

#: Artefactos v2 (los .pkl de la carpeta padre los sigue usando ``forecast.py`` legado).
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
MODEL_DIRS = {1: MODELS_ROOT / "rf_model" / "v2", 2: MODELS_ROOT / "xgb_model" / "v2"}
MODEL_NAMES = {1: "random_forest", 2: "xgboost"}
ARTIFACT_FILE = "model.joblib"
METADATA_FILE = "metadata.json"

#: Meses mínimos de historia para construir features e INFERIR con un artefacto ya entrenado (36 para el
#: crecimiento a 3 años + 1 año de base). Re-entrenar exige más: ver :func:`min_training_months`.
MIN_HISTORY_MONTHS = 48
#: Filas ``(origen, h)`` mínimas para entrenar (:func:`fit_model`).
MIN_TRAIN_ROWS = 50
#: Observaciones mínimas dentro de una ventana de 12 meses para calcular el nivel móvil.
MIN_TTM_OBS = 9
#: Horizonte máximo aceptado en inferencia (meses desde el último dato). Más allá se rechaza.
MAX_INFERENCE_HORIZON = 120
#: Años de historia del drift (misma definición que el baseline naive+drift del harness).
DRIFT_MONTHS = 36

OFFSETS = ("naive_drift", "nivel_estacional")

FEATURES = [
    "h",
    "k",
    "mes",
    "m_sin",
    "m_cos",
    "dias_t",
    "lab_t",
    "sab_t",
    "domfest_t",
    "d_dias",
    "d_lab",
    "d_sab",
    "d_domfest",
    "base_dev",
    "si_mean",
    "g12",
    "g3",
    "g36",
    "drift",
    "g_since_b",
    "off_gap",
]


# --------------------------------------------------------------------------------------------------
# Preparación de la serie
# --------------------------------------------------------------------------------------------------
def prepare_history(history: Union[pd.Series, pd.DataFrame], value_col: Optional[str] = None) -> pd.Series:
    """Normaliza el histórico a una Serie mensual (índice inicio de mes, float, ordenada, sin NaN ni <= 0).

    Acepta una Serie con índice de fechas, o un DataFrame con columna de fecha (``fecha``/``mes``/``date``)
    y una de valor (``value_col`` o la primera numérica). Meses repetidos -> error (ambigüedad).
    """
    if isinstance(history, pd.DataFrame):
        df = history.copy()
        date_col = next((c for c in ("fecha", "mes", "date", "month") if c in df.columns), None)
        if date_col is not None:
            df = df.set_index(date_col)
        if value_col is None:
            num = df.select_dtypes("number").columns
            if not len(num):
                raise ValueError("el DataFrame de histórico no tiene columnas numéricas")
            value_col = num[0]
        s = df[value_col]
    else:
        s = history
    s = pd.Series(np.asarray(s, dtype=float), index=pd.DatetimeIndex(pd.to_datetime(s.index)))
    s.index = s.index.to_period("M").to_timestamp()
    if s.index.has_duplicates:
        raise ValueError("el histórico tiene meses duplicados")
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() > 0)]
    return s.sort_index().rename("demanda_kwh")


# --------------------------------------------------------------------------------------------------
# Diseño (features + offset)
# --------------------------------------------------------------------------------------------------
def build_design(
    history: pd.Series,
    horizon: int,
    origins: Optional[list] = None,
    min_origin_months: int = MIN_HISTORY_MONTHS,
) -> pd.DataFrame:
    """Construye filas ``(origen, h)`` con features, offsets y target (si existe).

    Cada fila usa **solo** valores de ``history`` con fecha ``<= origen`` (más el calendario del mes
    objetivo). Con ``origins=None`` usa todos los meses observados con al menos ``min_origin_months`` de
    historia (para entrenar); en inferencia se pasa ``origins=[último mes]``.

    Columnas: ``origen, objetivo, h`` + :data:`FEATURES` + ``off_naive_drift, off_nivel_estacional, y``
    (``y`` = log demanda real del objetivo si está en ``history``, NaN si no).
    """
    s = prepare_history(history)
    if s.empty:
        raise ValueError("histórico vacío")
    first, last = s.index.min(), s.index.max()
    n_hist = (last.year - first.year) * 12 + last.month - first.month + 1
    n_tot = n_hist + horizon
    months = pd.date_range(first, periods=n_tot, freq="MS")
    y = s.reindex(months).to_numpy(dtype=float)
    ly = np.log(y)

    ys = pd.Series(y)
    lvl = ys.rolling(12, min_periods=MIN_TTM_OBS).mean().to_numpy()  # nivel medio mensual (trailing 12m)
    llvl = np.log(lvl)
    dev = ly - llvl  # desviación estacional del mes respecto a su nivel trailing
    yoy = ys / ys.shift(12) - 1
    drift = yoy.rolling(DRIFT_MONTHS, min_periods=1).mean().to_numpy()
    s3 = ys.rolling(3, min_periods=3).sum().to_numpy()

    cal = np.array([calendar_counts(m.year, m.month) for m in months], dtype=float)
    mon = months.month.to_numpy()

    if origins is None:
        o_pos = [i for i in range(n_hist) if i >= min_origin_months - 1 and np.isfinite(y[i])]
    else:
        o_pos = []
        for o in origins:
            o = pd.Timestamp(o).to_period("M").to_timestamp()
            i = (o.year - first.year) * 12 + o.month - first.month
            if not 0 <= i < n_hist:
                raise ValueError(f"origen {o:%Y-%m} fuera del histórico")
            o_pos.append(i)

    rows = []
    for o in o_pos:
        if o < 36 or not (np.isfinite(llvl[o]) and np.isfinite(llvl[o - 12]) and np.isfinite(llvl[o - 36])):
            if origins is None:
                continue
            raise ValueError("historia insuficiente al origen (se requieren >= 37 meses con nivel móvil)")
        d_o = drift[o] if np.isfinite(drift[o]) else 0.0
        ld = math.log1p(d_o)
        g12 = llvl[o] - llvl[o - 12]
        g36 = (llvl[o] - llvl[o - 36]) / 3.0
        g3 = math.log(s3[o] / s3[o - 12]) if np.isfinite(s3[o]) and np.isfinite(s3[o - 12]) else g12
        for h in range(1, horizon + 1):
            t = o + h
            if t >= n_tot:
                break
            p0 = t - 12 * math.ceil(h / 12)  # mismo mes calendario más reciente <= origen
            p = p0
            while p >= 0 and not np.isfinite(y[p]) and p > p0 - 36:
                p -= 12
            same = [q for q in (p0, p0 - 12, p0 - 24) if q >= 0 and np.isfinite(dev[q])]
            si_mean = float(np.mean([dev[q] for q in same])) if same else 0.0
            if p >= 0 and np.isfinite(y[p]):
                lyb = ly[p]
                base_dev = dev[p] if np.isfinite(dev[p]) else si_mean
                g_since_b = llvl[o] - llvl[p] if np.isfinite(llvl[p]) else g12 * (o - p) / 12
            else:  # sin mes base observado: nivel actual × índice estacional
                p = p0
                lyb = llvl[o] + si_mean
                base_dev = si_mean
                g_since_b = 0.0
            k = (t - p) / 12.0
            off_a = lyb + k * ld
            off_b = llvl[o] + si_mean + ((t - (o - 5.5)) / 12.0) * ld
            ct, cb = cal[t], cal[p]
            m = mon[t]
            rows.append(
                (
                    months[o], months[t], h, k, m, math.sin(2 * math.pi * m / 12), math.cos(2 * math.pi * m / 12),
                    ct[0], ct[1], ct[2], ct[3], ct[0] - cb[0], ct[1] - cb[1], ct[2] - cb[2], ct[3] - cb[3],
                    base_dev, si_mean, g12, g3, g36, d_o, g_since_b, off_b - off_a, off_a, off_b,
                    ly[t] if t < n_hist else np.nan,
                )
            )
    cols = ["origen", "objetivo"] + FEATURES + ["off_naive_drift", "off_nivel_estacional", "y"]
    return pd.DataFrame(rows, columns=cols)


def residual_target(design: pd.DataFrame, offset: str) -> pd.Series:
    """Target del ML: ``log y - offset``."""
    return design["y"] - design[f"off_{offset}"]


# --------------------------------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------------------------------
def make_estimator(model_id: int, params: Optional[dict] = None, seed: int = 42):
    """RandomForest (id 1) o XGBoost (id 2) con los hiperparámetros dados (por defecto los elegidos)."""
    params = dict(params or {})
    if model_id == 1:
        from sklearn.ensemble import RandomForestRegressor

        base = dict(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=seed)
        base.update(params)
        return RandomForestRegressor(**base)
    if model_id == 2:
        from xgboost import XGBRegressor

        base = dict(
            n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.0, objective="reg:squarederror", n_jobs=4, random_state=seed,
        )
        base.update(params)
        return XGBRegressor(**base)
    raise ValueError(f"model_id no soportado: {model_id} (solo 1=RF, 2=XGB)")


def training_rows(design: pd.DataFrame, exclude_targets: tuple = ()) -> pd.DataFrame:
    """Filas con target observado y features finitas; ``exclude_targets`` = [(ini, fin)] de meses objetivo
    a omitir del entrenamiento (p. ej. COVID)."""
    d = design[design["y"].notna()]
    d = d[np.isfinite(d[FEATURES].to_numpy(dtype=float)).all(axis=1)]
    for ini, fin in exclude_targets:
        d = d[~d["objetivo"].between(pd.Timestamp(ini), pd.Timestamp(fin))]
    return d


def min_training_months(h_train: int = 36, min_rows: int = MIN_TRAIN_ROWS) -> int:
    """Meses contiguos mínimos de historia para que :func:`fit_model` tenga ``min_rows`` filas.

    Con ``n`` meses, los orígenes válidos son las posiciones ``MIN_HISTORY_MONTHS-1 .. n-2`` y el origen ``o``
    aporta ``min(h_train, n-1-o)`` filas con target observado. Con ``h_train=36`` y 50 filas el mínimo es 58
    meses (48 -> 0 filas, 57 -> 45). Huecos o ``exclude_targets`` pueden exigir más (se valida al entrenar).
    """
    n = MIN_HISTORY_MONTHS
    while sum(min(h_train, n - 1 - o) for o in range(MIN_HISTORY_MONTHS - 1, n - 1)) < min_rows:
        n += 1
    return n


def fit_model(history: pd.Series, model_id: int, cfg: dict) -> dict:
    """Entrena con todo ``history`` y devuelve el artefacto (dict serializable con joblib).

    ``cfg``: ``offset`` (uno de :data:`OFFSETS`), ``params`` (hiperparámetros), ``h_train`` (horizonte
    máximo de entrenamiento), ``exclude_targets`` (periodos de target omitidos).
    """
    h_train = int(cfg.get("h_train", 36))
    n_hist = len(prepare_history(history))
    required = min_training_months(h_train)
    if n_hist < required:
        raise ValueError(
            f"historia insuficiente: se requieren >= {required} meses para entrenar (hay {n_hist})"
        )
    design = build_design(history, h_train)
    tr = training_rows(design, tuple(cfg.get("exclude_targets", ())))
    if len(tr) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"historia insuficiente: {len(tr)} filas de entrenamiento (< {MIN_TRAIN_ROWS}) con {n_hist} meses "
            "(huecos o periodos excluidos); amplíe la historia"
        )
    est = make_estimator(model_id, cfg.get("params"))
    est.fit(tr[FEATURES].to_numpy(dtype=float), residual_target(tr, cfg["offset"]).to_numpy(dtype=float))
    s = prepare_history(history)
    return {
        "version": 2,
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "estimator": est,
        "features": list(FEATURES),
        "offset": cfg["offset"],
        "h_train": h_train,
        "params": dict(cfg.get("params") or {}),
        "exclude_targets": [[str(a), str(b)] for a, b in cfg.get("exclude_targets", ())],
        "train_first_month": f"{s.index.min():%Y-%m}",
        "train_last_month": f"{s.index.max():%Y-%m}",
        "n_train_rows": int(len(tr)),
    }


def predict_from_artifact(artifact: dict, history: pd.Series, horizon: int) -> pd.Series:
    """Pronostica ``horizon`` meses tras el último mes de ``history`` con un artefacto en memoria.

    Horizontes > ``h_train`` se calculan con las features de ``h`` y ``k`` acotadas al rango visto en
    entrenamiento (el drift del offset sí se compone con los años reales).
    """
    s = prepare_history(history)
    design = build_design(s, horizon, origins=[s.index.max()])
    X = design[artifact["features"]].copy()
    h_train = int(artifact["h_train"])
    X["h"] = X["h"].clip(upper=h_train)
    X["k"] = X["k"].clip(upper=math.ceil(h_train / 12))
    r = artifact["estimator"].predict(X.to_numpy(dtype=float))
    pred = np.exp(design[f"off_{artifact['offset']}"].to_numpy() + r)
    return pd.Series(pred, index=pd.DatetimeIndex(design["objetivo"]), name="demanda_kwh")


# --------------------------------------------------------------------------------------------------
# Inferencia pública
# --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_artifact_cached(path: str, mtime: float) -> dict:
    import joblib

    return joblib.load(path)


def load_artifact(model_id: int, models_dir: Optional[Union[str, Path]] = None) -> dict:
    """Carga el artefacto v2 de ``model_id`` (1 RF, 2 XGB). Cacheado por ruta+mtime."""
    if model_id not in MODEL_DIRS:
        raise ValueError(f"model_id no soportado: {model_id} (solo 1=RF, 2=XGB)")
    d = Path(models_dir) if models_dir is not None else MODEL_DIRS[model_id]
    path = d / ARTIFACT_FILE
    if not path.exists():
        raise FileNotFoundError(f"no existe el artefacto {path}; ejecute training/forecast_rfxgb/train.py")
    art = _load_artifact_cached(str(path), path.stat().st_mtime)
    if art.get("model_id") != model_id:
        raise ValueError(f"el artefacto {path} es del modelo {art.get('model_id')}, no del {model_id}")
    return art


#: Núcleos por re-entreno en línea (hindcast desde la API): acotado para no saturar el worker.
ONLINE_REFIT_N_JOBS = 2
#: Re-entrenos en línea cacheados (clave: modelo + hash de la historia + huella del artefacto). Cada entrada
#: pesa ~4 MB (RF, 300 árboles) o ~0.3 MB (XGB): 32 entradas <= ~130 MB en el peor caso (todo RF) y cubren
#: los orígenes distintos de una sesión de demo (cada start de hindcast es una historia distinta).
ONLINE_REFIT_CACHE_MAX = 32
_ONLINE_REFIT_CACHE = SingleFlightLRU(ONLINE_REFIT_CACHE_MAX)


def artifact_fingerprint(model_id: int, models_dir: Optional[Union[str, Path]] = None) -> str:
    """Huella del artefacto en disco (mtime_ns + tamaño): cambia si se reentrena/sustituye el ``.joblib``."""
    d = Path(models_dir) if models_dir is not None else MODEL_DIRS[model_id]
    st = (d / ARTIFACT_FILE).stat()
    return f"{st.st_mtime_ns}-{st.st_size}"


def refit_from_artifact_cached(model_id: int, history: pd.Series, artifact: dict,
                               fingerprint: Optional[str] = None) -> dict:
    """Re-entrena el modelo con ``history`` y la config del ``artifact`` (offset, hiperparámetros, h_train,
    exclusiones). Cacheado (LRU + single-flight) por modelo, hash corto de la historia y huella del artefacto;
    el re-entreno usa :data:`ONLINE_REFIT_N_JOBS` núcleos y respeta el semáforo global de re-entrenos."""
    s = prepare_history(history)
    required = min_training_months(int(artifact.get("h_train", 36)))
    if len(s) < required:
        raise ValueError(f"historia insuficiente: se requieren >= {required} meses para re-entrenar (hay {len(s)})")
    fp = fingerprint if fingerprint is not None else (
        f"{artifact.get('train_last_month')}-{artifact.get('n_train_rows')}")
    key = (int(model_id), series_digest(s), fp)

    def _fit():
        params = dict(artifact.get("params") or {})
        params["n_jobs"] = ONLINE_REFIT_N_JOBS
        cfg = {
            "offset": artifact["offset"],
            "params": params,
            "h_train": int(artifact.get("h_train", 36)),
            "exclude_targets": [tuple(x) for x in artifact.get("exclude_targets", [])],
        }
        with RETRAIN_SEMAPHORE:
            return fit_model(s, model_id, cfg)

    return _ONLINE_REFIT_CACHE.get_or_compute(key, _fit)


def _month(x) -> pd.Timestamp:
    return pd.Timestamp(x).to_period("M").to_timestamp()


def predict_monthly(
    model_id: int,
    history_monthly_series: Union[pd.Series, pd.DataFrame],
    start: Union[str, date, pd.Timestamp],
    end: Union[str, date, pd.Timestamp],
    *,
    market_break_month: Optional[Union[str, date, pd.Timestamp]] = None,
    post_break_factor: float = 1.0,
    models_dir: Optional[Union[str, Path]] = None,
    artifact: Optional[dict] = None,
) -> pd.DataFrame:
    """Pronóstico mensual (kWh) de ``start`` a ``end`` (inclusive, por mes) con el modelo v2.

    Args:
        model_id: 1 = Random Forest, 2 = XGBoost.
        history_monthly_series: demanda mensual **de meses completos** (Serie indexada por fecha o
            DataFrame ``fecha``/valor). Se ignoran meses ``>= start`` (el pronóstico usa solo lo conocido
            antes del inicio pedido) y meses ``>= market_break_month``.
        start, end: rango pedido. Cualquier fecha (se normaliza al mes). Las fechas de salida van de
            ``start`` a ``end`` aunque el último dato sea anterior (los meses intermedios se pronostican
            pero no se devuelven).
        market_break_month: primer mes post-quiebre de mercado (p. ej. ``MarketBreak.first_post_month``).
            Sus datos (escala distinta) no se usan como historia.
        post_break_factor: fracción del mercado (0-1] a aplicar a los meses pronosticados ``>=
            market_break_month``. Por defecto 1.0 = todo en escala de mercado completo.
        models_dir / artifact: para tests o artefactos alternativos.

    Returns:
        DataFrame con ``fecha`` (inicio de mes), ``demanda_kwh``, ``demanda_mercado_completo_kwh``,
        ``h`` (meses desde el último dato), ``factor_mercado`` y atributos ``attrs`` con metadatos.

    Raises:
        ValueError: rango inválido, historia insuficiente (< 48 meses para inferir; re-entreno en
            hindcast: :func:`min_training_months`) u horizonte > 120 meses.
    """
    start_m, end_m = _month(start), _month(end)
    if end_m < start_m:
        raise ValueError("end debe ser >= start")
    if not 0 < post_break_factor <= 1.0 + 1e-9:
        raise ValueError("post_break_factor debe estar en (0, 1]")
    s = prepare_history(history_monthly_series)
    s = s[s.index < start_m]
    brk = _month(market_break_month) if market_break_month is not None else None
    if brk is not None:
        s = s[s.index < brk]
    if len(s) < MIN_HISTORY_MONTHS:
        raise ValueError(f"historia insuficiente: {len(s)} meses (< {MIN_HISTORY_MONTHS})")
    last = s.index.max()
    horizon = (end_m.year - last.year) * 12 + end_m.month - last.month
    if horizon > MAX_INFERENCE_HORIZON:
        raise ValueError(f"horizonte {horizon} meses > {MAX_INFERENCE_HORIZON} desde el último dato ({last:%Y-%m})")
    art = artifact if artifact is not None else load_artifact(model_id, models_dir)
    pred = predict_from_artifact(art, s, horizon)
    out = pd.DataFrame({"fecha": pred.index, "demanda_mercado_completo_kwh": pred.to_numpy()})
    out["h"] = np.arange(1, len(out) + 1)
    out["factor_mercado"] = 1.0
    if brk is not None and post_break_factor != 1.0:
        out.loc[out["fecha"] >= brk, "factor_mercado"] = float(post_break_factor)
    out["demanda_kwh"] = out["demanda_mercado_completo_kwh"] * out["factor_mercado"]
    out = out[(out["fecha"] >= start_m) & (out["fecha"] <= end_m)].reset_index(drop=True)
    out = out[["fecha", "demanda_kwh", "demanda_mercado_completo_kwh", "h", "factor_mercado"]]
    out.attrs = {
        "model_id": model_id,
        "model_name": art.get("model_name"),
        "ultimo_mes_historia": f"{last:%Y-%m}",
        "train_last_month": art.get("train_last_month"),
        "escala": "mercado_completo" if post_break_factor == 1.0 or brk is None else "bd_post_quiebre",
    }
    return out
