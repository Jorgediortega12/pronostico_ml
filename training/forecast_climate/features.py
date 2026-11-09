"""
Feature engineering utilities for the climate + ONI demand forecast model (model id=4).

All functions here are PURE: they take a DataFrame and return a new DataFrame with
extra columns appended. No I/O, no DB access, no global state. This lets
`train_candidates.py` (owned by another agent) compose these functions freely on
top of the datasets produced by `build_dataset.py`.

Convention: the input DataFrame is expected to be sorted chronologically
(ascending by `year`, then `month` when present) and to have one row per period
(month or quarter). Functions that need chronological order re-sort defensively
and restore the original row order before returning, so callers don't have to
worry about ordering side effects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Robust, cwd-independent path to the calibrated ONI decay-rate artifact
# (produced by `training/forecast_climate/calibrate_oni_decay.py`), mirroring
# how model artifact paths are resolved elsewhere in the repo (e.g.
# `_BACKEND_ROOT = Path(__file__).resolve().parents[2]` in
# `api/services/forecast.py`) instead of depending on the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ONI_DECAY_ARTIFACT_PATH = _REPO_ROOT / "api" / "models" / "climate_model" / "oni_decay_v1.json"

# Original fixed constant, kept as the fallback when the calibrated artifact
# is missing or fails to load.
_FALLBACK_DECAY_RATE = 0.15


def _load_calibrated_decay_rate() -> float:
    """Loads the calibrated `decay_rate` from the versioned artifact exported
    by `calibrate_oni_decay.py`. Falls back to the original fixed constant
    (`0.15`) if the artifact doesn't exist or fails to load for any reason
    (narrow try/except on purpose: any failure here must degrade gracefully
    to the previously-validated behavior, not crash the forecast)."""
    try:
        with _ONI_DECAY_ARTIFACT_PATH.open(encoding="utf-8") as stream:
            artifact = json.load(stream)
        return float(artifact["decay_rate"])
    except (OSError, ValueError, KeyError, TypeError):
        return _FALLBACK_DECAY_RATE


# ---------------------------------------------------------------------------
# Temperature delta features
# ---------------------------------------------------------------------------
def add_temperature_delta_features(
    df: pd.DataFrame,
    columna_temp: str = "temp_mean_mes",
    year_col: str = "year",
    month_col: str = "month",
) -> pd.DataFrame:
    """
    Adds three temperature-delta variants so a later modeling step can compare
    which one carries more signal. Semantic difference between the three:

      - delta_temp_mensual:   temp(mes) - temp(mes-1). Captures short-term
        month-to-month swings (e.g. entering/leaving a heat spell). Very
        sensitive to seasonality itself, since consecutive months are
        naturally different (Jan vs Feb).
      - delta_temp_interanual: temp(mes, año) - temp(mismo mes, año-1).
        Removes the seasonal component (same calendar month compared across
        years), so it isolates anomalies / climate trend for that specific
        month rather than the normal seasonal cycle.
      - delta_temp_lag1: delta_temp_mensual shifted by one period. An
        autoregressive feature on the delta itself (was the temperature
        already rising/falling last month?), useful for momentum-style
        signals independent of the current month's absolute delta.

    Returns a copy of `df` with the three new columns added; rows without a
    valid previous period (start of the series) get NaN.
    """
    out = df.copy()
    out = out.sort_values([year_col, month_col])  # keeps original index labels

    out["delta_temp_mensual"] = out[columna_temp] - out[columna_temp].shift(1)
    out["delta_temp_interanual"] = out[columna_temp] - out.groupby(month_col)[columna_temp].shift(1)
    out["delta_temp_lag1"] = out["delta_temp_mensual"].shift(1)

    # Restore the caller's original row order (sort back by original index labels)
    return out.sort_index()


# ---------------------------------------------------------------------------
# ONI derived features
# ---------------------------------------------------------------------------
def _oni_intensidad_categoria(valor_abs: float) -> str:
    if pd.isna(valor_abs):
        return np.nan
    if valor_abs < 0.5:
        return "debil"
    if valor_abs < 1.0:
        return "moderado"
    if valor_abs < 1.5:
        return "fuerte"
    return "muy_fuerte"


def _oni_fase_categoria(valor: float) -> str:
    if pd.isna(valor):
        return np.nan
    if valor >= 0.5:
        return "nino"
    if valor <= -0.5:
        return "nina"
    return "neutral"


def add_oni_derived_features(
    df: pd.DataFrame,
    columna_oni: str = "oni_actual",
    year_col: str = "year",
    month_col: str = "month",
) -> pd.DataFrame:
    """
    Adds ONI-derived features:
      - oni_lag_1 / oni_lag_3: value 1 and 3 periods back (autoregressive lags).
      - oni_intensidad: categorical, on abs(oni_actual): <0.5 debil,
        0.5-1.0 moderado, 1.0-1.5 fuerte, >1.5 muy_fuerte.
      - oni_fase: nino/nina/neutral using the standard NOAA +-0.5 threshold.
      - oni_transicion: oni_actual - oni_lag_3 (how much the anomaly has moved
        over the last 3 periods -- helps flag a phase transition in progress).
      - oni_duracion_fase: number of consecutive periods (including the
        current one) that have had the same `oni_fase`, computed with a
        simple sequential loop over the series ordered by date.
    """
    out = df.copy()
    original_index = out.index
    out = out.sort_values([year_col, month_col]).reset_index(drop=True)

    out["oni_lag_1"] = out[columna_oni].shift(1)
    out["oni_lag_3"] = out[columna_oni].shift(3)
    out["oni_intensidad"] = out[columna_oni].abs().apply(_oni_intensidad_categoria)
    out["oni_fase"] = out[columna_oni].apply(_oni_fase_categoria)
    out["oni_transicion"] = out[columna_oni] - out["oni_lag_3"]

    # Simple sequential loop: consecutive-run length of the same oni_fase.
    duraciones = []
    racha_actual = 0
    fase_previa = None
    for fase in out["oni_fase"]:
        if pd.isna(fase):
            racha_actual = 0
            fase_previa = None
            duraciones.append(np.nan)
            continue
        if fase == fase_previa:
            racha_actual += 1
        else:
            racha_actual = 1
        fase_previa = fase
        duraciones.append(racha_actual)
    out["oni_duracion_fase"] = duraciones

    out = out.sort_index()
    return out


# ---------------------------------------------------------------------------
# Seasonal (month) cyclical features
# ---------------------------------------------------------------------------
def add_seasonal_month_features(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """
    Adds `mes_sin` / `mes_cos`, replicating exactly the cyclical-month
    transform used by the current monthly production model
    (`prepare_year_monthly_features` in api/services/forecast.py):

        mes_sin = sin(2*pi*mes/12)
        mes_cos = cos(2*pi*mes/12)

    Kept identical on purpose so results are comparable across the macro and
    climate model families.
    """
    out = df.copy()
    months = out[month_col].to_numpy()
    out["mes_sin"] = np.sin(2 * np.pi * months / 12)
    out["mes_cos"] = np.cos(2 * np.pi * months / 12)
    return out


# ---------------------------------------------------------------------------
# ONI future projection
# ---------------------------------------------------------------------------
def project_oni_future(
    df_historico: pd.DataFrame,
    meses_futuros: int,
    columna_oni: str = "oni_actual",
    year_col: str = "year",
    month_col: str = "month",
    n_years_clim: int = 10,
    decay_rate: float | None = None,
) -> pd.DataFrame:
    """
    Projects ONI forward without manual assumptions (unlike the current macro
    model, which relies on user-supplied future macro variables).

    Method: combine (a) seasonal climatology -- the historical mean of the
    same quarterly window over the last `n_years_clim` years -- with (b) a
    decreasing persistence of the current anomaly. The last observed
    anomaly (last actual ONI value minus its own climatology) decays
    exponentially toward zero as we project further into the future:

        oni_proyectado(t) = climatologia(mes_t)
                             + anomalia_0 * exp(-decay_rate * pasos(t))

    where `anomalia_0` is the last real ONI value minus the climatology of
    its own month, and `pasos(t)` counts how many periods ahead `t` is from
    the last real observation.

    Parameters
    ----------
    df_historico : DataFrame with at least [year_col, month_col, columna_oni],
        containing real (non-projected) ONI values.
    meses_futuros : int, how many months ahead to project starting right
        after the last non-null observation in df_historico.
    n_years_clim : how many trailing years of history to use for the
        seasonal climatology of each calendar month.
    decay_rate : exponential decay rate applied per period (per month) to
        the persisted anomaly. Higher = anomaly fades to climatology faster.
        If ``None`` (the default), the calibrated value is loaded from
        ``api/models/climate_model/oni_decay_v1.json`` (produced by
        ``training/forecast_climate/calibrate_oni_decay.py`` against ~76
        years of real ONI history); if that artifact is missing or fails to
        load, falls back to the original fixed constant ``0.15``. An
        explicit value passed by the caller always takes precedence over
        both the artifact and the fallback.

    Returns
    -------
    DataFrame with columns (year, month, oni_proyectado).
    """
    if decay_rate is None:
        decay_rate = _load_calibrated_decay_rate()

    hist = df_historico.dropna(subset=[columna_oni]).sort_values([year_col, month_col]).reset_index(drop=True)
    if hist.empty:
        raise ValueError("df_historico no tiene valores ONI válidos para proyectar.")

    last_row = hist.iloc[-1]
    last_year, last_month = int(last_row[year_col]), int(last_row[month_col])
    last_oni = float(last_row[columna_oni])

    def climatologia_mes(mes: int, hasta_year: int) -> float:
        """Mean ONI for calendar month `mes` over the trailing n_years_clim years
        up to (and including) `hasta_year`, using all available history if fewer
        years exist."""
        ventana = hist[
            (hist[month_col] == mes)
            & (hist[year_col] <= hasta_year)
            & (hist[year_col] > hasta_year - n_years_clim)
        ]
        if ventana.empty:
            ventana = hist[hist[month_col] == mes]
        return float(ventana[columna_oni].mean())

    anomalia_0 = last_oni - climatologia_mes(last_month, last_year)

    resultados = []
    year, month = last_year, last_month
    for paso in range(1, meses_futuros + 1):
        month += 1
        if month > 12:
            month = 1
            year += 1
        clim = climatologia_mes(month, last_year)
        persistencia = anomalia_0 * np.exp(-decay_rate * paso)
        oni_proyectado = clim + persistencia
        resultados.append({"year": year, "month": month, "oni_proyectado": oni_proyectado})

    return pd.DataFrame(resultados)
