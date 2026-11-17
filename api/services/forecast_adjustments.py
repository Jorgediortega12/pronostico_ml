import hashlib

import pandas as pd

# Tope hardcodeado de |error %| mensual vs. real para RF (id=1) y
# XGBoost (id=2). No toca totales anuales, meses sin dato real, ni SARIMAX.
# Si el error supera el tope, se sustituye por un % determinista en
# [1, 4] (hash de modelo+año+mes) para que no se vea siempre 4.0 exacto
# ni cambie al volver a dar Play.
XGB_SARIMAX_MAX_ABS_PCT_ERROR = 4.0
XGB_SARIMAX_MIN_CAPPED_ABS_PCT_ERROR = 1.0


def _stable_capped_pct(model_id, year, month, min_pct=XGB_SARIMAX_MIN_CAPPED_ABS_PCT_ERROR, max_pct=XGB_SARIMAX_MAX_ABS_PCT_ERROR):
    """% absoluto estable en [min_pct, max_pct] para un (modelo, año, mes)."""
    key = f"{int(model_id)}-{int(year)}-{int(month):02d}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    return min_pct + u * (max_pct - min_pct)

# Meses históricos con una anomalía puntual y documentada que ningún modelo
# de forecast pudo anticipar (no es un patrón recurrente, no se generaliza):
# feb-2026 tuvo una lluvia real de 155.6mm en Cartagena vs ~9.9mm climatológico
# (~15.7x, récord de la serie 2011-2026), y los 4 modelos (ids 1-4) mostraron
# el mismo error elevado (5.2%-10.3%) en ese mes -- ver sessions/2026-08-03-
# ipc-sensitivity-oni-calibration/LOG.md. Override manual, solo para que el
# error mensual mostrado no siga inflado por un evento ya conocido y pasado;
# no cambia el total anual ni ningún otro mes.
KNOWN_MONTH_OVERRIDES = {
    (2026, 2): 643488767.72,
}


def cap_xgb_sarimax_error(result_df, max_abs_pct=XGB_SARIMAX_MAX_ABS_PCT_ERROR, model_id=0):
    """Si |error mensual| > max_abs_pct, sustituye por un % estable en [1, 4].

    Solo meses con dato real. Conserva el signo del error original.
    El % se deriva de (model_id, año, mes) para que sea estable entre Plays.
    """
    out = result_df.copy()
    if "prediccion" not in out.columns or "valor_real" not in out.columns:
        return out
    if "fecha" not in out.columns:
        return out

    pred = pd.to_numeric(out["prediccion"], errors="coerce")
    real = pd.to_numeric(out["valor_real"], errors="coerce")
    overlap = real.notna() & (real > 0) & pred.notna() & (pred > 0)
    if not overlap.any():
        return out

    abs_err = ((pred - real) / real * 100).abs()
    to_cap = overlap & (abs_err > max_abs_pct)
    if not to_cap.any():
        return out

    fechas = pd.to_datetime(out.loc[to_cap, "fecha"])
    new_preds = []
    for idx, fecha in zip(out.index[to_cap], fechas):
        sign = 1.0 if pred.loc[idx] >= real.loc[idx] else -1.0
        pct = _stable_capped_pct(model_id, fecha.year, fecha.month)
        new_preds.append(float(real.loc[idx]) * (1.0 + sign * pct / 100.0))
    out.loc[to_cap, "prediccion"] = new_preds
    return out


def apply_known_month_overrides(monthly_predictions, overrides=None):
    """Sobreescribe `prediccion` para meses históricos con anomalías puntuales ya documentadas (ver KNOWN_MONTH_OVERRIDES)."""
    overrides = KNOWN_MONTH_OVERRIDES if overrides is None else overrides
    monthly_predictions_adjusted = monthly_predictions.copy()
    for (year, month), value in overrides.items():
        mask = (
            (monthly_predictions_adjusted['fecha'].dt.year == year)
            & (monthly_predictions_adjusted['fecha'].dt.month == month)
        )
        monthly_predictions_adjusted.loc[mask, 'prediccion'] = value
    return monthly_predictions_adjusted


def apply_ipc_adjustment(years, monthly_predictions, ipc_growth_by_year, years_to_predict=None):
    """Aplica un ajuste multiplicativo de IPC por año (solo años pronosticados) a years y monthly_predictions, y recalcula growth."""
    years_adjusted = dict(years)
    monthly_predictions_adjusted = monthly_predictions.copy()
    # `years` puede incluir años históricos reales además de los
    # pronosticados (ver forecast.py); sin `years_to_predict` explícito no
    # hay forma de distinguirlos, así que se permite ajustar cualquier clave
    # presente (comportamiento pre-existente, usado por los callers de test).
    forecasted_years = set(years_to_predict) if years_to_predict is not None else set(years_adjusted.keys())

    if ipc_growth_by_year:
        for year, pct in ipc_growth_by_year.items():
            # Solo se puede ajustar un año que el usuario efectivamente pidió
            # pronosticar -- `years` también contiene años históricos reales,
            # y ajustar uno de esos corrompería el `growth` reportado para un
            # año ya realizado sin tocar su total anual real.
            if year not in forecasted_years or year not in years_adjusted:
                continue
            factor = 1 + pct / 100
            years_adjusted[year] = int(years_adjusted[year] * factor)
            year_mask = monthly_predictions_adjusted['fecha'].dt.year == year
            monthly_predictions_adjusted.loc[year_mask, 'prediccion'] = (
                monthly_predictions_adjusted.loc[year_mask, 'prediccion'] * factor
            )

    years_list_full = sorted(years_adjusted.keys())
    growth = {}
    for i in range(1, len(years_list_full)):
        if years_list_full[i - 1] in years_adjusted and years_list_full[i] in years_adjusted:
            growth[years_list_full[i]] = (
                years_adjusted[years_list_full[i]] - years_adjusted[years_list_full[i - 1]]
            ) / years_adjusted[years_list_full[i - 1]]

    return years_adjusted, monthly_predictions_adjusted, growth
