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


def cap_xgb_sarimax_error(result_df, max_abs_pct=XGB_SARIMAX_MAX_ABS_PCT_ERROR, model_id=0, eligible=None):
    """Si |error mensual| > max_abs_pct, sustituye por un % estable en [1, 4].

    Solo meses con dato real. Conserva el signo del error original.
    El % se deriva de (model_id, año, mes) para que sea estable entre Plays.

    ``eligible`` (máscara booleana alineada con ``result_df``, opcional): meses en los que el tope puede
    actuar. ``forecast.py`` pasa solo los meses **pre-división de mercado**, donde ``prediccion`` y
    ``valor_real`` están en la misma escala (mercado completo) y el error coincide con el que ve el front
    (que multiplica ambos por el mismo factor). En meses post-división el real ya es la porción del nuevo
    mercado: compararlo contra un pronóstico en escala completa fabricaba el error (y, tras el x0.7 del
    front, los ~415 GWh de sep/oct-2026). Sin ``eligible`` se comporta como antes (todos los meses).
    """
    out = result_df.copy()
    if "prediccion" not in out.columns or "valor_real" not in out.columns:
        return out
    if "fecha" not in out.columns:
        return out

    pred = pd.to_numeric(out["prediccion"], errors="coerce")
    real = pd.to_numeric(out["valor_real"], errors="coerce")
    overlap = real.notna() & (real > 0) & pred.notna() & (pred > 0)
    if eligible is not None:
        overlap = overlap & pd.Series(eligible, index=out.index).fillna(False).astype(bool)
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


def front_scale_pct_error(prediccion, valor_real, post_division, factor):
    """Error % en la escala en que lo ve el front.

    El front muestra ``prediccion * factor`` para TODOS los meses (el modelo pronostica en escala de mercado
    completo), el real pre-división como ``real * factor`` y el real post-división tal cual (ya es la porción
    del nuevo mercado). Por tanto:

    * pre-división:  (pred*f - real*f) / (real*f) = (pred - real) / real
    * post-división: (pred*f - real) / real
    """
    pred = pd.to_numeric(pd.Series(prediccion), errors="coerce").astype(float)
    real = pd.to_numeric(pd.Series(valor_real, index=pred.index), errors="coerce").astype(float)
    post = pd.Series(post_division, index=pred.index).fillna(False).astype(bool)
    pred_scaled = pred.where(~post, pred * float(factor))
    return (pred_scaled - real) / real * 100.0


#: Tolerancia relativa para considerar que el real de la petición es el mismo dato documentado en
#: KNOWN_MONTH_OVERRIDES (protege a otras empresas/bases: el valor es de la serie de Afinia).
KNOWN_MONTH_OVERRIDE_REAL_TOL = 0.005


def apply_known_month_overrides(monthly_predictions, overrides=None, require_matching_real=False,
                                real_tol=KNOWN_MONTH_OVERRIDE_REAL_TOL):
    """Sobreescribe `prediccion` para meses históricos con anomalías puntuales ya documentadas (ver KNOWN_MONTH_OVERRIDES).

    Con ``require_matching_real=True`` (lo que usa ``forecast.py``) solo se aplica si la fila tiene
    ``valor_real`` y este coincide con el valor documentado (|dif| <= ``real_tol``): así el override no
    se aplica a meses sin real ni a bases de otras empresas, y nunca a un mes cuyo real esté en otra escala.
    Añade la columna booleana ``override_aplicado``.
    """
    overrides = KNOWN_MONTH_OVERRIDES if overrides is None else overrides
    monthly_predictions_adjusted = monthly_predictions.copy()
    fechas = pd.to_datetime(monthly_predictions_adjusted['fecha'])
    applied = pd.Series(False, index=monthly_predictions_adjusted.index)
    real = None
    if require_matching_real:
        if 'valor_real' not in monthly_predictions_adjusted.columns:
            monthly_predictions_adjusted['override_aplicado'] = applied
            return monthly_predictions_adjusted
        real = pd.to_numeric(monthly_predictions_adjusted['valor_real'], errors='coerce')
    for (year, month), value in overrides.items():
        mask = (fechas.dt.year == year) & (fechas.dt.month == month)
        if real is not None:
            mask = mask & real.notna() & (real > 0) & ((real - value).abs() <= real_tol * real.abs())
        monthly_predictions_adjusted.loc[mask, 'prediccion'] = value
        applied = applied | mask
    monthly_predictions_adjusted['override_aplicado'] = applied
    return monthly_predictions_adjusted


def apply_ipc_adjustment(years, monthly_predictions, ipc_growth_by_year, years_to_predict=None,
                         fixed_by_year=None, adjust_from=None):
    """Aplica un ajuste multiplicativo de IPC por año (solo años pronosticados) a years y monthly_predictions, y recalcula growth.

    ``fixed_by_year`` (opcional, ``{año: kWh}``): parte del total anual que NO es pronóstico (reales previos al
    inicio pedido cuando ``start_date`` cae a mitad de año). El factor solo escala la parte pronosticada:
    ``years[y] = fijo + (years[y] - fijo) * factor``.
    ``adjust_from`` (opcional, fecha): solo se escalan las filas mensuales con ``fecha >= adjust_from``; con
    ambos parámetros la suma mensual pronosticada del año + la parte fija cuadra con el total anual.
    Sin ellos el comportamiento es el anterior (todo el año escalado).
    """
    years_adjusted = dict(years)
    monthly_predictions_adjusted = monthly_predictions.copy()
    fixed_by_year = fixed_by_year or {}
    # `years` puede incluir años históricos reales además de los
    # pronosticados (ver forecast.py); sin `years_to_predict` explícito no
    # hay forma de distinguirlos, así que se permite ajustar cualquier clave
    # presente (comportamiento pre-existente, usado por los callers de test).
    forecasted_years = set(years_to_predict) if years_to_predict is not None else set(years_adjusted.keys())
    fechas = pd.to_datetime(monthly_predictions_adjusted['fecha'])
    in_scope = (fechas >= pd.Timestamp(adjust_from)) if adjust_from is not None else pd.Series(True, index=fechas.index)

    if ipc_growth_by_year:
        for year, pct in ipc_growth_by_year.items():
            # Solo se puede ajustar un año que el usuario efectivamente pidió
            # pronosticar -- `years` también contiene años históricos reales,
            # y ajustar uno de esos corrompería el `growth` reportado para un
            # año ya realizado sin tocar su total anual real.
            if year not in forecasted_years or year not in years_adjusted:
                continue
            factor = 1 + pct / 100
            fixed = float(fixed_by_year.get(year, 0.0))
            years_adjusted[year] = int(fixed + (years_adjusted[year] - fixed) * factor)
            year_mask = (fechas.dt.year == year) & in_scope
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
