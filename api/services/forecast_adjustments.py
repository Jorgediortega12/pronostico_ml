import pandas as pd

# Ajustes aplicados sobre el resultado del forecast (portado de GMR_BackEnd).
# NO se incluyen aquí los topes artificiales del error ni los overrides fijos de meses
# (cap_xgb_sarimax_error / KNOWN_MONTH_OVERRIDES) que existen en el original, porque
# alteran el resultado de los modelos solo para que se vea más exacto.


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
