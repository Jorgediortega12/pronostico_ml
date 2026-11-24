# REPORTE agente-e2e

> **Cifras vigentes (ronda 2, 2026-11-24; CSV regenerados con `scripts/smoke_http.py` contra uvicorn + BD local,
> 30/30 peticiones OK, 1 de ellas error controlado esperado).** Clima usa `n_years_shape=2`: MAPE crudo ene-ago 2026
> **RF 3.69, XGB 4.41, SARIMAX 3.04, Clima 5.07**; con tope+override 1.36 / 1.83 / 2.29 / **4.01**. Curvas 2027:
> feb mín 728-756, jul máx 844-865 GWh, anual 9.73-9.85 TWh. Las cifras de Clima "4.45 / 3.47" de abajo eran con el
> parche nys=5 (retirado). `fechas` vuelve al formato de main (fin de mes ids 1-3, inicio de mes id 4).
Integración v2 en api/services/forecast.py (+ forecast_history.py, forecast_adjustments.py). Tope solo en misma escala; override feb-2026 solo si el real coincide; campos opcionales nuevos: por mes `valores_pronostico_crudos`, `errores_porcentuales_crudos`, `tope_aplicado`, `override_aplicado`, `post_division`, `factor_division`; globales `metricas_error_crudas`, `escala_pronostico`, `division_mercado`, `modelo`. `errores_porcentuales` ahora en escala del front.
MAPE ene-ago 2026 (start 2026-01): crudo RF 3.69, XGB 4.41, SARIMAX 3.04, Clima 4.45 (antes de volver a nys=2); con tope+override 1.36/1.83/2.29/3.47. Smoke HTTP 30/30 OK. Curvas 2027 estacionales (feb mín 728-752, jul máx 844-865 GWh), anual 9.73-9.85 TWh. Resultados en docs/work/05_e2e/.
Abiertos: id4 lee CSV propio (no la BD de la petición); annual_predictions de años pasados salen de la BD.
