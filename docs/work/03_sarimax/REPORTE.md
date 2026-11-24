# REPORTE agente-sarimax (id=3)

> **Nota de honestidad (ronda 2, docs/work/07_correcciones_r2).** "Orígenes posteriores a la selección" (4.19)
> cubre solo objetivos <= 2025-12 (n=276), que ya se vieron en la selección. Con el mismo protocolo (re-estimación
> por origen, h <= 24, objetivos 2026-01..2026-08) los **objetivos >= 2026 (n=164, fuera de muestra pura) dan
> 4.03** (naive+drift 5.40); total n=440: 4.13. Inicio mínimo en la API: 2014-01 (>= 36 meses).
Modelo: SARIMA(1,0,0)(0,1,1,12)+c sobre log(y/días del mes), datos limpios (training.common.data) cortados en el quiebre 2026-09, sin exógenas, COVID sin tratar. 4 parámetros (original: 21). Elegido entre >70 candidatos solo con orígenes <=2023-12 y reales <=2025-12. Exógenas de calendario, ETS y ensamble no mejoraron; COVID nan/dummy empeora.
Módulo: `api/services/forecast_sarimax.py::predict_monthly(history, start, end, exclude_from=None, refit="auto")` — respeta start_date, re-ancla estado solo con reales < start, re-estima si el artefacto vio meses >= start (sin fuga). Devuelve escala de mercado completo (el factor de división se aplica fuera). Artefacto: `api/models/sarimax_model/sarimax_v2.json`. e2e debe pasar historia limpia (`forecast_history.load_clean_history_session`) y `exclude_from` = primer mes post-quiebre.

## Walk-forward dev 2017-12..2025-12 (97 orígenes mensuales)
| Modelo | MAPE | Sesgo | h1-12 | h13-24 | anual |
|---|---|---|---|---|---|
| Nuevo | 4.07 | +1.69 | 3.69 | 4.51 | 3.29 |
| Original reajustado | 7.37 | +0.88 | 6.03 | 8.96 | 8.35 |
| Naive+drift | 6.20 | +1.01 | 6.00 | 6.42 | 5.89 |

Orígenes posteriores a la selección (2024-01..2025-12, n=576, `report_dev_mensual_detalle.csv.gz`): nuevo 4.19 |
naive+drift 4.24 | naive estacional 3.84 | original 6.19. Honesto: empata/gana levemente a naive+drift fuera de
la muestra de selección, pero no al naive estacional puro en ese corte.

## Test final 2026 (una vez; origen 2025-12, ene-ago) — un solo origen, 8 meses
| Modelo | MAPE | Sesgo | Máx |
|---|---|---|---|
| Nuevo | 3.04 | -0.94 | 6.0 |
| Artefacto original (API crudo) | 3.43 | -1.29 | 7.5 |
| Original reajustado | 3.93 | -2.15 | 7.4 |
| Naive+drift | 3.96 | -1.63 | 7.1 |
Orígenes móviles 2025-12..2026-07: nuevo 3.44, original 4.58. Hindcast 2025: 2.88 %. Ventaja 2026 pequeña (un origen); sigue subestimando may-ago (ago -5.3 %) por salto de demanda no anticipable. Pytest: 19 tests propios OK.
