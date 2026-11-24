# 01_datos — Sub-plan (agente-datos)

Lee: `docs/work/PLAN.md`, `docs/work/00_entorno/DIAGNOSTICO.md` (§2, §4-Transversal).
Objetivo: una base común (datos limpios + harness walk-forward + baselines) que usan agente-rfxgb,
agente-sarimax y agente-clima. Sin fuga, sin hardcodear valores de negocio, sin inventar futuro.

## Entregables
| Archivo | Contenido |
|---|---|
| `training/common/config.py` | Constantes de dominio documentadas (periodo COVID excluido de métricas, ventana de dedupe, URL por defecto). |
| `training/common/data.py` | `load_daily_demand`, `deduplicate_daily`, `aggregate_monthly`, `load_monthly_demand`, `target_series`, `MarketBreak`, `load_market_break`, `to_full_market_scale`. |
| `training/common/walkforward.py` | `make_origins`, `walk_forward`, `seasonal_naive`, `seasonal_naive_drift`, `default_exclusions`, `summarize`, `evaluate_with_baselines`. |
| `tests/common/test_*.py` | Unitarios sintéticos (fuga, dedupe, imputación, baselines, métricas) + integración `integration_db` (skip sin BD). |
| `docs/work/01_datos/run_baselines.py` | Corre baselines naive sobre la BD real, orígenes 2017-12..2025-12. |
| `docs/work/01_datos/baselines_walkforward.{md,csv}` | Piso a superar. |
| `docs/work/01_datos/REPORTE.md` | API, decisiones, hallazgos, uso. |

## Decisiones de diseño
1. **Fuente**: `PRONOSTICO_demands.total` (verificado = Σ hora_1..24; el loader re-verifica y cuenta discrepancias).
2. **Dedupe por fecha** (2020-Q4): para cada fecha con >1 fila se elige la fila cuyo `total` está más cerca
   (|log(total/ref)|) de la mediana de los días NO duplicados en ±N días (N en config). Criterio verificable:
   se guarda el log de decisiones (fecha, candidatos, ref, elegido, razón entre candidatos).
3. **Meses incompletos**: `completo = dias_con_dato == dias_mes`. `preliminar` = mes con filas `tx2`.
   Imputación opcional (promedio diario × días) SOLO para huecos internos del histórico txf (mes incompleto
   seguido por algún mes completo); nunca el último mes parcial ni meses preliminares ni el futuro.
4. **Quiebre de mercado**: `demand_split_config` (fecha_limite, porcentaje), misma semántica que
   `api/services/demand_config_service.adjust_period_value`: meses cuyo fin ≤ fecha_limite = escala mercado
   completo; posteriores = porción del nuevo mercado. `to_full_market_scale` divide los meses post-quiebre por
   porcentaje/100 con el valor configurado; no estima el factor desde los datos (2 meses → inestable).
5. **Walk-forward**: expanding window; el `fit_predict_fn` solo recibe una copia de la serie truncada ≤ origen.
   Métricas excluyen COVID (2020-03..2021-02) y meses ≥ primer mes post-quiebre, y meses no reales (imputados o
   incompletos) vía serie `actuals` separada de la de entrenamiento.
6. **Baselines**: naive estacional (último mismo mes disponible); naive estacional + drift (media de yoy
   mensuales de los últimos 36 meses del train, compuesta por años saltados).

## Progreso real
- [x] Lectura de docs y exploración BD (duplicados 2020-10..12: 92 fechas, razón 8.2–11.8×; split local 2026-09-01/100 %).
- [x] data.py  - [x] walkforward.py  - [x] tests (18; suite completa 132 passed)  - [x] baselines BD real  - [x] REPORTE.md
- Pendiente (orquestador): revisión multi-ángulo y commit.
