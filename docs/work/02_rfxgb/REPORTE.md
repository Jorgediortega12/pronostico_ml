# REPORTE agente-rfxgb (ids 1 RF, 2 XGB) v2

> **Nota de honestidad (ronda 2, docs/work/07_correcciones_r2).** La selección usó orígenes <= 2023-12 con
> objetivos <= 2025-12, así que de la fila "Orígenes 2024-01..2025-12 (n=440)" solo los **objetivos >= 2026-01
> (n=164)** son fuera de muestra pura. Separado: objetivos <= 2025-12 (n=276, ya vistos al seleccionar) RF 6.67,
> XGB 6.29, naive+drift 4.24; **objetivos >= 2026 (n=164) RF 5.92, XGB 5.79, naive+drift 5.40**. RF/XGB pierden
> contra naive+drift en ambas. La API exige >= 58 meses para re-entrenar en hindcast (inicio mínimo 2015-11).
Enfoque: log(demanda) mensual directo a 1-24+ meses. Base = nivel móvil 12m x índice estacional medio 3 años x drift; el ML aprende solo la corrección. Features solo con info hasta el start + calendario colombiano (laborables, sábados, domingos, festivos vía holidays_co). SIN clima ni ENSO (enso.csv solo clasifica años ex-post => fuga). Entrena <2026-09 (antes del quiebre). Selección solo con orígenes 2017-12..2023-12.
- RF: min_samples_leaf=20, max_features=0.5, COVID excluido como target. XGB: max_depth=2, min_child_weight=5, lr=0.03.
Módulo: `api/services/forecast_rfxgb.py::predict_monthly(model_id, history, start, end, market_break_month=None, post_break_factor=1.0)`; ignora historia >= start y post-quiebre; fechas desde start; maneja start 2027/año parcial; historia = meses completos (`target_series`). Artefactos: `api/models/{rf,xgb}_model/v2/` (entrenados 2011-01..2026-08: backtest 2026 con ellos es in-sample). .pkl legacy intactos.

## MAPE mensual (%) walk-forward h=24, sin COVID ni >=2026-09
| Conjunto | RF v2 | XGB v2 | media RF+XGB | naive+drift |
|---|---|---|---|---|
| 97 orígenes mensuales 2017-12..2025-12 | 5.56 | 5.52 | 5.49 | 6.13 |
| Orígenes <=2023-12 | 5.31 | 5.34 | 5.26 | 6.57 |
| Objetivos 2024-2025 | 5.10 | 4.91 | 4.89 | 4.46 |
| **Orígenes 2024-01..2025-12 (posteriores a la selección; objetivos con real hasta 2026-08, n=440)** | **6.39** | **6.11** | 6.24 | **4.67** |
| idem, solo objetivos <= 2025-12 (n=276) | 6.67 | 6.29 | 6.48 | 4.24 |

**Lectura honesta (añadida en 06_correcciones):** el titular (5.56/5.52 vs 6.13) mezcla orígenes <= 2023-12, que
son los que se usaron para ELEGIR offset e hiperparámetros, con 24 orígenes posteriores. En esos orígenes
posteriores a la selección — los únicos realmente fuera de muestra — **RF y XGB pierden contra naive+drift**
(6.39/6.11 vs 4.67 %, sesgo +5.9/+5.4 %). NO se puede afirmar que sean "mejores que naive+drift".
Causa (sobreajuste de nivel de 2024): las features de crecimiento al origen (g12, g3, drift, g_since_b) y la
corrección ML aprendieron en 2011-2023 que el crecimiento reciente persiste; tras el salto de nivel de 2024 lo
extrapolan a 2025 y sobreestiman (pronosticar 2025 desde 2024-12: 7.5-7.8% vs 3.05% naive+drift).
Mezcla 50/50 con naive+drift (4.19% en 2024-25 pero 5.55% orígenes<=2023) NO adoptada: se calculó tras ver el
test y adoptarla sería tunear contra él; si se quiere, debe fijarse la regla de antemano y validarse en orígenes
nuevos (2026+).

## Test final 2026 ene-ago (una vez, origen 2025-12, sin tope/override) — UN solo origen, 8 meses: no concluyente
RF v2 3.69 (antes 6.52) | XGB v2 4.41 (antes 6.27) | media 4.05 | naive+drift 3.96 (drift 36m; 5.47 con drift 12m de DIAGNOSTICO).
Pronóstico 2027 (escala mercado completo): RF 9.79 / XGB 9.71 TWh (start 2026-01); 9.91 / 9.82 (start 2027-01).
