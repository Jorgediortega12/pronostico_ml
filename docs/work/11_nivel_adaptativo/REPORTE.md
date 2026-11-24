# REPORTE corrección adaptativa de nivel — 2026-11-24

Implementado por agente (SUBPLAN.md, resultados/). Reporte volcado por el orquestador. **Resultado: implementado pero APAGADO por defecto (opt-in `LEVEL_ADAPT=1`) porque no gana de forma robusta.**

## Qué es
`training/common/level_adapt.py`: residuos fuera de muestra (pronóstico de referencia desde `ultimo_real − k`, reajustado solo con datos hasta ese origen), factor multiplicativo amortiguado (mediana, k=6, hl_w=2, phi=0.5, hl_h=3, cap=0.06) con decaimiento por horizonte. Factor 1 si no hay meses observados, hay COVID o falla el modelo de referencia. Integración en `forecast.py` (ids 1, 2, 4) con campo opcional `ajuste_nivel`. Hiperparámetros elegidos solo con orígenes ≤2023-12 (objetivos ≤2025-12); 2026 visto una vez con la spec congelada. Hallazgo: en datos hasta 2023 el error de nivel se revierte rápido; decaimiento largo (≥12 m) empeora.

## MAPE % (h1-24 agrupado, sin COVID): sin → con corrección
| | TUNE | VAL 2024-25 | TEST 2026 |
|---|---|---|---|
| RF | 5.31 → 5.23 | 6.67 → 6.23 | 5.77 → 5.73 |
| XGB | 5.34 → 5.29 | 6.29 → 5.85 | 5.76 → 5.73 |
| SARIMAX | 4.04 → 4.04 | 4.19 → 4.07 | 3.96 → 3.99 |
| Clima v3 | 3.91 → 3.88 | 3.32 → 3.19 | 4.15 → 4.10 |
| naive+drift | 6.57 | 4.24 | 5.28 |
Ganancia concentrada en h1-3 (RF VAL 4.93 → 3.78); h13-24 sin cambio. Orígenes de 2026 empeoran (RF 4.89→5.31, XGB 5.55→6.25, SARIMAX 3.56→4.03, Clima 4.00→4.08): residuos previos eran de sobreestimación y el escalón llegó después. Sin shock: neutro (2021 y 2023 pierden hasta 0.2 pp).

## Vía API real (hasta 2026-08): RF / XGB / Clima, sin → con
| Start | RF | XGB | Clima |
|---|---|---|---|
| 2026-01 | 3.69 → 3.72 | 4.41 → 4.01 | 3.98 → 3.64 |
| 2026-05 | 6.15 → 6.56 | 6.76 → 7.80 | 4.13 → 4.71 |
| 2026-07 | 5.27 → 3.47 | 6.67 → 4.47 | 4.08 → 2.35 |
Ayuda cuando los meses observados ya son del régimen nuevo (start 2026-07); pierde cuando la ventana de residuos aún es del régimen viejo (start 2026-05). SARIMAX: no conviene aplicarlo.

## Decisión y riesgos
Apagado por defecto. Memoria corta (solo h1-3); un único origen de referencia por petición; selección sobre ~6000 specs (algo optimista); variantes de memoria más corta probadas tras ver 2026 (no cuentan como selección); id 4 lee su CSV propio. Latencia con `LEVEL_ADAPT=1`: +0.5-1.9 s en frío, casi nada en caché tibio. Con flag apagado la respuesta es idéntica (sin `ajuste_nivel`), probado en ids 1-4. 311 tests pasan.

## Revisión (ronda 1) — precisiones de honestidad
- **VAL no es independiente de TUNE:** 300 de 1440 observaciones de TUNE (21 %) tienen objetivo en 2024-25 (orígenes ≤2023-12), los mismos meses reales de VAL. La ganancia de VAL (RF −0.44 pp) se ve en meses que el tuning ya miró.
- **Regla de elección:** la regla escrita ("dentro de 0.01 del óptimo, ventana k más larga") habría dado k=9 (8 specs k=9 dentro de 0.01); la spec elegida fue k=6 (9.ª en el ranking). La elección fue más manual que la regla. Sin fuga con 2026 (el tuning solo usa TUNE).
- **Incertidumbre en TEST 2026** (bootstrap por origen, 31 orígenes, diferencia media con−sin, IC 95 %): RF −0.10 pp [−0.32, +0.10]; XGB −0.08 [−0.36, +0.19]; Clima −0.10 [−0.28, +0.06]; SARIMAX +0.02 [−0.12, +0.17]. Todos incluyen 0: no hay evidencia de mejora en 2026.
- **Tabla "vía API real":** es un solo origen por fila (start 2026-01 = 8 meses objetivo, 2026-05 = 4, 2026-07 = 2). Las ganancias 5.27→3.47 y 4.08→2.35 son anecdóticas.
- **Latencia con LEVEL_ADAPT=1** (start 2026-07, frío/tibio, sin→con): RF 3.0→5.0 s / 0.54→0.57 s; XGB 2.2→2.7 / 0.40→0.45; Clima 0.50→0.68 / 0.45→0.61 (Clima no tiene caché: +0.16 s siempre). Cada petición añade un refit de referencia al LRU de 32 entradas.
- **Hallazgos menores aceptados:** `ajuste_nivel` informa el origen de la llamada hindcast (no el de años futuros en empalme); feb-2026 (override documentado) no se excluye de los residuos (efecto de mediana despreciable); `adjust_forecast` sin try/except ante JSON corrupto con flag encendido; duplicación de helpers (`_excluded`, `level_from_R`) y ramas solo de investigación (`ref_fn_sarimax`, `tau`, `agree`).

## Ronda 1 CERRADA
4 revisores + regresión (sonnet): sin críticos/altos; regresión limpia (flag apagado = JSON idéntico a HEAD en ids 1-4; 25 casos con LEVEL_ADAPT=1 sin fallos). Corregido: `adjust_forecast` ahora omite la corrección ante cualquier error (sirve el pronóstico sin ajustar), imports/variables sin uso. Verificación estándar: pytest 311 passed, pyflakes limpio en archivos nuevos.
Aceptados/documentados: con el flag encendido `valores_pronostico_crudos`/`metricas_error_crudas` incluyen el ajuste de nivel (se aplica antes del tope); `ajuste_nivel` solo reporta la llamada principal (no la de años futuros en empalme); `LEVEL_ADAPT` en el entorno afecta también a walk-forward/benchmarks del id 4 (no dejarlo exportado al evaluar); LRU de reajustes (32) con más presión y semáforo global con el flag encendido; Clima +0.16 s por petición con el flag.
