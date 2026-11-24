# REPORTE clima v3 (id 4 Climático+ONI) — 2026-11-24

Implementado por agente-clima-v3 (detalle de progreso en `SUBPLAN.md`, grillas en `resultados/`). Reporte volcado por el orquestador.

## Qué es v3
- Módulo `training/forecast_climate/clima_v3.py`: mensual, re-estimado en cada llamada con el histórico previo a `start_date`.
- Regresión del log de la demanda sobre ventana de 60 meses: nivel reciente, estacionalidad mensual, deriva `g` y sensibilidad ENSO `β·ONI`.
- β por mínimos cuadrados expansivos sobre `Δ12 log D` vs `Δ12 ONI`, β ≥ 0 (hoy 2.26 % por unidad de ONI, r = 0.58).
- Deriva `g`: mezcla 50/50 de los últimos 72 pares interanuales y la deriva larga.
- Con `start_date` a mitad de año usa los meses ya observados del año en curso.
- Selección: mínimo MAPE con orígenes ≤ 2023-12 (grillas `resultados/tuning_grid*_v3.csv`; superficie plana, 3.91-4.11). 2026 se miró una sola vez tras congelar el spec.
- Walk-forward y API usan la misma función (test de paridad).

## Métricas (MAPE crudo %, mismas filas)
| Periodo | naive+drift | v2 | v3 |
|---|---|---|---|
| TUNE (orígenes ≤ 2023-12, EN MUESTRA de selección) | 6.57 | 4.69 | 3.91 |
| VAL (orígenes 2024-25) | 4.24 | 4.06 | 3.32 |
| TEST 2026 (objetivos ≥ 2026) | 5.28 | 4.09 | 4.15 |
| Orígenes diciembre | 5.97 | 4.55 | 3.72 |
| Ene-ago 2026 vía API (origen 2025-12) | 3.96 | 5.07 | 3.98 |
v3 gana a v2 en 79.8 % de los orígenes mensuales (83 de 104). Pierde o empata en 21: 2019-12, 2020-01 y 2020-10..12, 2021-01..11 (COVID y rebote) y 2024-04..08 (5 meses de validación real, no COVID). Total anual con origen en diciembre: 3.39 → 2.03 (el 3.39 sale de `metrics.pkl['v2']`, otro soporte; mezcla 7 de 9 orígenes ≤2023-12 en muestra con 2024-25, por lo que NO es una métrica fuera de muestra pura).

## Escenarios ENSO (GWh/año)
v2 2026: 8982 / 9096 / 9261 (Niña/Normal/Niño, 3.1 %). v3 2026: 9070 / 9211 / 9417 (3.8 %); v3 2027: 9529 / 9677 / 9893. Orden Niña < Normal < Niño siempre.

## Sanidad de forma y nivel
Crecimiento 6.8 / 4.9 / 4.9 / 5.0 % (2026-2029), sin el salto +8.3 % de 2027; correlación de forma con 2023-25: 0.992 (v2 0.893); pico julio, valle febrero.

## Riesgos honestos
- En el pool TEST 2026 v3 no mejora a v2 (4.15 vs 4.09); a h13-24 desde orígenes 2024-25 pierde (4.26 vs 3.36). Con start 2026-01 empata con naive+drift (3.98 vs 3.96) y mejora a v2 en 1.1 puntos.
- ONI aporta poco: mejora VAL (3.32 vs 4.34 sin ONI), casi nada en TUNE y nada en el pool 2026. El clima (temp/lluvia) no entra al modelo.
- Sesgo +3 % en VAL y −4 a −7 % en may-ago 2026 (escalón de demanda no anticipable). Dataset estático hasta 2026-08. Corrector v2 no aplica a v3.
- `metrics.pkl` trae métricas v3; las de v2 en `metrics.pkl['v2']`.

## Rollback
`CLIMATE_MODEL_VERSION=v2` o borrar `api/models/climate_model/climate_v3.json`. Artefactos v2 en `api/models/climate_model/v2_backup/`.

## Nota de proceso
La selección se hizo en tres etapas de grilla (~280 specs) sobre TUNE; VAL se imprimió en cada etapa aunque no se usó para elegir, así que la selección es algo optimista. La mejora verificable fuera de muestra está en VAL (3.32 vs 4.06) y en ene-ago 2026 vía API (3.98 vs 5.07); en el pool TEST 2026 no hay mejora.

## Revisión multi-ángulo (ronda 1) — resultado
Sin críticos ni altos; regresión sin cambios en ids 1-3 (JSON idéntico a HEAD) ni en id 4 con CLIMATE_MODEL_VERSION=v2. Corregido: MAPE mostrado con rollback v2 (`get_mape` leía v3); reporte (pérdidas 2024-04..08, TUNE en muestra, origen del 3.39).
Aceptados (bajos/medios, documentados): (1) un escenario ENSO forzado en un mes solo mueve ese mes (no se arrastra al año siguiente porque el modelo es yoy); con 5 meses forzados Niño−Niña = 0.9 % en 2026 (el ~3.8 % corresponde a forzar el año completo); (2) fases ENSO sobre meses ya observados se ignoran sin aviso; (3) ONI 'proyectado' 2026-06..08 del CSV se trata como observado; (4) `modelo` ya no trae `n_years_shape` (version v3); (5) duplicación menor de helpers/ENSO_PHASE_ONI y ramas de investigación (familia A, beta_mode) en clima_v3.py; (6) id 4 sigue leyendo CSV propio (estático hasta 2026-08).
