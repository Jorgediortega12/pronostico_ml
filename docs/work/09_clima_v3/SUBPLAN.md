# SUBPLAN clima v3 (id=4) — agente-clima-v3

Lee primero: `PLAN.md` (reglas), `docs/work/04_clima/REPORTE.md`, `docs/work/08_sanidad/REPORTE.md`.

## Hallazgos de partida (2026-11-24)
- v2 usa cutoff = start_year-1 (solo años completos): ignora los meses ya observados del año en curso y el nivel reciente.
- Δ12 log demanda vs Δ12 ONI (contemporáneo, 2012-2026 sin COVID 2020-03..2022-02): beta ≈ 2.26 %/unidad ONI (se 0.26 %), r=0.58
  => señal ENSO real en el modelo mensual yoy (en v2 anual se había diluido).

## Diseño de candidatos (módulo `training/forecast_climate/clima_v3.py`, evaluación `walkforward_clima_v3.py`)
- A: recursivo yoy: D_t = D_{t-12}·exp(g + beta·(ONI_t − ONI_{t-12})); g = deriva reciente (K meses) mezclada con larga (w).
- B: regresión log-nivel sobre ventana W: y − beta·ONI = a + g·t + estacionalidad(12 dummies); pendiente OLS o fija.
- beta por OLS expansivo sobre Δ12 (sin pares COVID), >= 0, con shrinkage opcional.
- ONI futuro = `project_oni_future` (igual que inferencia); escenarios = mediana de fase (`ENSO_PHASE_ONI`).
- Selección SOLO con orígenes mensuales <= 2023-12 (objetivos <= 2025-12); 2024-2025 validación; 2026 una vez.

## Progreso real (2026-11-24, TODO hecho salvo REPORTE.md/sessions, que el orquestador escribe desde el reporte final)
- [x] lectura de plan/código; exploración beta ONI
- [x] `clima_v3.py` + `walkforward_clima_v3.py` (grillas 1-3 en `resultados/tuning_grid*_v3.csv`; familia A descartada)
- [x] tuning (<=2023-12) -> CHOSEN = familia B, window=60, k_recent=72, w_recent=0.5, beta OLS expansivo >=0, oni_lag=0; VAL 2024-25; TEST26 mirado una vez (`--final`)
- [x] decisión: INTEGRAR (ver resumen final del agente: MAPE TUNE 4.69->3.91, VAL 4.06->3.32, ene-ago 2026 API 5.07->3.98)
- [x] integración: `climate_v3.json` + `metrics.pkl` (v2 en `v2_backup/`), `forecast.py` (`_predict_future_climate_v3`, `start_month`, `model_version`, rama id 4; env `CLIMATE_MODEL_VERSION=v2` = rollback), `tests/forecast_climate/test_clima_v3.py` (15), pytest 281 pasan
- [x] baseline.py (tag clima_v3, CSV borrados) y sanidad_forma_nivel.py: crecimiento 6.8/4.9/4.9/5.0 %, corr forma 0.992 (sin salto +8.3 %)
- [ ] REPORTE.md (la herramienta bloqueó la escritura desde el subagente) + sección "clima v3" en sessions/2026-11-23-mejora-modelos-forecast.md: pendiente del orquestador
Reproducir: `python -m training.forecast_climate.walkforward_clima_v3 --tune|--tune2|--tune3|--final`; `python -m training.forecast_climate.train_final_clima_v3`.
