# Sub-plan agente-clima (modelo id=4 Climático+ONI)

Lee: `docs/work/PLAN.md`, `docs/work/00_entorno/DIAGNOSTICO.md`, `docs/work/01_datos/REPORTE.md`.

## Restricciones de integración (no tocar `api/services/forecast.py`)
- Inferencia (`predict_future_climate`) lee `dataset_mensual.csv` / `dataset_anual.csv` y los pkl:
  `annual_scaler_x` (StandardScaler con `feature_names_in_` ⊂ {temp_*_anio, humidity_mean_anio,
  wind_speed_mean_anio, rain_sum_anio, oni_mean_anio, lag_1, lag_2, year_index, year_index_norm,
  growth_rate_lag1}), `annual_model` (.predict sobre X escalado → y escalado), `annual_scaler_y`
  (inverse_transform → log1p total), `monthly_model` (solo `feature_names_in_`, no se llama),
  `metrics.pkl`, `model_info.pkl`, `postproc_corrector_v1.json` (cargado, no aplicado por defecto).
- Features de inferencia del año objetivo: clima = climatología 10 años hasta el corte; ONI =
  `project_oni_future` (decay calibrado) o mediana de fase si hay `enso_phase_by_month`; lag_1/lag_2
  = log1p totales previos; growth_rate_lag1 = media de los últimos 3 log-crecimientos (año 1).
- Reparto mensual = media de shares de los últimos `n_years_shape` años completos (default 2 en
  forecast.py — no editable aquí; se evalúa 2/3/5 y se documenta).

## Pasos
1. `training/forecast_climate/build_clean_dataset.py`: regenerar `dataset_mensual/anual.csv` con la capa
   limpia (`training/common/data.py`: dedupe 2020-Q4, imputación 2021-09, corte pre-quiebre) + clima/ONI
   del dataset original (congelado en `training/forecast_climate/data/`), ONI extendido con el CSV NOAA.
2. `training/forecast_climate/annual_climate_model.py`: modelo anual parsimonioso
   log D_t = log D_{t-1} + μ + β·ONI_t (+ variantes), features construidas EXACTAMENTE como en inferencia
   (`project_oni_future` al origen, climatología al origen). Exporta a pkl compatibles.
3. `training/forecast_climate/walkforward_climate.py`: walk-forward con el harness (orígenes dic y
   mensuales 2017-12..2025-12, h=24), "antes" (receta original reentrenada por origen + artefacto
   productivo) vs candidatos; selección de hiperparámetros solo con orígenes ≤2023-12 (objetivos ≤2025-12);
   test final 2026 una sola vez; sensibilidad a escenarios ONI.
4. Corrector post-proceso: evaluar en walk-forward sin fuga; conservar solo si mejora.
5. Reentrenar con todo hasta 2025-12 (más 2026 parcial en el dataset de inferencia), respaldar originales,
   escribir artefactos, tests nuevos en `tests/forecast_climate/`, correr `pytest tests`.
6. `docs/work/04_clima/REPORTE.md`.

## Progreso real
- [x] 1 dataset limpio (`build_clean_dataset.py`) — [x] 2 modelo (`clima_oni_model.py`) — [x] 3 walk-forward
  (`walkforward_clima.py`) — [x] 4 corrector (no mejora, no se aplica) — [x] 5 artefactos (`train_final_clima.py`),
  tests (`tests/forecast_climate/test_clima_v2.py`), pytest verde — [x] 6 REPORTE.md.
- Pendiente para agente-e2e: parche opcional `docs/work/04_clima/forecast_clima_patch.md`; revisar el cambio de umbral
  del test de spread (ver REPORTE).
