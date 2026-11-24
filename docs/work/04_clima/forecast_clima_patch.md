# Parche OPCIONAL de inferencia para el modelo id=4 (Climático+ONI) — para agente-e2e

> **ESTADO (2026-11-24, agente-correcciones): DESCARTADO / REVERTIDO.** Se aplicó en la integración y se
> retiró tras la ronda 1 de revisión: `nys=5` empata con `nys=2` en el walk-forward total (MAPE 4.52 vs 4.52,
> 97 orígenes) y es PEOR en los orígenes posteriores a la selección (2024-01..2025-12: 4.22 vs 3.97). La
> mejora 2026 (5.07 -> 4.45) es un solo origen de 8 meses. `forecast.py` usa `n_years_shape=2` (contrato
> original) y `model_info.pkl` trae `n_years_shape_recomendado = 2`. Este archivo queda solo como registro
> (antes vivía en `api/services/`).

Autor: agente-clima (2026-11-23). **Los artefactos v2 de `api/models/climate_model/` funcionan SIN este
parche** (mismo contrato de carga y mismas features que `forecast.py` ya arma). El parche solo activa una
mejora pequeña validada por walk-forward; aplicarlo es opcional.

## Parche 1 (recomendado, opcional): reparto mensual con `n_years_shape` del artefacto

**Qué**: `predict_future_climate` usa por defecto `n_years_shape=2` (promedio de shares de los últimos 2 años
completos). La validación walk-forward (orígenes ≤ 2023-12, objetivos ≤ 2025-12, sin COVID) prefiere 5 años
(ver `docs/work/04_clima/REPORTE.md`, tabla "nys=5, con parche"). `model_info.pkl` v2 trae la clave
`n_years_shape_recomendado = 5`.

**Dónde**: `api/services/forecast.py`, rama `elif search.id == 4:` de `ForecastService.predict`, en las DOS
llamadas a `predictor.predict_future_climate(...)` (la principal, ~línea 1950, y la "genuina" del empalme,
~línea 1979). Añadir el argumento:

```python
n_years_shape=int(getattr(predictor, 'climate_model_info', {}).get('n_years_shape_recomendado', 2)),
```

Ejemplo (llamada principal):

```python
annual_predictions, monthly_predictions = predictor.predict_future_climate(
    forecast_years=num_years_to_predict,
    atypical_years=atypical_years,
    cutoff_year=cutoff_year,
    enso_phase_by_month=getattr(search, 'enso_phase_by_month', None),
    n_years_shape=int(getattr(predictor, 'climate_model_info', {}).get('n_years_shape_recomendado', 2)),
)
```

**No cambia**: la firma ni el default de `predict_future_climate` (los tests de
`tests/forecast_climate/test_predictor.py` llaman directo con el default 2 y siguen igual), ni el contrato de
escenarios `enso_phase_by_month` (los shares condicionales por fase usan todos los años completos, no
`n_years_shape`).

**Verificación**: `python -m pytest tests/forecast_climate -q` y un `POST /v1/forecast` con id=4,
2026-01-01..2027-12-31: el total anual NO cambia (el reparto se renormaliza al total), solo la forma mensual.

## Descartado (no aplicar)
- Feature `oni_mean_anio_lag1` (sensibilidad al CAMBIO de ONI, ΔONI): mejora el total anual a 1 año (2.7 vs
  3.2 % en validación) pero NO mejora el MAPE mensual h=1..24 (4.65 vs 4.51 %); no justifica tocar la
  inferencia. El código de evaluación lo soporta (`training/forecast_climate/clima_oni_model.py`, `ONI_DELTA`).
- Anualización del año parcial por shares en vez de 12/n: diferencia ≤ 0.8 % con 8 meses; no vale el cambio.
