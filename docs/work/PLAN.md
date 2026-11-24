# PLAN MAESTRO — Mejora de modelos de forecast mensual de demanda (GMR en Sphere)

Fuente de verdad. Todo agente lee esto al empezar y actualiza su sub-carpeta `docs/work/<parte>/`.

## Objetivo
Los 4 modelos de demanda mensual (id1 RF, id2 XGBoost, id3 SARIMAX, id4 Climático+ONI) dan mal en la
demo (ver `Test_screenshots/`): RF/XGB caen a ~415 GWh en sep-oct 2026 (real ~580); SARIMAX y
Climático subestiman ~30-40 GWh. Mañana hay prueba/presentación. Meta: error mensual honesto
(MAPE out-of-sample) lo más bajo posible, curvas plausibles a 2027, API robusta.

## Alcance (decidido con el usuario)
- SOLO forecast mensual de demanda en `gmr_en_sphere` (no MPM, no EPM).
- Mantener tope de error (`cap_xgb_sarimax_error`) y override feb-2026, PERO buscar mejor manera:
  mejorar el modelo para que el tope casi no actúe; reportar siempre métricas CRUDAS (sin tope).
- Prohibido: overfitting, hardcodear valores, fuga de datos (leakage), tunear contra el test final.
- Reglas ML: validación temporal (walk-forward/expanding window), cero fuga, features solo con
  información disponible al momento de predecir, comparación contra baselines (naive estacional).
- Rama git: `mejora/modelos-forecast-noche`. Commits sí, push NO. Backup de .pkl originales en
  `docs/work/backup_modelos_originales/` antes de sobrescribir.

## Entorno
- Postgres local aislado (initdb propio) puerto 55432, db `afinia`, restaurada de `afinia2.backup`
  (user postgres, trust). Arranque: ver `docs/work/00_entorno/README.md`.
- Python: anaconda 3.11 (xgboost 2.0.3, sklearn 1.7.0, statsmodels 0.14.4, pandas 2.2).
- Servicio: FastAPI `api/main.py`; lógica en `api/services/forecast.py` (DemandPredictor, ForecastService).

## Partes y dueños
0. Entorno + línea base (orquestador/agente-0): correr API y backtest crudo de los 4 modelos.
1. Datos/features: auditoría de calidad (huecos, outliers, enso.csv faltante, clima) — agente-datos.
2. RF + XGBoost mensuales (id 1,2) — agente-rfxgb.
3. SARIMAX (id 3) — agente-sarimax.
4. Climático+ONI (id 4) — agente-clima.
5. Integración API/e2e + ajustes (tope/override, front contract) — agente-e2e.
6. Revisión multi-ángulo + regresión (agentes separados, model sonnet).

## Criterio de "terminado"
- Backtest walk-forward reportado antes/después por modelo (MAPE mensual, sesgo, MAPE anual).
- Ningún mes con real disponible con error crudo >~5% salvo anomalía justificada.
- `pytest tests` verde; endpoints /v1/forecast/* responden para los 4 modelos, rangos 2026-2027.
- Todo documentado en `sessions/2026-11-23-mejora-modelos-forecast.md`.
