# Sub-plan agente-rfxgb (modelos id=1 RF, id=2 XGB)

Lee: `docs/work/PLAN.md`, `docs/work/00_entorno/DIAGNOSTICO.md`, `docs/work/01_datos/REPORTE.md`.

## Diagnóstico de partida
- RF/XGB actuales = Lasso anual compartido (lag_1 + drift, entrenado hasta 2022) × share mensual RF/XGB.
  MAPE crudo ene-ago 2026: 6.52 % / 6.27 % (naive+drift 5.47 %). Walk-forward piso naive+drift 6.13 %.
- Features inconsistentes train/serve (pct_lag_1, enso→0, trend fuera de rango); colapso con start 2027.

## Enfoque (se elige por validación, no por intuición)
Pronóstico **directo multi-horizonte** sobre log(demanda) con un *offset* estructural:
`log y[o+h] = offset(o,h) + f_ML(features(o,h))`, h = 1..H_MAX.
- Offsets candidatos: (a) naive estacional + drift (mismo mes último año × (1+drift)^k);
  (b) nivel TTM (12 meses móviles) × índice estacional medio 3 años × crecimiento.
- Features (todas con info ≤ origen, salvo calendario del mes objetivo, que es determinista):
  h, mes objetivo (sin/cos), días, laborables, sábados, domingos+festivos (holidays_co) del mes objetivo y
  diferencia vs el mes base; crecimientos al origen (yoy TTM, yoy 3 meses, CAGR 3 años, drift harness);
  desviación del mes base vs su TTM; índice estacional histórico del mes.
- Learners distintos: id1 = RandomForestRegressor (bagging, árboles profundos);
  id2 = XGBRegressor (boosting superficial, shrinkage). Ensamble media solo si mejora en validación.
- Clima/ONI: en inferencia solo habría climatología (≈ estacionalidad, redundante) y ONI anual ex-post
  (fuga). Se documenta y no se usa salvo que ONI-al-origen mejore en validación.
- Quiebre de mercado: entrenar con `target_series(..., market_break=mb)` (corte en el quiebre) y además
  cortar meses post-quiebre en inferencia; salida en escala mercado completo + factor opcional.

## Protocolo de validación
- Harness `training/common/walkforward.py`, ventana expansiva, orígenes mensuales.
- Selección (variantes/hiperparámetros): orígenes 2017-12..2023-12, h=24 (objetivos ≤ 2025-12), excl. COVID.
- Reporte comparable con piso 01_datos: orígenes 2017-12..2025-12, h=24 (se corre UNA vez al final).
- Test final intocable: origen 2025-12, ene-ago 2026 crudo, una sola vez.

## Entregables
- `training/forecast_rfxgb/` (features.py, experiments.py, train.py) leen la BD.
- `api/services/forecast_rfxgb.py` autocontenido: `predict_monthly(model_id, history, start, end, ...)`.
- Artefactos en `api/models/rf_model/v2/` y `api/models/xgb_model/v2/` (no se tocan los .pkl que usa forecast.py).
- Tests `tests/forecast_rfxgb/`.
- `docs/work/02_rfxgb/REPORTE.md`.

## Progreso real
- [x] Lectura de docs y harness
- [x] Features + experimentos de selección (seleccion_resumen.csv)
- [x] Entrenamiento final + artefactos (v2/)
- [x] Módulo de inferencia + tests
- [x] Reporte walk-forward completo + test 2026 (una vez) -> REPORTE.md
- [x] pytest completo (192 passed)
