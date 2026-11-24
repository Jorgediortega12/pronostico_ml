# SUBPLAN nivel adaptativo (ids 1 RF, 2 XGB, 4 Clima v3; evalua id 3) — agente-nivel

Lee primero: `PLAN.md` (reglas), `docs/work/09_clima_v3/REPORTE.md`.

## Diseno
- Modulo puro `training/common/level_adapt.py` (sin I/O): dado el historico observado hasta `last`, un
  pronosticador de referencia `ref_fn(hist_hasta_origen_ref, horizonte)` y los meses a pronosticar, calcula residuos
  log(real/pron) FUERA DE MUESTRA de los ultimos K meses observados (pronostico hecho desde el origen `last-K`,
  reajustado solo con datos <= origen), los resume (media/mediana con pesos exponenciales), y devuelve factores
  multiplicativos `exp(phi * L * 0.5**(h/hl_h))` recortados a `[-cap, cap]` (log). Sin residuos validos => factor 1.
- Residuos en meses COVID (config) se excluyen. Escala: log (multiplicativo), misma escala que el modelo.
- Cache de predicciones por origen (RF/XGB/SARIMAX/Clima/naive+drift, h=1..24) en
  `docs/work/11_nivel_adaptativo/resultados/pred_cache.csv.gz` para que el walk-forward sea una operacion vectorizada.
- Tuning SOLO origenes <= 2023-12 (objetivos <= 2025-12); VAL = origenes 2024-01..2025-12 (objetivos <= 2025-12);
  2026 (objetivos >= 2026-01) se mira UNA vez al final con el spec congelado.

## Progreso real (se actualiza)
- [x] lectura de plan/codigo, BD local OK (127.0.0.1:55432)
- [x] `training/common/level_adapt.py` (nucleo puro; paridad exacta con la evaluacion vectorizada) - tests pendientes
- [x] cache de predicciones: `python -m training.level_adapt.build_cache` (127 origenes 2016-01..2026-07, 5 modelos, 49 s con 14 procesos) -> resultados/pred_cache.csv.gz
- [x] tuning solo TUNE (`python -m training.level_adapt.evaluate --grid`; grillas tuning_grid.csv, tuning_grid2.csv, tuning_grid_sinCOVID.csv). Hallazgo: en TUNE la persistencia del error de nivel es corta (hl_h=3 gana; hl_h>=12 empeora); spec congelada CHOSEN en evaluate.py (k=6, mediana, hl_w=2, phi=.5, hl_h=3, cap=.06) por regla ex-ante (ventana mas larga dentro de 0.01 del optimo)
- [x] VAL visto (`--report`): RF 6.67->6.23, XGB 6.29->5.85, Clima 3.32->3.19, SARIMAX 4.19->4.07
- [x] TEST2026 visto una sola vez (`--final`): pooled ~neutro (RF 5.77->5.73, XGB 5.76->5.73, Clima 4.15->4.10, SARIMAX 3.96->3.99); origenes 2026 PEOR (RF 4.89->5.31, XGB 5.55->6.25). Decision: NO activar por defecto
- [x] integracion opt-in (`LEVEL_ADAPT=1`, por defecto APAGADA): api/services/forecast_level.py, api/models/level_adapt.json, forecast.py, clima_v3.predict_years(level_spec). Campo `ajuste_nivel` solo con flag. Latencia: +1.0-1.9 s frio RF, +0.5 s XGB, +0 Clima; tibio ~0
- [x] tests/level_adapt/test_level_adapt.py (30); `python -m pytest tests -q` = 311 passed
- [x] baseline.py start 2026-01/05/07 antes/despues (resultados/baseline_api_pre_post.txt); CSV generados borrados
- [ ] REPORTE.md y seccion en sessions/...: la herramienta bloqueo la escritura de .md desde el subagente; el contenido completo va en el reporte final del agente para que el orquestador lo vuelque
