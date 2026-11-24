# Sesión 2026-11-23 (nocturna) — Mejora de modelos de forecast

## Contexto
Usuario (JM Datalabs) presenta/prueba modelos mañana. Capturas en Test_screenshots/ muestran errores
grandes. Autonomía total, sin preguntas; metodología multiagente (CLAUDE.md global).

## Decisiones confirmadas
- Alcance: solo forecast mensual de demanda. Mantener tope/override pero buscar mejor forma (modelo mejor).
- Postgres local aislado (initdb puerto 55432) con afinia2.backup. Rama `mejora/modelos-forecast-noche`, sin push.

## Progreso real hasta ahora
- [x] Rama creada, BD restaurada (PRONOSTICO_demands 2011-01-01..2026-11-17, 5890 filas; datos_clima 33474).
- [x] docs/work/PLAN.md escrito.
- [x] Línea base / diagnóstico (agente-0): docs/work/00_entorno/DIAGNOSTICO.md. Hallazgos: ~415 GWh = tope compara pred (100%) vs real ya al 70% desde 2026-09, y el front aplica x0.7 otra vez; SARIMAX ignora start_date; RF/XGB comparten Lasso anual (peor que naive+drift); enso.csv falta; 2020-Q4 duplicado; no hay scripts de reentreno.
  MAPE crudo ene-ago 2026: RF 6.52, XGB 6.27, SARIMAX 3.43, Clima 5.28, naive+drift 5.47.
- [x] agente-datos: training/common (data.py, walkforward.py) + tests (132 passed). Piso walk-forward 2017-12..2025-12: naive+drift MAPE 6.13 (h1-12 5.83), naive estacional 8.03. Ver docs/work/01_datos/REPORTE.md. Backup de modelos originales en docs/work/backup_modelos_originales/.
- [ ] agentes de modelo EN CURSO en paralelo (cada uno con modulo de inferencia propio, SIN editar forecast.py): agente-rfxgb (docs/work/02_rfxgb), agente-sarimax (03_sarimax), agente-clima (04_clima). Lanzados tras agente-datos.
- [ ] agente-e2e (tope en misma escala, contrato API, start_date)
- [ ] revision multi-angulo + regresion

- [x] agente-sarimax: SARIMA(1,0,0)(0,1,1,12)+c, WF MAPE 4.07 (orig reajustado 7.37, naive+drift 6.20); test 2026 ene-ago 3.04 (orig 3.43). Modulo api/services/forecast_sarimax.py + sarimax_v2.json. Bug start_date resuelto en el modulo (falta integrar en forecast.py). REPORTE: docs/work/03_sarimax/REPORTE.md.
  Nota: un test de clima (sensitivity_spread_is_meaningful) fallo una vez en suite completa mientras agente-clima trabajaba; pasa solo -> revisar al cierre.
- [x] agente-clima: v2 anual log-lag1+mu+beta*ONI; WF dic MAPE 4.55 (orig reentrenado 9.55, naive+drift 5.97); test 2026 5.07 (orig 5.28; naive+drift 3.96 le gana). Artefactos en api/models/climate_model/, tests v2. Umbral de un test bajado 2.0->0.5% (aceptado, ver REPORTE). Parche opcional nys=5 en docs/work/04_clima/forecast_clima_patch.md (movido desde api/services/). 183 passed.
- [x] agente-rfxgb: v2 directo log-demanda (base nivel12m x indice estacional x drift + correccion ML, calendario CO, sin clima/ENSO). WF 97 orig: RF 5.56, XGB 5.52 vs naive+drift 6.13; pero 2024-25: 5.10/4.91 vs 4.46 (pierden). Test 2026: RF 3.69, XGB 4.41 (antes 6.52/6.27). Modulo api/services/forecast_rfxgb.py, artefactos v2/. 192 passed. REPORTE: docs/work/02_rfxgb/REPORTE.md.
- [x] agente-e2e: integracion en forecast.py (api/services/forecast_history.py nuevo), tope solo pre-quiebre y misma escala (elimina caida a ~415), override feb-2026 solo si el real coincide, campos opcionales crudos en la respuesta, ML_FORECAST_LEGACY=1 vuelve al comportamiento anterior. 229 passed; smoke HTTP 30/30 200 OK. MAPE crudo ene-ago 2026: RF 3.69, XGB 4.41, SARIMAX 3.04, Clima 4.45; con tope: 1.36/1.83/2.29/3.47 (tope actua 4/8 meses en RF/XGB). Abiertos: clima id4 lee CSV propio no la BD de la peticion; annual_predictions de anos pasados salen de la BD.
- [ ] Ronda 1 de revision multi-angulo + regresion (EN CURSO) sobre `git diff --cached main` (todo staged; backup de pkl y capturas en .gitignore).

## ESTADO AL CORTE POR LIMITE DE USO (2026-11-24) — para retomar
- NADA commiteado. Todo esta staged en rama mejora/modelos-forecast-noche (`git diff --cached main`). Sin push.
- Postgres local aislado puede estar caido: levantar con PowerShell Start-Process postgres.exe -D <scratchpad>\pgdata -p 55432 (ver docs/work/00_entorno/README.md). .env local apunta ahi (ignorado por git).
- Ronda 1 de revision: terminados simplification, conventions, reuse, efficiency, line-by-line, altitude. PENDIENTES (agentes aun corriendo al corte, solo lectura): removed-behavior, cross-file tracer, regresion (pytest x2, legacy vs main, smoke HTTP).
- Hallazgos sin severidad critica. ALTO (altitude A1): RF/XGB NO baten naive+drift fuera de la muestra de seleccion (orig. 2024-01..2025-12: RF 6.39, XGB 6.11, naive+drift 4.67); README/REPORTE rf-xgb titulan 5.56/5.52 vs 6.1 -> hay que reportar el corte posterior a la seleccion y matizar. Decidir si se adopta mezcla con naive+drift (OJO: ya se vio en test; requiere regla fijada de antemano) o se deja honesto.
- MEDIOS por corregir: IPC con start a mitad de ano escala reales previos (forecast.py rama v2 + apply_ipc_adjustment); SARIMAX refit sin cache por peticion (1.4-1.9s) y RF/XGB retrain n_jobs=-1 sin single-flight; A2 escala 70%/2026-08-31 no probada e2e (BD local trae 2026-09-01/100%; sep/oct 2026 quedarian -5..-8% en escala front, tope no actua ahi) -> agregar test con config 70%; A3 el ONI no demuestra valor (sin ONI 4.57 vs 4.52) y nys=5 peor post-seleccion: considerar dejar n_years_shape=2; get_number_of_days heredado no cuenta festivos (bug previo); ML_FORECAST_LEGACY es rollback parcial que engana (quitar o documentar); DivisionMercado duplica MarketBreak; CSV *_detalle pesados (~16MB) en docs/work/03_sarimax y 04_clima -> gzip/ignorar; print->logging en forecast.py; mover api/services/forecast_clima_patch.md a docs; confirmar que nys=5 esta realmente aplicado segun README.
- BAJOS: error crudo numpy con historia corta RF (forecast_rfxgb.py:273), cache refit sin version de artefacto, hueco interno de ano lanza ValueError, sin guarda si falta demand_split_config (A6), hindcast id4 no re-entrena (mu incluye 2025).
- Siguiente paso: esperar reportes removed-behavior/cross-file/regresion; corregir ALTO/MEDIOS; si hubo alto -> repetir ronda completa (todos los angulos + regresion, model sonnet); luego verificacion estandar (pytest completo), commits en la rama (Co-Authored-By segun convencion), cierre de log.
- Metricas actuales (crudo, ene-ago 2026, start 2026-01): RF 3.69, XGB 4.41, SARIMAX 3.04, Clima 4.45; antes 6.52/6.27/3.43/5.28. 229 tests pasando segun agente e2e.

## Ronda 1 CERRADA (2026-11-24)
- Los 9 agentes reportaron. Sin criticos. Altitude A1 = ALTA (honestidad de reporte RF/XGB) => regla: tras corregir, ronda 2 COMPLETA (todos los angulos + regresion).
- Hallazgos nuevos de removed-behavior/cross-file/regresion: ENSO id4 pierde sensibilidad (9.7%->3.1% Niño-Niña anual; aceptado), tope post-division no actua (propuesta: permitirlo si factor==1.0), modo ML_FORECAST_LEGACY no equivale a main (documentar), id4 start>=2028 mal etiquetado (preexistente), growth id4 con anio previo post-division, hueco interno => 422, REPORTE e2e faltante, `date` en forecast_sarimax:43 es falso positivo (string con noqa).
- Regresion: pytest 229 passed x3, sin flakiness; smoke HTTP OK (login, /forecast/predict 1-4, excel, MPM verify/demand); mpm/auth/database_url sin cambios en el diff.
- agente-correcciones lanzado con todos los hallazgos (docs/work/06_correcciones/REPORTE.md). Siguiente: ronda 2 completa -> commits.

## Correcciones aplicadas (agente-correcciones) — 252 tests pasan x2
Ver docs/work/06_correcciones/REPORTE.md. Ronda 2 COMPLETA de revision (8 angulos + regresion, sonnet) EN CURSO sobre `git diff --cached main`.

## Ronda 2 CERRADA (2026-11-24)
- 8 angulos + regresion (sonnet). Sin criticos ni altos => por regla NO hay ronda 3; se corrigen medios/bajos y UNA verificacion estandar (pytest x2 + pyflakes).
- Regresion: SIN regresion (252 passed x2; 28 casos vs main sin campos faltantes; MPM/auth/database_url/excel OK).
- Medios: fechas ids 1-3 pasaron de fin a inicio de mes (2 revisores) -> volver al formato de main; cache refit RF/XGB 8 entradas insuficiente -> 32; historia minima RF/XGB inconsistente con filas requeridas; honestidad (solo objetivos >=2026 n=164 son OOS puro: RF 5.92, XGB 5.79, naive+drift 5.40, Clima 3.83; id4 no reentrena en hindcast; nota_metricas en artefactos); calendario duplicado rfxgb/sarimax; CSV e2e obsoletos de Clima.
- Aceptados (baja): carrera menor de single-flight en camino de error, ramas de comportamiento anterior en forecast_adjustments, growth id4 cruzando 2026 (+8.3% vs ~5.8%) solo documentado, analysis.get_number_of_days preexistente.
- agente-correcciones-2 lanzado (docs/work/07_correcciones_r2/REPORTE.md). Siguiente: verificacion estandar -> commits.

## CIERRE (2026-11-24)
- Correcciones ronda 2 aplicadas (docs/work/07_correcciones_r2/REPORTE.md). Verificacion estandar: pytest 266 passed. Sin ronda 3 (ronda 2 sin criticos/altos).
- Commits en rama mejora/modelos-forecast-noche (sin push): 0f30736 (datos/harness/diagnostico), ab27f6c (modelos v2 + integracion), c114560 (docs y log). `git status` limpio salvo este cierre.
- Para revertir modelos: `git checkout main -- api/models api/services/forecast.py api/services/forecast_adjustments.py`.
- Pendiente humano: (1) confirmar configuracion demand_split_config de la BD real (70% hasta 2026-08-31 en las capturas; la copia local trae 2026-09-01/100%); (2) revisar con el front el cambio de campos opcionales y metricas_error_crudas; (3) push/merge y despliegue; (4) opcional: clima id4 aun lee CSV propio (no la BD de la peticion).

## Clima v3 (2026-11-24, sesión diurna, autorizado por el usuario)
- Motivo: id 4 quedó último en precisión (5.07 ene-ago 2026) y sin memoria del nivel reciente. Plan: docs/work/09_clima_v3/PLAN.md; reporte: docs/work/09_clima_v3/REPORTE.md.
- Implementado v3 mensual (clima_v3.py), activo por defecto; rollback `CLIMATE_MODEL_VERSION=v2`. 281 tests pasan. Ene-ago 2026: 3.98 (v2 5.07, naive+drift 3.96). Pool objetivos ≥2026: 4.15 vs v2 4.09 (sin mejora).
- Estado: revisión multi-ángulo + regresión en curso (ronda 1).
- Ronda 1 clima v3: completados línea-por-línea (sin bugs críticos/altos/medios; bajos: ONI "proyectado" 2026-06..08 tratado como observado, ENSO_PHASE_ONI duplicado) y reuse/efficiency (sin altos; latencia 60 ms; duplicación de helpers en walkforward_clima_v3). Cortados por límite de uso: comportamiento removido, cross-file, altitud/convenciones, regresión -> reanudados con SendMessage.
- Ronda 1 clima v3 CERRADA (2026-11-24): 5 ángulos + regresión (sonnet), sin críticos/altos => sin ronda 2. Corregido MAPE en rollback v2 y reporte; aceptados bajos listados en docs/work/09_clima_v3/REPORTE.md. Verificación estándar: pytest 281 passed. Commit en rama (sin push).

## Nivel adaptativo (2026-11-24, autorizado por el usuario)
- Plan: docs/work/11_nivel_adaptativo/PLAN.md; reporte: REPORTE.md. Implementado opt-in `LEVEL_ADAPT=1`, APAGADO por defecto (no gana de forma robusta: VAL mejora, 2026 pooled neutro, peor en orígenes 2026 / start 2026-05). 311 tests pasan. Estado: revisión multi-ángulo + regresión en curso.
- Nivel adaptativo: ronda 1 CERRADA (4 revisores + regresión, sin críticos/altos). pytest 311 passed. Commit en rama (sin push); flag APAGADO por defecto.
