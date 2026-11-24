# DIAGNÓSTICO — línea base de los 4 modelos de forecast mensual (agente-0)

Fecha: 2026-11-23. Rama `mejora/modelos-forecast-noche`. Solo diagnóstico: no se modificó código de producción ni `.pkl`.
Reproducible con:
- `python docs/work/00_entorno/scripts/audit_datos.py` → `audit_demanda_mensual.csv`
- `python docs/work/00_entorno/scripts/baseline.py [--start 2026-01-01 --end 2027-12-31]` → `baseline_<tag>.csv` + `_resumen.csv`
  (corre `ForecastService.predict` real contra la BD local, "tope" = como lo sirve la API, "crudo" = con
  `cap_xgb_sarimax_error` y `apply_known_month_overrides` anulados por monkeypatch; añade baselines ingenuos).

Generados: `baseline_2026_2027*.csv`, `baseline_2025_2025*.csv` (hindcast), `baseline_riesgo_inicio_2027*.csv`.

---

## 0. Hallazgo principal: los ~415 GWh de RF/XGB en sep-oct 2026 son un artefacto, no del modelo

**Causa raíz (confirmada numéricamente):** división de mercado + tope de error, con el 70 % aplicado dos veces.

1. Desde **2026-09** la BD (`PRONOSTICO_demands`) ya guarda solo la porción del mercado nuevo (~70 %):
   sep-2026 = 587.2 GWh, oct-2026 = 576.9 GWh (yoy −17.6 % / −18.2 %), mientras may-ago 2026 crecían +10…+14 % yoy.
   En las capturas, la configuración de producción es "División de mercado · 70 % hasta 2026-08-31"
   (la BD local tiene `demand_split_config` = 2026-09-01 / 100 %).
2. El front (repo Node/React, no está en este repo) muestra: real ≤ 2026-08-31 × 0.70, real posterior sin escalar,
   y **todo el pronóstico × 0.70** (mismo criterio que `ForecastService.predict_excel`: "el modelo siempre extrapola
   en la escala completa"). Verificación: real ene-2026 740.6 × 0.7 = 518.4 (captura: 518); real sep 587 (captura 587).
3. Para RF/XGB, `cap_xgb_sarimax_error` reemplaza cualquier mes con |error| > 4 % por `real × (1 ± 1–4 %)`.
   En sep/oct el pronóstico crudo (~730–742 GWh, escala 100 %) se compara contra el real ya al 70 % (587/577),
   da +24/+29 % → el tope lo "pega" al real: RF sep = 596.9, oct = 590.6 GWh.
4. El front vuelve a multiplicar por 0.7: 596.9 × 0.7 = **417.8** y 590.6 × 0.7 = **413.4** → exactamente los ~415 GWh
   de las capturas (XGB: 596.7→417.7, 588.8→412.2).
5. SARIMAX y Climático no pasan por el tope → muestran crudo × 0.7 (SARIMAX sep 764.7 × 0.7 = 535 ≈ 536 de la captura),
   por eso "no colapsan" pero quedan ~30–40 GWh debajo del real de ene-ago (subestimación cruda real, ver §3).

Conclusión: arreglar el modelo no basta; hay que (a) tratar el quiebre estructural de mercado de 2026-09 en datos
(entrenamiento, lags, valores reales de comparación y tope) y (b) que el tope/métricas comparen en la misma escala.
Además, el tope **fabrica** el error mostrado (por diseño) y esconde que RF/XGB crudos tienen MAPE ~6.5 % en 2026.

---

## 1. Arquitectura de entrenamiento / predicción

Flujo común (`ForecastService.predict`, `api/services/forecast.py` ~1563–2298):
- Histórico mensual = `DemandService.get_demands(tipo=2)` = suma de `total` por mes, `file_type='txf'`,
  fechas hasta el día anterior a `start_date`. Anual = `tipo=3` (suma por año hasta `start_date.year-1`).
  **Unidad: kWh** (mes típico 7.4e8 kWh = 740 GWh; el comentario "Wh→MWh" en `error_absoluto_mwh` es erróneo: divide kWh/1e6 = GWh).
- `total` coincide con Σ hora_1..24 en todas las filas (0 discrepancias). `tipo_fecha` 0/1/2 = laboral/sábado/festivo-domingo (no se usa en forecast).
- Real para comparar: mismo `tipo=2` en la ventana pedida, solo meses con TODOS los días `txf`.
- Post-proceso: IPC por año (solo id 1/2, multiplicativo), `KNOWN_MONTH_OVERRIDES` (feb-2026 = valor real exacto
  643,488,767.72 → error 0 %), `cap_xgb_sarimax_error` (solo id 1/2, meses con real).

| id | Anual | Mensual | Datos de entrenamiento | Entrenado hasta | Script |
|---|---|---|---|---|---|
| 1 RF | Lasso(alpha=0.001) sobre log1p(total), StandardScaler; feats: 6 macro + lag_1..3 + growth_rate. **Coef. macro = 0** → solo lag_1 (1.044) y growth_rate (0.158): paseo aleatorio con drift amortiguado | RandomForest sobre `percentage` (share del mes en el año); feats mes, sin/cos, trend, pct_lag_12, pct_lag_1, dias_laborales, pct_mean_mes (media 3 años), dias_totales, enso | `../GMR/pronostico_gmr/gmr_annual_es.csv` (2011–2025, 15 filas) y `gmr_demanda_es.csv` (2011-01…2025-12) | Lasso: **solo 2014–2022** (el `best_estimator_` del GridSearch se ajusta con train sin los 3 años de test y nunca se reentrena); scaler sí ve hasta 2025. RF mensual: hasta **2024-12** (último año = test, no se reentrena) | `../GMR/pronostico_gmr/produccion/lasso_rf.py` |
| 2 XGB | **Idéntico a id 1** (mismos annual_model/scalers; totales anuales iguales 8.809/8.992 e9) | XGBRegressor mismo set de feats | igual | igual | `.../produccion/lasso_xgb.py` |
| 3 SARIMAX | — | SARIMAX(5,1,6)(1,1,1,12) sobre log1p(total) con 6 macro anuales (RobustScaler) como exógenas | `gmr_demanda_es.csv` + `gmr_annual_es.csv`, 2011-01…2025-12 (180 obs) | **2025-12** (orden elegido por "MAPE de test" 2025 que es in-sample: el modelo final se ajusta con todo y `predict(start=train_size)` es one-step in-sample → MAPE 1.6 % no es out-of-sample) | `.../produccion/sarimax.py` |
| 4 Clima+ONI | Ridge(alpha=1) sobre log1p(total): temp/humedad/viento/lluvia anuales, oni_mean_anio, lag_1, lag_2, year_index, year_index_norm, growth_rate_lag1 (13 muestras 2013–2025) | **No usa el XGB mensual serializado**: share fijo = promedio de los 2 últimos años completos; ONI futuro con `project_oni_future` (decay calibrado), clima futuro = climatología 10 años | `training/forecast_climate/artifacts/dataset_mensual.csv` / `dataset_anual.csv` (hasta 2026-04; **no lee la BD**) | Ridge con todo hasta 2025 | `../GMR/GMR_production/GMR_BackEnd/training/forecast_climate/{build_dataset,train_candidates,evaluate,serialize_final_model}.py` (no copiados a este repo) |

**No existe script de reentrenamiento en `gmr_en_sphere`**. Los originales están en `../GMR/pronostico_gmr/produccion/`
(RF/XGB/SARIMAX, leen CSV exportados a mano, no la BD) y en `../GMR/GMR_production/GMR_BackEnd/training/forecast_climate/` (id 4).

### Fallos de inferencia (independientes del quiebre de mercado)
- **`enso.csv` ausente** en `api/models/` → `_get_enso` devuelve 0 (Neutral) para todo año; en entrenamiento 2024–2026 = −1.
  Impacto pequeño (shares cambian ≤ ±0.0015, ~1–2 % por mes). Además el CSV original tiene columna `año` pero el código lee `ano` → aunque se copie, fallaría (cae a `{}` en silencio).
- **SARIMAX ignora `start_date`**: `model.forecast(steps)` siempre arranca en 2026-01 (fin del ajuste) y luego se rotula
  desde `last_date+1`. Con start=2025-01 devuelve exactamente los valores de 2026 rotulados como 2025; con start=2027
  devuelve 2026 desplazado y el anual 2027 = 1.43e9 (basura). Solo es correcto si start = 2026-01-01.
  Tampoco aplica `remove_atypical_years`/`exog` históricos al modelo (calcula `y_log`/`exog_scaled` y no los usa) y la macro futura es constante.
- **RF/XGB con año parcial**: con start=2027-01 el histórico mensual termina en 2026-10 (nov es `tx2`, dic no existe) →
  lag_1 anual = total parcial 2026 (6.79e9, además con 2 meses al 70 %) → 2027 = 6.31e9 (−27 % vs nivel real) y las fechas
  se rotulan desde 2026-11 (2027 sale con 10 meses, nov-dic vacíos). El guard de "caída > 30 %" no salta.
- RF/XGB: `trend` en inferencia = 180+ (fuera del rango de entrenamiento; árboles no extrapolan → inocuo pero inútil).
- Macro de la BD (`PRONOSTICO_macroeconomics_data`) ≠ macro de entrenamiento (`gmr_annual_es.csv`): p. ej. crecimiento
  población 2023 BD 0.41 vs CSV 1.12; importaciones 2025 5.05e10 vs 7.0e10. Hoy no afecta (Lasso anuló la macro) pero
  es un desajuste train/serve.
- Id 4: features anuales de clima/ONI del **mismo año objetivo** (observadas en entrenamiento, climatología/proyección en
  inferencia) → mismatch train/serve; Ridge entrenado con 2025 → su hindcast 2025 (MAPE 1.48 %) es in-sample.

---

## 2. Hallazgos de datos (BD local restaurada de `afinia2.backup`)

Ver `audit_demanda_mensual.csv`. 2011-01 … 2026-10 `txf` completo salvo:

| Mes | Problema | Efecto |
|---|---|---|
| 2020-10, 2020-11, 2020-12 | **Filas duplicadas por día** (2 filas/fecha): una a escala correcta (~2e7 kWh/día) y otra ~1/10 (~2e6, ids 3654–3745). La suma mensual infla ~10 %: 676/655/695 GWh en vez de 615/597/631 (dedup por día tomando la fila mayor) | Contamina `gmr_demanda_es.csv`, SARIMAX (endog 2020-Q4 inflado), shares de 2020 y el total 2020 de `SphaerAI_yearly_demand` (6.99e9). En `dataset_mensual.csv` del id 4 el 2020-Q4 está al revés (306/220/356 GWh, deflactado) → 2020 anual 5.85e9 |
| 2021-09 | Faltan 2021-09-01 y 09-02 (28/30 días) → 575 GWh (−7 %) | Outlier bajo en entrenamiento; el validador de reales lo descartaría |
| 2026-09, 2026-10 | **Quiebre estructural**: división de mercado, BD ≈ 70 % del mercado previo (587/577 GWh) | Ver §0. Todo modelo entrenado con la serie cruda verá una caída de ~30 % |
| 2026-11 | Solo 1–17 y `file_type='tx2'` (preliminar) → excluido como real (correcto); el histórico `tipo=2` también lo excluye | Cualquier forecast con start ≥ 2026-12 tiene año 2026 incompleto |
| 2020, 2021 | Marcados `ATIPICO` en `SphaerAI_type_year` (solo SARIMAX los quita del df, sin efecto real en el modelo ya ajustado) | — |

Otros: 0 nulos; `total` = Σ horas; sin huecos fuera de los citados. `SphaerAI_monthly_demand.percentage` de 2026
se calcula contra un año parcial (0.109–0.125) → no usar como feature. Crecimiento yoy 2026: ene +4.3 %, feb −0.7 %,
mar +3.8 %, abr +5.4 %, **may–ago +10.4…+13.6 %** (escalón a partir de mayo, no anticipable al 1-ene con info anual).

---

## 3. Línea base numérica (código real, BD local, escala 100 % BD, kWh)

Ventana 2026-01-01..2027-12-31 (`baseline_2026_2027_resumen.csv`). "pre" = 8 meses ene–ago 2026 (horizonte 1–8 desde
dic-2025). "post" = sep–oct 2026 (real ya al 70 %: errores no interpretables como calidad del modelo).

| Serie | MAPE pre | Sesgo pre | Máx |err| pre | Err. total ene–ago | MAPE post (quiebre) |
|---|---|---|---|---|---|
| RF crudo | **6.52 %** | −5.36 % | 10.6 % (ago) | −5.73 % | 26.5 % |
| RF con tope | 2.24 % | −2.24 % | 3.8 % | −2.27 % | 2.0 % (fabricado) |
| XGB crudo | **6.27 %** | −5.32 % | 10.0 % (ago) | −5.65 % | 26.4 % |
| XGB con tope | 1.64 % | −1.64 % | 3.9 % | −1.71 % | 1.9 % (fabricado) |
| SARIMAX crudo | **3.43 %** | −1.29 % | 7.5 % (feb) | −1.60 % | 32.4 % |
| SARIMAX con override feb | 2.50 % | −2.23 % | 6.1 % | −2.37 % | 32.4 % |
| Clima+ONI crudo | **5.28 %** | −3.08 % | 9.8 % (ago) | −3.48 % | 26.4 % |
| Clima+ONI con override feb | 4.48 % | −3.88 % | 9.8 % | −4.14 % | 26.4 % |
| Naive estacional (mismo mes 2025) | 6.97 % | −6.80 % | 12.0 % | −7.11 % | 21.8 % |
| Naive estacional × drift 12m | 5.47 % | −4.75 % | 10.0 % | −5.07 % | 24.5 % |

Totales anuales crudos: RF/XGB 2026 = 8.809e9, 2027 = 8.992e9 (+2.1 %/+2.1 %); SARIMAX 9.197e9 / 9.787e9;
Clima 8.919e9 / 9.287e9. Real 2025 = 8.625e9.

Lectura: **RF y XGB crudos son peores que el naive estacional con drift** (6.5/6.3 % vs 5.5 %); SARIMAX es el único
que le gana claramente. Todos subestiman (sesgo −1…−5 %), sobre todo may–ago (escalón de demanda +10–14 % yoy).
El error de RF/XGB viene casi todo del nivel anual (Lasso = lag_1 + drift pequeño → +2.1 % en 2026 vs ~+7.6 % real ene-ago),
no del reparto mensual.

Hindcast 2025 (`baseline_2025_2025_resumen.csv`, start=2025-01-01): RF crudo 3.75 % (sesgo +3.75), XGB 3.73 %,
Clima 1.48 % (**in-sample**, Ridge vio 2025), SARIMAX 6.63 % (**inválido**: devuelve los valores de 2026 rotulados 2025),
naive estacional 3.02 %, naive+drift 4.58 %. Nota: el `cap` en 2025 baja RF a 2.83 % solo por maquillaje.

Riesgo start=2027-01 (`baseline_riesgo_inicio_2027.csv`): RF/XGB ~510–550 GWh/mes (colapso por año parcial + quiebre),
SARIMAX con fechas corridas, Clima no cambia (no lee BD).

---

## 4. Propuesta priorizada de mejoras ML (sin fuga, sin hardcode, validación walk-forward)

### Transversal (hacer primero; lo usan todos)
1. **Capa de datos limpia única** (`training/common/data.py` o similar): leer `PRONOSTICO_demands` de la BD;
   deduplicar por fecha (quedarse con la fila de escala correcta en 2020-Q4); marcar meses incompletos (2021-09 →
   imputar por promedio diario×días o excluir del target), excluir `tx2`/meses parciales del target.
2. **Quiebre de mercado como dato, no como sorpresa**: leer `demand_split_config` (fecha_limite, porcentaje) y entrenar/
   predecir en **escala "mercado completo"**: real_post / (porcentaje/100) es una estimación inestable con 2 meses; mejor
   alternativa: entrenar con la serie ≤ 2026-08 y para meses > fecha_limite comparar `pred × porcentaje` vs real. Exponer la
   escala explícitamente en la respuesta para que el front no aplique el 70 % dos veces (coordinar con agente-e2e).
   Mientras tanto el tope debe comparar en la misma escala (pred × factor vs real) o no aplicarse post-corte.
3. **Harness walk-forward común** (expanding window, origen mensual o anual, horizonte 1–12 y 13–24): orígenes
   2019-12 … 2025-12 (excluyendo 2020-2021 de la evaluación, no necesariamente del train), métricas MAPE mensual, sesgo,
   error anual, siempre contra **naive estacional** y **naive+drift**. Reutilizar `baseline.py` solo como "prueba e2e".
   Prohibido elegir hiperparámetros mirando 2026.
4. Scripts de reentrenamiento versionados en este repo que lean la BD (hoy no existen) + `metrics.pkl` con métricas
   walk-forward reales (las actuales son engañosas: 0.29 % es MAPE sobre log escalado; SARIMAX 1.6 % es in-sample).

### id 1 RF / id 2 XGB
1. Reemplazar el Lasso anual (que es lag_1+drift, entrenado hasta 2022 y nunca reentrenado) por un modelo de nivel
   honesto: p. ej. log-total ~ tendencia amortiguada (Holt/ETS damped) o regresión sobre crecimiento con media de los
   últimos k años; o mejor, **pronóstico mensual directo** (target = log demanda mensual o yoy) con lags ≥ h (direct
   multi-horizon) en vez de anual × share. Validar contra naive+drift.
2. Reparto mensual: el share aprendido ya es razonable; quitar `trend` (fuera de rango), arreglar `enso` (copiar
   `enso.csv` con columna `ano` o leer `año`, y para años futuros usar proyección ONI disponible al origen, no la clasificación
   ex-post del año), `pct_lag_1` usa el último mes del año anterior para los 12 meses (inconsistente con el entrenamiento,
   donde es el mes previo real) → reemplazar por features que existan en inferencia.
3. Reentrenar con datos hasta el último mes completo, refit final con todo tras elegir hiperparámetros por walk-forward.
4. Robustez: si el último año del histórico está incompleto, anualizar/usar 12 meses móviles para lags y rotular las
   fechas desde enero del primer año pedido (no desde `last_date+1`).
5. Diferenciar realmente RF de XGB (hoy comparten el modelo anual).

### id 3 SARIMAX
1. **Bug crítico**: re-anclar el estado con los datos hasta `start_date` (`res.apply(endog_nuevo, exog_nuevo)` o
   `res.append`) y luego `forecast`; rotular fechas desde start. Sin esto solo funciona para start=2026-01.
2. Reducir el orden (5,1,6)(1,1,1,12) → buscar órdenes parsimoniosos por AIC/BIC en train y elegir por walk-forward
   out-of-sample (no por MAPE in-sample). Probar sin exógenas macro (constantes a futuro no aportan) y con
   dummies 2020 (COVID) / intervención de mercado 2026-09.
3. Limpiar 2020-Q4 duplicado antes de ajustar (hoy endog 2020-10..12 inflado ~10 %).

### id 4 Clima + ONI
1. Hacer que lea la demanda de la BD (o regenerar `dataset_*.csv` con la capa limpia) y corregir 2020-Q4 deflactado.
2. Features anuales del año objetivo: entrenar con los **mismos** insumos que habrá en inferencia (climatología y ONI
   proyectado al origen), o usar solo lags/ONI del origen → elimina el mismatch train/serve.
3. Ridge con 13 muestras: preferir modelo de crecimiento con pocos parámetros; evaluar walk-forward 2017–2025.
4. El share fijo de 2 años es honesto y estable; evaluar n_years_shape por walk-forward (2 vs 3 vs 5).

### Ajustes (tope / override)
- Mantener pero reportar siempre crudo; el tope debe ser no-op cuando el modelo está bien. Aplicarlo en la misma escala de
  mercado y nunca a meses post-corte sin factor. `KNOWN_MONTH_OVERRIDES` fija feb-2026 al valor real exacto (error 0 %):
  documentarlo como maquillaje y excluirlo de cualquier métrica.
