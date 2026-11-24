# Pronóstico ML Service

Microservicio **FastAPI** que expone únicamente el cómputo de Machine Learning que
consume el backend Node (`pronostico_backend`):

- `POST /v1/autenticacion/login` — login de cuenta de servicio (devuelve JWT).
- `/v1/forecast/*` — predicción de demanda (usado por el módulo **Demanda Pronóstico**).
- `/v1/mpm/*` — predicción MPM (usado por el módulo **MPM**).

Extraído de `GMR_BackEnd`, dejando fuera todo lo que el backend Node ya reemplaza
(usuarios, roles, sesiones, convocatoria, cubrimiento, valoración, análisis, etc.).

## Diferencias clave vs. GMR original

1. **Base de datos → Postgres de `pronostico`** (no Oracle). Lee/escribe las mismas
   tablas que el backend Node: `PRONOSTICO_demands`, `SphaerAI_monthly_demand`,
   `SphaerAI_yearly_demand`, `SphaerAI_type_year`, `SphaerAI_users_models`,
   `SphaerAI_users_models_values` y `datos_clima`.
2. **Auth simplificada**: login de cuenta de servicio con `ML_SERVICE_USER` /
   `ML_SERVICE_PASSWORD` (sin LDAP ni tabla de usuarios). El JWT se firma con
   `JWT_SECRET_KEY` y los routers lo validan con `get_current_user`.
3. **No crea tablas** (`create_all` removido): el esquema lo administra el backend Node.

## Requisitos

- Python 3.11+
- El Postgres de `pronostico` accesible (mismas credenciales del backend).

## Puesta en marcha

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # y completa los valores
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

El backend Node debe apuntar a este servicio (ya lo hace por defecto):
`DEMANDA_ML_PORT=8000`, y `DEMANDA_ML_USER` / `DEMANDA_ML_PASSWORD` deben coincidir
con `ML_SERVICE_USER` / `ML_SERVICE_PASSWORD` de este `.env`.

## Base de datos por petición (multi-empresa)

Una sola instancia del ML atiende a todas las empresas. Cada petición del backend trae el `database_url` de la empresa con
la que inició sesión el usuario, y el ML consulta ESA base:

- POST: campo `database_url` del cuerpo JSON. GET: parámetro `?database_url=...`. También se acepta el encabezado `X-Database-Url`
  (tiene prioridad sobre los otros dos).
- Hay un motor de conexión por base (en caché, máximo `ML_MAX_DB_ENGINES`, 32 por defecto).
- Si la petición no trae `database_url`, se usa la base del `.env` (`ML_DATABASE_URL` o `POSTGRES_*`): útil en desarrollo.

Variables opcionales (recomendadas en producción):

| Variable | Efecto |
|---|---|
| `ML_REQUIRE_DATABASE_URL=1` | Rechaza (400) las peticiones sin `database_url`, en vez de caer en la base del `.env`. |
| `ML_ALLOWED_DB_HOSTS=host:puerto,host:puerto` | Solo acepta bases en esos servidores (por ejemplo `gacademic.jmdatalabs.co:5432,localhost:5432`). |

Los errores de `database_url` no incluyen la URL (trae usuario y contraseña).

## Pendientes / caveats

- **`GMR_last_demand_document`** y **`GMR_session`**: estas dos tablas NO existen en el
  Postgres de `pronostico`. `mpm.py` lee `GMR_last_demand_document` (verificación de
  documentos / última fecha). Si esa ruta se usa, hay que: (a) crear la tabla en
  Postgres, o (b) adaptar el servicio para derivar la última fecha de
  `PRONOSTICO_demands` (`MAX(fecha)`). El `predict` de MPM, que sí funcionó en pruebas,
  no depende de esa tabla.
- `requirements.txt` conserva libs que ya no se usan tras el recorte (`oracledb`,
  `ldap3`, etc.); se pueden depurar más adelante sin prisa.

## Actualización de modelos desde GMR_BackEnd

Los modelos y el motor de predicción se sincronizaron con `GMR_BackEnd` (rama `main`):

- **Modelos reentrenados**: `rf_model`, `xgb_model` (mensual, ahora con las variables `dias_totales` y `enso`),
  `xgboost_only`, `xgboost_multimodelo` y `xgboost_varias_multimodelo`.
- **Modelo climático + ONI (id=4)**: `api/models/climate_model/` y el soporte en `training/forecast_climate/`
  (`features.py`, `postprocess_calibration.py` y los datasets históricos de `artifacts/`).
- **MPM**: el motor de paridad con el sistema anterior (`api/services/mpm_legacy/`) reemplaza al XGBoost
  en `mpm.py`.
- **Forecast**: sensibilidad a IPC por año (`ipc_growth_by_year`) y escenario ENSO mes a mes
  (`enso_phase_by_month`), correcciones del modelo climático y del modelo anual.

No se portó lo que en el original solo existe para la demostración: `LOCAL_DEMO_MODE` y sus datos locales
y la macro constante de SARIMAX "como la demo".

Sí se mantienen, para que RF/XGBoost den igual que el original (GMR Atenea), el tope del error mensual (1–4 %,
`cap_xgb_sarimax_error`, solo RF/XGBoost y solo meses con dato real) y el valor fijo de feb-2026
(`KNOWN_MONTH_OVERRIDES`), ambos en `api/services/forecast_adjustments.py`.

**Pendiente (solo camino legacy, `ML_FORECAST_LEGACY=1`; los modelos v2 no usan ENSO):** los modelos RF/XGBoost mensuales usan `api/models/enso.csv` (clasificación anual
`-1` La Niña / `0` Neutral / `1` El Niño). Ese archivo no está versionado en `GMR_BackEnd`; sin él el servicio
usa `0` (Neutral) para todos los años y lo avisa en el log. Copiarlo desde el servidor del original.

Pruebas: `pytest tests` (modelo climático, post-procesamiento, motor MPM legacy y, desde 2026-11, los modelos
v2: `tests/common`, `tests/forecast_rfxgb`, `tests/forecast_sarimax`, `tests/forecast_e2e`). Los tests marcados
`integration_db` necesitan la BD local (`ML_DATABASE_URL`) y **se omiten solos** si no está disponible.

## Modelos de demanda mensual v2 (2026-11)

Detalle y reproducción en `docs/work/` (PLAN, DIAGNOSTICO y REPORTE de cada parte).

**Qué cambió**
- **Historia limpia por petición** (`api/services/forecast_history.py`): se lee `PRONOSTICO_demands` con la
  sesión de la petición (`database_url`), se deduplica 2020-Q4 (filas repetidas a ~1/10), se imputa 2021-09
  (2 días faltantes) y solo se usan meses completos `txf`. Se corta en la **división de mercado**
  (`demand_split_config`): los meses post-`fecha_limite` están en otra escala y no son historia.
  Reutiliza `training/common` (la imagen Docker copia `training/`).
- **id 1 RF / id 2 XGBoost** (`forecast_rfxgb.py`, artefactos `api/models/{rf,xgb}_model/v2/`): pronóstico
  directo de log-demanda mensual (nivel 12m × índice estacional × drift + corrección ML con calendario
  colombiano). Ya no usan Lasso anual ni variables macro. En hindcast (artefacto entrenado con meses >= inicio)
  se re-entrenan al vuelo con la historia previa, para que el error mostrado sea fuera de muestra (caché LRU con
  single-flight por historia + huella del artefacto, 2 núcleos por re-entreno y como mucho 2 re-entrenos
  simultáneos; `api/services/forecast_cache.py`; hasta 32 re-entrenos cacheados, ~4 MB cada uno en RF y ~0.3 MB
  en XGB). **No baten al naive+drift fuera de la muestra de selección** (ver métricas abajo).
- **id 3 SARIMAX** (`forecast_sarimax.py`, `sarimax_v2.json`): SARIMA(1,0,0)(0,1,1,12)+c sobre log demanda
  diaria media. **Respeta `start_date`** (antes siempre pronosticaba desde 2026-01). En hindcast re-estima los
  parámetros con la historia previa (cacheado por petición repetida, misma caché).
- **id 4 Clima+ONI**: artefactos v2 (crecimiento log + β·ONI) y reparto mensual con los últimos **2 años**
  completos (contrato original; el parche `n_years_shape=5` se probó y se retiró: empata en el walk-forward total
  y es peor en los orígenes posteriores a la selección, ver `docs/work/04_clima/forecast_clima_patch.md`).
  Honestamente es **"deriva + sensibilidad ONI no validada"**: sin ONI el walk-forward da 4.57 % vs 4.52 % con ONI
  (diferencia no significativa). Sigue leyendo su dataset de `training/forecast_climate/artifacts/` (no la BD de la
  petición). Solo pronostica desde el año siguiente al último de su histórico (hoy 2027): un `start_date` en 2028
  devuelve un error explícito en vez de años mal etiquetados.
- **Pérdida de sensibilidad ENSO (aceptada)**: con el modelo v2 la diferencia anual entre los escenarios
  "todo NIÑO" y "todo NIÑA" baja de ~9.7 % (modelo original) a ~3.1 %. El β del ONI se estimó con los datos y la
  validación no muestra que el ONI aporte; el orden NIÑA < NORMAL < NIÑO se conserva.
- **Escala**: `valores_pronostico` está siempre en escala de **mercado completo**; el consumidor (front /
  `predict_excel`) lo multiplica una sola vez por `porcentaje/100`.
- **Tope (`cap_xgb_sarimax_error`, id 1/2) y override feb-2026 se mantienen**, pero: el tope solo actúa en meses
  pre-división con real (misma escala que compara el front) y **nunca en meses post-división con factor ≠ 1**
  (antes comparaba pronóstico al 100 % contra real al 70 % y producía los ~415 GWh de sep/oct-2026); con factor
  = 1.0 (porcentaje 100 %) la escala no cambia y también actúa en post-división; el override solo aplica si el
  real de la petición coincide con el valor documentado (no se impone a otras bases).
- Los errores (`errores_porcentuales`, `metricas_error`) se calculan en la escala del front:
  pre-división `(pred−real)/real`; post-división `(pred·f−real)/real`.
- Robustez: cualquier `start_date`/`end_date` (mitad de mes, 1 mes, 2028 en id 1/2/3), sin variables macro, JSON sin NaN/inf,
  real mensual siempre con el mes completo.
- Rollback **parcial**: `ML_FORECAST_LEGACY=1` revierte SOLO la inferencia de id 1/2/3 (Lasso anual + RF/XGB y
  SARIMAX(5,1,6) serializados, con variables macro; los `.pkl` legacy siguen en su carpeta). NO revierte el
  post-proceso (tope solo en meses elegibles, override solo con real coincidente, errores en la escala del front,
  real con el mes completo), los campos nuevos ni el id 4 v2: la respuesta **no es equivalente a `main`**. Para
  volver exactamente al comportamiento anterior hay que desplegar `main`. Los artefactos originales **no** están
  versionados en esta rama (la copia local `docs/work/backup_modelos_originales/` está en `.gitignore`); para
  recuperarlos: `git show main:api/models/climate_model/annual_model.pkl > ...` o
  `git checkout main -- api/models/climate_model/` (ídem `api/models/{rf,xgb,sarimax}_model/`).
- **Inicio mínimo por modelo** (la BD empieza en 2011-01): id 1/2 desde **2015-11** (~2016): en hindcast se
  re-entrenan y `fit_model` exige >= 58 meses de historia (50 filas origen×horizonte con `h_train=36`; para
  inferir con el artefacto bastan 48); id 3 desde **2014-01** (>= 36 meses); id 4 según su dataset climático.
  Antes de eso la API responde `historia insuficiente: se requieren >= N meses`.
- Calendario de Colombia único (`api/services/calendar_co.py`, `holidays_co`): features de RF/XGB y SARIMAX y
  `get_monthly_info`/`get_day_behavior`; un test fija sus valores 2010-2030 (los artefactos dependen de ellos).

**Contrato**: no se quitó ni renombró ningún campo. Campos opcionales nuevos: por mes
`valores_pronostico_crudos`, `errores_porcentuales_crudos`, `tope_aplicado`, `override_aplicado`,
`post_division`, `factor_division`; global `metricas_error_crudas`, `escala_pronostico`, `division_mercado`,
`modelo`. Cambios de **valores** (mismo formato) respecto a `main`:
- `monthly_predictions.fechas`: igual que `main` — **fin de mes** en ids 1-3 (`2026-01-31`) e **inicio de mes**
  en id 4 (`2026-01-01`); test de regresión `test_fechas_formato_main`. Un rango de un solo mes o a mitad de mes
  (p. ej. `2026-06-15..2026-06-20`) ahora responde en vez de fallar.
- `get_monthly_info` / `get_day_behavior`: los días laborables y domingos+festivos **ahora cuentan los festivos**
  (antes nunca los contaba). Enero 2026: 20 laborables / 5 sábados / 6 domingos+festivos.
- `errores_porcentuales` / `metricas_error`: en la escala del front (post-división `(pred·f−real)/real`).
- ids 1-2 tienen un inicio mínimo (2015-11, ver arriba); antes respondían con cualquier fecha.
- id 4: en rangos que cruzan 2026 (p. ej. `2026-01..2027-12`), el `growth` de 2027 puede verse algo alto
  (~+8.3 % vs ~+5.8 % implícito): 2026 sale del hindcast anclado en 2025 y 2027 del pronóstico anclado en el
  último año real (empalme); no se corrigió, solo se documenta.

**Métricas honestas (MAPE mensual crudo, sin tope ni override)**

Las specs se eligieron con orígenes <= 2023-12 **y objetivos <= 2025-12**. Para los orígenes posteriores
(2024-01..2025-12, h 1-24, solo objetivos pre-división <= 2026-08) hay que separar:
- objetivos <= 2025-12 (n=276): son meses que **ya se vieron** como objetivo durante la selección (no es fuera de
  muestra pura);
- objetivos >= 2026-01 (n=164): **la única ventana fuera de muestra pura** (nadie los vio al elegir).

| Modelo | WF total 2017-12..2025-12 (h 1-24; selección) | Post-sel. objetivos <= 2025-12 (n=276) | **Objetivos >= 2026 (n=164, OOS puro)** | Post-sel. total (n=440) | Test ene-ago 2026 (origen 2025-12) antes → ahora | Con tope+override |
|---|---|---|---|---|---|---|
| RF | 5.56 % | 6.67 % | **5.92 %** | 6.39 % | 6.52 → 3.69 % | 1.36 % |
| XGB | 5.52 % | 6.29 % | **5.79 %** | 6.11 % | 6.27 → 4.41 % | 1.83 % |
| SARIMAX | 4.07 % | 4.19 % | **4.03 %** | 4.13 % | 3.43 → 3.04 % | 2.29 % (override feb) |
| Clima+ONI v2 (nys=2) | 4.52 % (97 orígenes) / 4.55 % (orígenes dic) | 4.06 % | **3.83 %** | 3.97 % | 5.28 → 5.07 % | 4.01 % (override feb) |
| Clima sin ONI (mismo spec) | 4.57 % | 3.90 % | 3.93 % | 3.91 % | — | — |
| Clima receta original reentrenada | — | 2.72 % | 4.88 % | 3.52 % | — | — |
| naive estacional + drift | 6.13 % | 4.24 % | **5.40 %** | 4.67 % | 3.96 % | — |

Todas las columnas post-selección usan la misma ventana (SARIMAX recalculado con el mismo protocolo: la columna
n=276 coincide con su `report_dev`, 4.19 %; antes la tabla mezclaba n=276 para SARIMAX con n=440 para el resto).
Para el id 4 el walk-forward reentrena por origen, pero **la API no reentrena el id 4 en hindcast**: con
`start_date` <= 2025 usa coeficientes ajustados con datos hasta 2025 (parcialmente in-sample). El error que muestra
la API en hindcast es fuera de muestra solo para ids 1-3.

- **RF/XGB no baten al naive+drift fuera de la muestra de selección**: ni en objetivos <= 2025-12 (6.67/6.29 vs
  4.24 %) ni en los >= 2026 (5.92/5.79 vs 5.40 %). El titular 5.56/5.52 vs 6.13 % lo ganan con orígenes <= 2023,
  que son los que se usaron para elegirlos. Causa: **sobreajuste de nivel tras el salto de 2024** — las features
  de crecimiento al origen (g12, g3, drift) y la corrección ML extrapolan el salto de nivel de 2024 hacia 2025 y
  sobreestiman (sesgo +5..+6 %; pronosticar 2025 desde 2024-12 da 7.5-7.8 % vs 3.05 % del naive+drift). No se
  adopta la mezcla 50/50 con naive+drift: se calculó después de ver el test y sería tunear contra él.
- SARIMAX es el más consistente: empata/gana levemente al naive+drift en objetivos <= 2025 (4.19 vs 4.24) y le
  gana en >= 2026 (4.03 vs 5.40).
- Clima+ONI: el ONI no demuestra valor (sin ONI 3.90/3.93 vs 4.06/3.83). La v2 **no mejora a la receta original
  fuera de la selección salvo en objetivos >= 2026** (3.83 vs 4.88; en <= 2025 la original da 2.72 vs 4.06): con
  una grilla de 288 specs hay riesgo de sobreajuste de la selección; es deriva + una sensibilidad ONI no validada.
- **El único test 2026 es un solo origen (2025-12) y 8 meses**: no basta para afirmar que un modelo es mejor que
  otro; las diferencias de 2026 (3.0-5.1 %) están dentro del ruido de un origen.
- La columna "lo que se muestra" está maquillada por el tope (1-4 %): mirar siempre `metricas_error_crudas`.

**Caveats**
- En ene-ago 2026 el naive+drift (3.96 %) sigue siendo comparable: el salto de demanda de may-ago 2026
  (+10–14 % interanual) no era anticipable; todos los modelos lo subestiman (sesgo −1 a −2 %).
- Hindcast 2025 de RF/XGB crudo ≈ 7.5–7.8 % (arrastran el salto de 2024); el tope actúa en 11 de 12 meses ahí y
  en 4 de 8 meses de 2026. El tope sigue siendo maquillaje: mirar siempre `metricas_error_crudas`.
- Meses post-división (sep-oct 2026): el error depende del `porcentaje` configurado. Con 70 % (p. ej.
  `fecha_limite` 2026-08-31) salen ≈ −5…−9 % en la escala del front y el tope no los toca (escala distinta). Con
  100 % (BD local) el pronóstico y el real están en la misma escala para el front, así que el tope **sí** actúa
  (el error crudo es +28…+35 % porque los datos implican ≈ 74 %; ver `errores_porcentuales_crudos`).
- IPC (`ipc_growth_by_year`, id 1/2) con `start_date` a mitad de año: el factor solo escala los meses
  pronosticados (>= start); los reales previos del año no se tocan y la suma mensual cuadra con el total anual.
- Un mes sin real usable dentro del año del último real (hueco interno) se rellena con el pronóstico del modelo
  hecho desde el primer hueco (`modelo.meses_hueco_rellenados`).
- `growth` del año previo a `start_date` cuando está incompleto en la BD se estima con reales + pronóstico
  (`modelo.anios_completados_con_pronostico`); `annual_predictions` histórico sigue saliendo de la BD.
