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

**Pendiente:** los modelos RF/XGBoost mensuales usan `api/models/enso.csv` (clasificación anual
`-1` La Niña / `0` Neutral / `1` El Niño). Ese archivo no está versionado en `GMR_BackEnd`; sin él el servicio
usa `0` (Neutral) para todos los años y lo avisa en el log. Copiarlo desde el servidor del original.

Pruebas: `pytest tests` (modelo climático, post-procesamiento y motor MPM legacy).
