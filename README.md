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

No se portó lo que en el original solo existe para la demostración: `LOCAL_DEMO_MODE` y sus datos locales,
los topes artificiales del error mensual (1–4 %), el valor fijo forzado de feb-2026 y la macro constante de
SARIMAX "como la demo".

**Pendiente:** los modelos RF/XGBoost mensuales usan `api/models/enso.csv` (clasificación anual
`-1` La Niña / `0` Neutral / `1` El Niño). Ese archivo no está versionado en `GMR_BackEnd`; sin él el servicio
usa `0` (Neutral) para todos los años y lo avisa en el log. Copiarlo desde el servidor del original.

Pruebas: `pytest tests` (modelo climático, post-procesamiento y motor MPM legacy).
