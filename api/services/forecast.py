import joblib
from ..config.schemas import ForecastSchema, NewUserModelSchema
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
import pandas as pd
import numpy as np
from ..config.schemas import DemandSearchSchema, CreateOrUpdateUsersModelsSchema, MonthlyInfoSchema, DayBehaviorSchema, CreateTypeYearListSchema, UpdateTypeYearListSchema, UpdateMonthlyTypeSchema, SearchAllMacroeconomicSchema, ForecastTypeMonthSchema, ForecastModelSaveBasedOnYear
from sklearn.preprocessing import StandardScaler
from .analysis import DemandService, MacroeconomicService
from sklearn.impute import SimpleImputer
import joblib
import warnings
from ..other_models.demand import MonthlyDemand, Demand, YearlyDemand, TypeYear
from ..other_models.session import UsersModels, UsersModelsValues
import calendar
from holidays_co import get_colombia_holidays_by_year
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
import json
import io
from fastapi import HTTPException
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
import logging
import os
from ..other_models.demand import MonthlyDemand, Demand, YearlyDemand, TypeYear
from api.services import sarimax_macro_constante
from api.services.demand_config_service import DemandConfigService, adjust_period_value
from api.services.forecast_adjustments import (
    apply_ipc_adjustment,
    apply_known_month_overrides,
    cap_xgb_sarimax_error,
    front_scale_pct_error,
)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _use_legacy_models() -> bool:
    """Rollback PARCIAL de emergencia: ``ML_FORECAST_LEGACY=1`` vuelve solo a la **inferencia** anterior de
    id 1/2/3 (Lasso anual + RF/XGB y SARIMAX(5,1,6) serializados, con sus variables macro).

    NO revierte el resto de cambios de la rama v2: el post-proceso (tope solo en meses elegibles, override solo
    con real coincidente, errores en la escala del front, real mensual con el mes completo), los campos nuevos
    de la respuesta y el id 4 (artefactos v2) siguen siendo los nuevos. La respuesta NO es equivalente a la de
    ``main``; para volver exactamente a ``main`` hay que desplegar ese commit. Por defecto se usan los v2."""
    return os.getenv("ML_FORECAST_LEGACY", "").strip().lower() in ("1", "true", "si", "sí")


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _month_ts(value) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.max.to_period("M").to_timestamp()
    return pd.Timestamp(value).to_period("M").to_timestamp()


def contract_month_labels(fechas: pd.Series, model_id: int) -> list:
    """Etiquetas ``monthly_predictions.fechas`` con el formato de ``main`` (contrato del front).

    ids 1/2/3: FIN de mes (``main`` usaba ``pd.date_range(freq='M')``, p. ej. ``2026-01-31``); id 4: INICIO de
    mes (``2026-01-01``). Internamente todo se calcula por inicio de mes; solo la salida se re-etiqueta (es
    idempotente para fechas que ya son fin de mes, como las de la rama legacy).
    """
    fechas = pd.to_datetime(pd.Series(fechas))
    if model_id in (1, 2, 3):
        fechas = fechas.dt.to_period('M').dt.to_timestamp(how='end').dt.normalize()
    return fechas.dt.strftime('%Y-%m-%d').tolist()


def predict_v2_full_scale(model_id: int, hist_serie: pd.Series, start_m: pd.Timestamp,
                          end_m: pd.Timestamp, corte=None) -> tuple:
    """Pronóstico v2 (ids 1/2/3) en escala de mercado completo, desde el mes siguiente al último real usable
    anterior a ``start_m`` hasta ``end_m`` (inclusive).

    Devuelve ``(forecast, historia_previa, info)``: ``forecast`` es una Serie (kWh) indexada por inicio de mes
    que también cubre los meses entre el último real y ``start_m`` (necesarios para totales anuales), e
    ``historia_previa`` son los reales usados (< ``start_m``, < ``corte``).
    """
    from api.services import forecast_rfxgb, forecast_sarimax

    hist = hist_serie[hist_serie.index < start_m]
    if corte is not None:
        hist = hist[hist.index < corte]
    if hist.empty:
        raise ValueError(f"No hay histórico de demanda utilizable antes de {start_m:%Y-%m}")
    last = hist.index.max()
    fc_start = last + pd.offsets.MonthBegin(1)
    if fc_start > end_m:
        raise ValueError("Rango de pronóstico vacío")
    if model_id in (1, 2):
        art = forecast_rfxgb.load_artifact(model_id)
        refit = _month_ts(art.get("train_last_month")) >= fc_start
        if refit:
            # Hindcast: el artefacto vio meses >= fc_start -> se re-entrena con la historia disponible (misma
            # config) para que el error mostrado sea fuera de muestra (igual que SARIMAX refit="auto").
            art = forecast_rfxgb.refit_from_artifact_cached(
                model_id, hist, art, fingerprint=forecast_rfxgb.artifact_fingerprint(model_id))
        df = forecast_rfxgb.predict_monthly(model_id, hist, fc_start, end_m, market_break_month=corte, artifact=art)
        fc = pd.Series(df["demanda_mercado_completo_kwh"].to_numpy(dtype=float),
                       index=pd.DatetimeIndex(df["fecha"]))
        info = {"nombre": df.attrs.get("model_name"), "entrenado_hasta": art.get("train_last_month"),
                "reestimado": bool(refit)}
    elif model_id == 3:
        df = forecast_sarimax.predict_monthly(hist, fc_start, end_m, exclude_from=corte)
        fc = pd.Series(df["prediccion"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["fecha"]))
        info = {"nombre": df.attrs.get("modelo"), "reestimado": bool(df.attrs.get("refit"))}
    else:
        raise ValueError(f"Modelo v2 no soportado: id={model_id}")
    expected = _months_between(fc_start, end_m) + 1
    if len(fc) != expected or not np.isfinite(fc.to_numpy()).all() or (fc.to_numpy() <= 0).any():
        raise ValueError("El modelo devolvió un pronóstico incompleto o no finito")
    info.update({"version": "v2", "ultimo_mes_historia": f"{last:%Y-%m}",
                 "corte_division": f"{corte:%Y-%m}" if corte is not None else None})
    return fc, hist, info


def v2_internal_gaps(hist_prev: pd.Series) -> pd.DatetimeIndex:
    """Meses sin real usable dentro del año del último real (enero .. último mes), p. ej. un mes con menos del
    50 % de días que no se imputó. Sin rellenarlos, el total anual de ese año no tendría 12 meses."""
    if hist_prev.empty:
        return pd.DatetimeIndex([])
    last = hist_prev.index.max()
    months = pd.date_range(pd.Timestamp(year=last.year, month=1, day=1), last, freq="MS")
    return months.difference(hist_prev.index)


def v2_annual_totals(years: dict, hist_prev: pd.Series, fc: pd.Series, years_to_predict, start_m: pd.Timestamp,
                     gap_fill: pd.Series = None) -> tuple:
    """Totales anuales en escala de mercado completo para la rama v2.

    Años pedidos = reales previos a ``start_m`` (+ pronóstico de huecos internos ``gap_fill``) + pronóstico del
    resto. Años anteriores a ``start`` con meses sin real usable (incompleto, preliminar o post-división) se
    completan con pronóstico solo para que ``growth`` compare en la misma escala.

    Devuelve ``(years, anios_estimados, fijo_por_anio)``; ``fijo_por_anio[y]`` es la parte del total del año
    pedido ``y`` que corresponde a meses anteriores a ``start_m`` (el ajuste IPC no la escala).
    """
    years = dict(years)
    parts = [hist_prev]
    if gap_fill is not None and len(gap_fill):
        parts.append(gap_fill)
    parts.append(fc)
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep='first')]  # un real nunca se sustituye por pronóstico
    last_hist = hist_prev.index.max()
    start_year = min(years_to_predict)
    estimated, fixed = [], {}
    for year in sorted(set(full.index.year)):
        if year < last_hist.year:
            continue
        vals = full[full.index.year == year]
        if len(vals) != 12:
            continue
        if year in years_to_predict:
            years[year] = int(vals.sum())
            fixed[year] = float(vals[vals.index < start_m].sum())
        elif year < start_year and (vals.index > last_hist).any():
            years[year] = int(vals.sum())
            estimated.append(int(year))
    missing = [y for y in years_to_predict if y not in years or not np.isfinite(years[y])]
    if missing:
        raise ValueError(f"No se pudo calcular el total anual de {missing}")
    return years, estimated, fixed


def run_v2_forecast(model_id: int, hist_serie: pd.Series, corte, start_m: pd.Timestamp, years_to_predict,
                    years: dict) -> dict:
    """Rama v2 completa (ids 1/2/3): pronóstico hasta diciembre del último año pedido, relleno de los huecos
    internos del año del último real con el pronóstico del modelo (pronosticando desde el primer hueco) y
    totales anuales.

    Devuelve un dict con ``fc`` (Serie), ``years``, ``estimated_years``, ``fixed_by_year`` e ``info``.
    """
    fc_end = pd.Timestamp(year=max(years_to_predict), month=12, day=1)
    fc, hist_prev, info = predict_v2_full_scale(model_id, hist_serie, start_m, fc_end, corte)
    gaps = v2_internal_gaps(hist_prev)
    gap_fill = None
    if len(gaps):
        fc_gap, _, _ = predict_v2_full_scale(model_id, hist_serie, gaps[0], gaps[-1], corte)
        gap_fill = fc_gap.reindex(gaps)
        info["meses_hueco_rellenados"] = [f"{m:%Y-%m}" for m in gaps]
    years_out, estimated, fixed = v2_annual_totals(years, hist_prev, fc, years_to_predict, start_m, gap_fill)
    return {"fc": fc, "years": years_out, "estimated_years": estimated, "fixed_by_year": fixed, "info": info}


# Soporte para el modelo climático + ONI (id=4). Este import de `training.forecast_climate` es defensivo (si
# falla, solo la rama id=4 falla en tiempo de uso). OJO: el paquete `training` es OBLIGATORIO de todas formas:
# `forecast_history` (historia limpia de los modelos v2) importa `training.common` a nivel de módulo justo
# debajo, sin try/except, así que sin `training/` este módulo no carga. El Dockerfile copia `./training`.
try:
    import sys as _sys
    from pathlib import Path as _Path
    _BACKEND_ROOT = _Path(__file__).resolve().parents[2]
    if str(_BACKEND_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_BACKEND_ROOT))
    from training.forecast_climate import features as climate_features
    from training.forecast_climate.postprocess_calibration import (
        apply_correctors as climate_apply_correctors,
        load_corrector_artifact as climate_load_corrector_artifact,
    )
except Exception as _climate_import_error:
    climate_features = None
    climate_apply_correctors = None
    climate_load_corrector_artifact = None
    print(f"[climate_model] No se pudo importar el soporte del modelo climático: {_climate_import_error}")

# Historia limpia por petición (usa training.common; la raíz del backend ya está en sys.path arriba). Import
# NO defensivo a propósito: sin `training/` la API falla al arrancar en vez de degradar en silencio.
from api.services.forecast_history import division_from_config, load_clean_history_session  # noqa: E402

# Medianas históricas de ONI por fase ENSO (dataset_mensual.csv). Usadas
# cuando el cliente envía `enso_phase_by_month` para forzar la fase de un mes.
ENSO_PHASE_ONI = {"NIÑO": 0.9, "NIÑA": -0.8, "NORMAL": -0.1}
_ENSO_VALID_PHASES = frozenset(ENSO_PHASE_ONI.keys())

# Fuerza del shrinkage LEVE aplicado en `_isotonic_phase_month_shares`
# (shrunk = fallback + n/(n+k_light) * (promedio_condicional - fallback))
# ANTES de correr PAVA. A diferencia del viejo `PHASE_SHARE_SHRINKAGE_K=85`
# (shrinkage uniforme y agresivo, único mecanismo disponible para evitar
# violar el orden físico NIÑA <= NORMAL <= NIÑO, que dejaba la sensibilidad
# del selector "Escenario ENSO (mes a mes)" casi eliminada: ~1% de spread
# entre NIÑA y NIÑO), el orden físico ahora lo garantiza estructuralmente
# `_weighted_pava_nondecreasing` (pool-adjacent-violators ponderado por
# tamaño de muestra) -- PAVA es la identidad si el orden crudo ya se
# cumple, y solo poolea (promedia ponderado) las celdas adyacentes que
# violan el orden. El rol de k_light pasa a ser exclusivamente amortiguar
# ruido de muestras pequeñas (n=1-2), no evitar violaciones. Fijado por
# barrido explícito sobre los artefactos reales de api/models/climate_model/:
# probando k_light en {0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 85}, todos
# alcanzan 100% de compliance (garantía estructural de PAVA). Se descartaron
# k_light=0 y 0.5 pese a superar el umbral de spread porque no amortiguan
# celdas reales de muestra mínima (n=1, ej. junio con n_niña=1); entre los
# candidatos restantes con spread promedio sustancialmente mayor (>=3x) que
# el baseline k_light=85 (~1%) -- {1: 3.99%, 2: 3.47%, 3: 3.11%} -- se
# eligió k_light=3 por dar margen cómodo sobre el borde del umbral 3x en
# vez de operar al límite exacto, sin que el spread máximo esté dominado
# por una celda n=1 aislada. Ver
# training/forecast_climate/artifacts/reporte_isotonic_enso.md.
PHASE_SHARE_LIGHT_SHRINKAGE_K = 3


def _oni_to_enso_phase(oni_actual: float) -> str:
    """Clasifica ONI en NIÑO / NIÑA / NORMAL (umbrales NOAA ±0.5)."""
    if oni_actual >= 0.5:
        return "NIÑO"
    if oni_actual <= -0.5:
        return "NIÑA"
    return "NORMAL"


def _phase_conditional_stats(hist_pct: pd.DataFrame, month: int, phase: str):
    """Media y tamaño de muestra crudos de `demand_percentage` para la
    celda (month, phase), sin fallback ni shrinkage.

    `hist_pct` debe tener columnas `month`, `oni_actual`, `demand_percentage`.

    Devuelve `(mean, n)`. `phase` inválida, mes sin filas, o subset vacío
    (n=0) devuelven `(None, 0)`.
    """
    if phase not in _ENSO_VALID_PHASES:
        return None, 0
    subset = hist_pct.loc[hist_pct['month'] == month]
    if subset.empty:
        return None, 0
    phase_mask = subset['oni_actual'].map(_oni_to_enso_phase) == phase
    phase_series = subset.loc[phase_mask, 'demand_percentage']
    n = len(phase_series)
    if n == 0:
        return None, 0
    return float(phase_series.mean()), n


def _weighted_pava_nondecreasing(values, weights):
    """PAVA (pool-adjacent-violators) ponderado: dada una secuencia
    `values` (en orden de índice, p.ej. NIÑA/NORMAL/NIÑO) y sus `weights`
    (p.ej. tamaño de muestra `n` por celda), devuelve la secuencia no
    decreciente más cercana (mínimos cuadrados ponderados), preservando el
    orden.

    Si `values` ya es no decreciente, el resultado es exactamente `values`
    (identidad) -- no se preserva ninguna magnitud artificialmente, solo se
    corrige cuando hay una violación real. Cuando hay violación, las
    entradas adyacentes que la causan se funden (poolean) en un único
    bloque cuyo valor es el promedio ponderado de sus miembros; el proceso
    se repite en cascada mientras el nuevo bloque siga violando el orden
    con el bloque anterior.

    Implementación clásica por pila de bloques `[valor, peso_total,
    cantidad_de_elementos]`.
    """
    blocks = []  # cada bloque: [valor_ponderado, peso_total, cantidad]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            merged_w = w1 + w2
            if merged_w > 0:
                merged_v = (v1 * w1 + v2 * w2) / merged_w
            else:
                merged_v = (v1 + v2) / 2.0
            blocks.append([merged_v, merged_w, c1 + c2])

    result = []
    for v, _w, c in blocks:
        result.extend([v] * c)
    return result


def _isotonic_phase_month_shares(
    hist_pct: pd.DataFrame,
    month: int,
    fallback: float,
    shrinkage_k: float = None,
) -> dict:
    """Calcula los 3 shares de `month` condicionados a fase ENSO
    (NIÑA/NORMAL/NIÑO) de una sola vez: shrinkage LEVE hacia `fallback`
    ponderado por tamaño de muestra, seguido de PAVA ponderado por `n` que
    garantiza estructuralmente NIÑA <= NORMAL <= NIÑO.

    `hist_pct` debe tener columnas `month`, `oni_actual`, `demand_percentage`.
    `shrinkage_k=None` usa `PHASE_SHARE_LIGHT_SHRINKAGE_K`.

    Devuelve `{"NIÑA": share, "NORMAL": share, "NIÑO": share}`.
    """
    if shrinkage_k is None:
        shrinkage_k = PHASE_SHARE_LIGHT_SHRINKAGE_K

    phases = ("NIÑA", "NORMAL", "NIÑO")
    shrunk_values = []
    weights = []
    for phase in phases:
        mean, n = _phase_conditional_stats(hist_pct, month, phase)
        if n == 0:
            shrunk_values.append(fallback)
        else:
            shrunk_values.append(fallback + (n / (n + shrinkage_k)) * (mean - fallback))
        weights.append(n)

    pava_values = _weighted_pava_nondecreasing(shrunk_values, weights)
    return dict(zip(phases, pava_values))


def _phase_conditional_month_share(
    hist_pct: pd.DataFrame,
    month: int,
    phase: str,
    fallback: float,
    shrinkage_k: float = None,
) -> float:
    """Participación de `month` condicionada a la fase ONI `phase`, con
    shrinkage LEVE hacia `fallback` + PAVA ponderado por tamaño de muestra
    (ver `_isotonic_phase_month_shares`) que garantiza estructuralmente el
    orden físico NIÑA <= NORMAL <= NIÑO entre las 3 fases del mismo mes.

    Wrapper de compatibilidad: misma firma que la versión anterior
    (shrinkage-only). `phase` inválida devuelve `fallback` exacto sin tocar
    PAVA. Internamente calcula las 3 fases del mes (NIÑA/NORMAL/NIÑO) y
    devuelve la de `phase`.
    """
    if phase not in _ENSO_VALID_PHASES:
        return fallback
    shares = _isotonic_phase_month_shares(hist_pct, month, fallback, shrinkage_k=shrinkage_k)
    return shares[phase]


class DemandPredictor:
    def __init__(self, model_path='arima_model/'):
        try:
            """
            Inicializa el predictor cargando los modelos y escaladores necesarios.
            """
            self.model_path = model_path
            
            if model_path.endswith('xgb_model/') or model_path.endswith('rf_model/'):

                self.scaler_x = joblib.load(f'{model_path}annual_scaler_x.pkl')
                self.scaler_y = joblib.load(f'{model_path}annual_scaler_y.pkl')
                self.annual_model = joblib.load(f'{model_path}annual_model.pkl')
                self.monthly_model = joblib.load(f'{model_path}monthly_model.pkl')
                self.imputer = SimpleImputer(strategy='mean')
            
        
            # Obtener las características directamente del scaler
        
                self.annual_features = list(self.scaler_x.feature_names_in_)
            
                # Variables derivadas (para identificarlas al filtrar)
                self.derived_features = {
                    'annual': ['lag_1', 'lag_2', 'lag_3', 'growth_rate',
                                'year_index', 'year_index_norm', 'trend_squared',
                                'growth_momentum', 'ma_2'],
                    'monthly': ['mes', 'mes_sin', 'mes_cos', 'trend',
                                'percentage_lag_12', 'percentage_lag_1', 'dias_laborales',
                                'pct_mean_mes', 'dias_totales', 'enso']
                }

                # Cargar clasificación ENSO histórica y futura
                try:
                    _enso_path = '/'.join(model_path.rstrip('/').rsplit('/', 1)[:-1]) + '/enso.csv'
                    _enso_df = pd.read_csv(_enso_path)
                    self.enso_map = _enso_df.set_index('ano')['enso'].to_dict()
                    self._enso_max_year = int(_enso_df['ano'].max())
                except Exception:
                    self.enso_map = {}
                    self._enso_max_year = 0
                '''
                print("Características requeridas para el modelo anual:")
                print(self.scaler_x.feature_names_in_)
                '''
                metrics=joblib.load(f'{model_path}metrics.pkl')
                self.annual_mape=metrics['annual']['mape']
                self.monthly_mape=metrics['monthly']['mape']
        
            
            elif model_path=='arima_model/':
                self.arima_model = joblib.load(f'{model_path}arima_model.pkl')
                self.mape=joblib.load(f'{model_path}metrics.pkl')
        
            elif model_path.endswith('sarimax_model/'):
                self.sarimax_model = joblib.load(f'{model_path}sarimax_model.pkl')
                self.scaler=joblib.load(f'{model_path}scaler.pkl')
                self.info=joblib.load(f'{model_path}model_info.pkl')
                self.mape=self.info['mape']

            elif model_path.endswith('climate_model/'):
                # Modelo id=4: Ridge anual + XGBoost mensual sobre clima + ONI.
                # NO toca ni comparte estado con las ramas xgb/rf/sarimax de arriba.
                self.climate_annual_scaler_x = joblib.load(f'{model_path}annual_scaler_x.pkl')
                self.climate_annual_scaler_y = joblib.load(f'{model_path}annual_scaler_y.pkl')
                self.climate_annual_model = joblib.load(f'{model_path}annual_model.pkl')
                self.climate_monthly_model = joblib.load(f'{model_path}monthly_model.pkl')
                self.climate_model_info = joblib.load(f'{model_path}model_info.pkl')
                climate_metrics = joblib.load(f'{model_path}metrics.pkl')
                self.climate_annual_mape = climate_metrics['annual']['mape']
                self.climate_monthly_mape = climate_metrics['monthly']['mape']

                # Las listas de features se recuperan de los artefactos, no se
                # hardcodean (mismo patrón usado para el modelo anual macro).
                self.climate_annual_features = list(self.climate_annual_scaler_x.feature_names_in_)
                self.climate_monthly_features = list(self.climate_monthly_model.feature_names_in_)

                # Corrector post-predicción calibrado con datos reales (artefacto
                # postproc_corrector_v1.json; lo regenera training/forecast_climate/train_final_clima.py).
                try:
                    self.climate_corrector = climate_load_corrector_artifact(f'{model_path}postproc_corrector_v1.json')
                except Exception as corrector_error:
                    print(f"[climate_model] No se pudo cargar el corrector post-predicción: {corrector_error}")
                    self.climate_corrector = None

                # Histórico real de clima + ONI + demanda (mensual/anual),
                # usado como base para proyectar clima/ONI futuros vía
                # climatología estacional + features.project_oni_future.
                # A diferencia de id=1/2/3, este modelo no depende de que el
                # usuario suministre variables macro futuras.
                _climate_artifacts_dir = _BACKEND_ROOT / 'training' / 'forecast_climate' / 'artifacts'
                self.climate_hist_monthly = pd.read_csv(_climate_artifacts_dir / 'dataset_mensual.csv')
                self.climate_hist_annual = pd.read_csv(_climate_artifacts_dir / 'dataset_anual.csv')
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")

    def _get_enso(self, year: int) -> int:
        """
        Devuelve el valor ENSO para el año dado.
        Si el año no está en enso.csv, usa 0 (Neutral) y emite un aviso (una sola vez por año).
        Actualizar enso.csv anualmente con el pronóstico de IDEAM/NOAA.
        """
        if year in self.enso_map:
            return self.enso_map[year]
        if not hasattr(self, '_enso_warned'):
            self._enso_warned = set()
        if year not in self._enso_warned:
            print(f"[ENSO] Año {year} sin clasificación en enso.csv — usando Neutral (0). "
                  f"Actualiza enso.csv si el pronóstico de IDEAM/NOAA está disponible.")
            self._enso_warned.add(year)
        return 0

    def get_mape(self):
        if self.model_path=='arima_model/':
            return self.mape
        elif self.model_path=='xgb_model/' or self.model_path=='rf_model/':
            mape={'annual':self.annual_mape,'monthly':self.monthly_mape}
            return mape
        elif self.model_path=='sarimax_model/':
            return self.mape
        elif self.model_path.endswith('climate_model/'):
            return {'annual': self.climate_annual_mape, 'monthly': self.climate_monthly_mape}

    def prepare_annual_features_future(self, historical_data, forecast_years, future_macro_vars):
        """
        Prepara las características para predicciones futuras anuales.
        Versión actualizada con features de tendencia cuadrática.
        """
        required_features = [
            'ipc',
            'Crecimiento de la población (%) anual',
            'Inflación, deflactor del PIB: series vinculadas (% anual)',
            'Comercio de mercaderías (% del PIB)',
            'Importaciones de bienes (balanza de pagos, US$ a precios actuales)',
            'Empleo de tiempo parcial, total (% del total de empleo)',
            'lag_1',
            'lag_2',
            'lag_3',
            'growth_rate',
            'year_index',
            'year_index_norm',
            'trend_squared',
            'growth_momentum',
            'ma_2'
        ]
        
        # 1. Preparar los datos históricos con todas las features
        hist_data = historical_data.copy()
        hist_data['log_total'] = np.log1p(hist_data['total'])
        
        # Calcular year_index para datos históricos
        hist_data['year_index'] = range(len(hist_data))
        hist_data['year_index_norm'] = hist_data['year_index'] / len(hist_data)
        hist_data['trend_squared'] = hist_data['year_index_norm'] ** 2
        hist_data['growth_rate'] = hist_data['log_total'].diff()
        hist_data['growth_momentum'] = hist_data['growth_rate'].rolling(
            window=3, min_periods=1
        ).mean()
        hist_data['ma_2'] = hist_data['log_total'].rolling(
            window=2, min_periods=1
        ).mean()
        
        # 2. Crear DataFrame futuro con las variables macro
        future_data = future_macro_vars.copy()
        
        # 3. Calcular lags usando los datos históricos
        last_log_total = hist_data['log_total'].iloc[-3:]
        future_data['lag_1'] = last_log_total.iloc[-1]
        future_data['lag_2'] = last_log_total.iloc[-2]
        future_data['lag_3'] = last_log_total.iloc[-3]
        
        # 4. Calcular growth rate
        last_growth = hist_data['log_total'].diff().iloc[-1]
        future_data['growth_rate'] = last_growth
        
        # 5. Calcular features de tendencia (NUEVO)
        # El year_index continúa desde donde quedó el histórico
        last_year_index = hist_data['year_index'].iloc[-1]
        total_years = len(hist_data) + forecast_years
        
        future_year_indices = range(last_year_index + 1, last_year_index + 1 + forecast_years)
        future_data['year_index'] = list(future_year_indices)
        future_data['year_index_norm'] = future_data['year_index'] / total_years
        future_data['trend_squared'] = future_data['year_index_norm'] ** 2
        
        # 6. Growth momentum (usar último valor histórico como aproximación)
        last_momentum = hist_data['growth_momentum'].iloc[-1]
        future_data['growth_momentum'] = last_momentum
        
        # 7. MA_2 (usar últimos valores históricos)
        last_ma_2 = hist_data['ma_2'].iloc[-1]
        future_data['ma_2'] = last_ma_2
        
        # 8. Verificar y ordenar columnas
        for feature in required_features:
            if feature not in future_data.columns:
                future_data[feature] = np.nan
        
        # 9. Ordenar columnas según el orden requerido
        X_future = future_data[required_features]
        
        return X_future

    def prepare_single_month_features(self, hist_monthly, current_year, current_month, year_monthly_values, year_total=None):
        """
        Prepara features para un solo mes, similar al método predict_monthly_values del script de entrenamiento.
        """
        # Calcular días laborales para este mes (con fallback si no hay workalendar)
        try:
            from workalendar.america import Colombia
            cal = Colombia()
            first_date = pd.Timestamp(current_year, current_month, 1)
            if current_month == 12:
                last_date = pd.Timestamp(current_year + 1, 1, 1) - pd.Timedelta(days=1)
            else:
                last_date = pd.Timestamp(current_year, current_month + 1, 1) - pd.Timedelta(days=1)
            dias_laborales = cal.get_working_days_delta(first_date.to_pydatetime(), last_date.to_pydatetime())
        except ImportError:
            # Fallback: aproximación simple de días laborales
            days_in_month = pd.Timestamp(current_year, current_month, 1).days_in_month
            # Aproximadamente 22 días laborales por mes (5/7 de los días)
            dias_laborales = int(days_in_month * 5/7)

        # Features básicas
        features = {
            'mes': current_month,
            'mes_sin': np.sin(2 * np.pi * current_month / 12),
            'mes_cos': np.cos(2 * np.pi * current_month / 12),
            'trend': len(hist_monthly) + len(year_monthly_values),  # Índice continuo
            'dias_laborales': dias_laborales,
            'trimestre': ((current_month - 1) // 3 + 1)
        }

        # Lags
        if len(year_monthly_values) > 0:
            # Si ya tenemos predicciones de meses anteriores de este año
            last_month_percentage = year_monthly_values[-1] / year_total if year_total else 1/12
            features['percentage_lag_1'] = last_month_percentage
        else:
            # Primer mes del año, usar último mes del año anterior
            prev_year_data = hist_monthly[hist_monthly['ano'] == current_year - 1]
            if len(prev_year_data) > 0:
                features['percentage_lag_1'] = prev_year_data['percentage'].iloc[-1]
            else:
                features['percentage_lag_1'] = 1/12

        # Lag 12 (mismo mes año anterior)
        prev_year_same_month = hist_monthly[
            (hist_monthly['ano'] == current_year - 1) &
            (hist_monthly['mes'] == current_month)
        ]
        if len(prev_year_same_month) > 0:
            features['percentage_lag_12'] = prev_year_same_month['percentage'].iloc[0]
        else:
            # Usar promedio del mes
            same_month_data = hist_monthly[hist_monthly['mes'] == current_month]
            features['percentage_lag_12'] = same_month_data['percentage'].mean() if len(same_month_data) > 0 else 1/12

        # Lag 13 (mismo mes hace 2 años)
        prev_year_2_same_month = hist_monthly[
            (hist_monthly['ano'] == current_year - 2) &
            (hist_monthly['mes'] == current_month)
        ]
        if len(prev_year_2_same_month) > 0:
            features['percentage_lag_13'] = prev_year_2_same_month['percentage'].iloc[0]
        else:
            features['percentage_lag_13'] = features['percentage_lag_12']  # Usar lag_12 como fallback

        # Media móvil del mismo mes (últimos 3 años)
        same_month_data = hist_monthly[
            (hist_monthly['mes'] == current_month) &
            (hist_monthly['ano'] < current_year) &
            (hist_monthly['ano'] >= max(current_year - 3, hist_monthly['ano'].min()))
        ]['percentage']
        features['ma_same_month_3y'] = same_month_data.mean() if len(same_month_data) > 0 else 1/12

        # Promedio histórico del mes
        hist_data = hist_monthly[
            (hist_monthly['mes'] == current_month) &
            (hist_monthly['ano'] < current_year)
        ]['percentage']
        features['month_historical_avg'] = hist_data.mean() if len(hist_data) > 0 else 1/12

        # Convertir a DataFrame en el orden correcto
        feature_order = ['mes', 'mes_sin', 'mes_cos', 'trend',
                        'percentage_lag_1', 'percentage_lag_12', 'percentage_lag_13',
                        'dias_laborales', 'trimestre',
                        'ma_same_month_3y', 'month_historical_avg']

        return pd.DataFrame([features], columns=feature_order)

    def prepare_year_monthly_features(self, hist_monthly, current_year):
        """
        Prepara features para TODOS los meses de un año de una vez.
        Versión actualizada 2025 con nuevas features (pct_mean_mes).
        """
        months = range(1, 13)

        # Calcular días laborales para cada mes
        try:
            from workalendar.america import Colombia
            cal = Colombia()
            dias_lab = []
            for month in months:
                first_date = pd.Timestamp(current_year, month, 1)
                if month == 12:
                    last_date = pd.Timestamp(current_year + 1, 1, 1) - pd.Timedelta(days=1)
                else:
                    last_date = pd.Timestamp(current_year, month + 1, 1) - pd.Timedelta(days=1)
                dias_laborales = cal.get_working_days_delta(first_date.to_pydatetime(), last_date.to_pydatetime())
                dias_lab.append(dias_laborales)
        except ImportError:
            # Fallback: aproximación simple
            dias_lab = []
            for month in months:
                days_in_month = pd.Timestamp(current_year, month, 1).days_in_month
                dias_lab.append(int(days_in_month * 5/7))

        # Calcular year_idx como en el script de entrenamiento
        last_year_in_data = hist_monthly['ano'].max()
        last_idx = hist_monthly[hist_monthly['ano'] == last_year_in_data].index[-1]
        years_ahead = current_year - last_year_in_data
        year_idx = last_idx + 1 + (years_ahead - 1) * 12

        dias_tot = [calendar.monthrange(current_year, m)[1] for m in months]
        enso_val = self._get_enso(current_year)

        # Features básicas
        X_pred = pd.DataFrame({
            'mes': months,
            'mes_sin': np.sin(2 * np.pi * np.array(months)/12),
            'mes_cos': np.cos(2 * np.pi * np.array(months)/12),
            'trend': range(year_idx, year_idx + 12),
            'dias_laborales': dias_lab,
            'dias_totales': dias_tot,
            'enso': enso_val,
        })

        # Lags
        # Los porcentajes mensuales solo sirven de un año COMPLETO: si el histórico llega a mitad de año
        # (p. ej. se pronostica desde abril o noviembre), el porcentaje de cada mes de ese año parcial se
        # calcula contra un total parcial y no representa la estacionalidad (además tiene menos de 12 meses,
        # lo que antes fallaba con "Length of values (N) does not match length of index (12)").
        hist_pct = hist_monthly[hist_monthly.groupby('ano')['mes'].transform('nunique') == 12]
        if len(hist_pct) > 0 and current_year > hist_pct['ano'].min():
            # Datos del año anterior
            prev_year_data = hist_pct[hist_pct['ano'] == current_year - 1]

            if len(prev_year_data) > 0:
                # lag_12: usar año anterior completo
                X_pred['percentage_lag_12'] = prev_year_data['percentage'].values
            else:
                # Fallback: promedio histórico por mes
                hist_avg_by_month = []
                for m in months:
                    hist_data = hist_pct[hist_pct['mes'] == m]['percentage']
                    hist_avg_by_month.append(hist_data.mean() if len(hist_data) > 0 else 1/12)
                X_pred['percentage_lag_12'] = hist_avg_by_month

            # lag_1: usar último mes del año anterior para TODOS los meses
            if len(prev_year_data) > 0:
                last_month_prev_year = prev_year_data['percentage'].iloc[-1]
                X_pred['percentage_lag_1'] = last_month_prev_year
            else:
                X_pred['percentage_lag_1'] = 1/12

            # pct_mean_mes: Promedio móvil de 3 años del mismo mes
            # Replica: df.groupby('mes')['percentage'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            pct_mean_values = []
            for m in months:
                # Obtener porcentajes históricos del mismo mes
                same_month_data = hist_pct[
                    (hist_pct['mes'] == m) &
                    (hist_pct['ano'] < current_year)
                ].sort_values('ano')['percentage']

                if len(same_month_data) > 0:
                    # Tomar los últimos 3 valores disponibles y promediar
                    recent_values = same_month_data.tail(3)
                    pct_mean_values.append(recent_values.mean())
                else:
                    pct_mean_values.append(1/12)

            X_pred['pct_mean_mes'] = pct_mean_values

        else:
            # Primer año: usar valores por defecto
            X_pred['percentage_lag_1'] = 1/12
            X_pred['percentage_lag_12'] = 1/12
            X_pred['pct_mean_mes'] = 1/12

        return X_pred[['mes', 'mes_sin', 'mes_cos', 'trend', 'percentage_lag_12', 'percentage_lag_1', 'dias_laborales', 'pct_mean_mes', 'dias_totales', 'enso']]

    def prepare_monthly_features_future(self, historical_monthly_data, num_months):
        """
        Prepara las características para predicciones futuras mensuales.
        Versión actualizada 2025 con nuevas features (pct_mean_mes).
        """
        # Calcular el porcentaje mensual en los datos históricos
        hist_data = historical_monthly_data.copy()
        hist_data['yearly_total'] = hist_data.groupby('ano')['total'].transform('sum')
        hist_data['percentage'] = hist_data['total'] / hist_data['yearly_total']

        # Crear DataFrame para meses futuros
        last_date = pd.to_datetime(hist_data['fecha'].max())
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=num_months,
            freq='M'
        )

        future_df = pd.DataFrame({
            'fecha': future_dates,
            'ano': future_dates.year,
            'mes': future_dates.month,
            'dias_laborales': 22,  # Valor por defecto
            'dias_totales': [calendar.monthrange(y, m)[1] for y, m in zip(future_dates.year, future_dates.month)],
            'enso': [self._get_enso(y) for y in future_dates.year],
        })

        # Features básicas
        future_df['mes_sin'] = np.sin(2 * np.pi * future_df['mes']/12)
        future_df['mes_cos'] = np.cos(2 * np.pi * future_df['mes']/12)
        future_df['trend'] = range(
            len(hist_data),
            len(hist_data) + len(future_df)
        )

        # Combinar datos históricos y futuros para calcular lags
        df_combined = pd.concat(
            [hist_data[['fecha', 'mes', 'ano', 'percentage']], future_df],
            ignore_index=True
        )
        df_combined = df_combined.sort_values('fecha')

        # Calcular lags
        df_combined['percentage_lag_1'] = df_combined['percentage'].shift(1)
        df_combined['percentage_lag_12'] = df_combined['percentage'].shift(12)

        # Calcular pct_mean_mes: Promedio móvil de 3 años del mismo mes
        pct_mean_values = []
        for idx, row in df_combined.tail(num_months).iterrows():
            mes_actual = row['mes']
            año_actual = row['ano']

            # Obtener porcentajes históricos del mismo mes
            same_month_data = hist_data[
                (hist_data['mes'] == mes_actual) &
                (hist_data['ano'] < año_actual)
            ].sort_values('ano')['percentage']

            if len(same_month_data) > 0:
                # Tomar los últimos 3 valores disponibles y promediar
                recent_values = same_month_data.tail(3)
                pct_mean_values.append(recent_values.mean())
            else:
                # Fallback: distribución uniforme
                pct_mean_values.append(1/12)

        # Obtener solo las filas futuras y agregar las features calculadas
        future_df['percentage_lag_1'] = df_combined['percentage_lag_1'].tail(num_months).values
        future_df['percentage_lag_12'] = df_combined['percentage_lag_12'].tail(num_months).values
        future_df['pct_mean_mes'] = pct_mean_values

        # Rellenar NaN con valores por defecto (para casos donde no hay suficiente historia)
        future_df['percentage_lag_1'].fillna(1/12, inplace=True)
        future_df['percentage_lag_12'].fillna(1/12, inplace=True)
        future_df['pct_mean_mes'].fillna(1/12, inplace=True)

        features = ['mes', 'mes_sin', 'mes_cos', 'trend',
                    'percentage_lag_12', 'percentage_lag_1', 'dias_laborales',
                    'pct_mean_mes', 'dias_totales', 'enso']

        return future_df[features]
    
    def filter_features(self, df, model_name):
        """
        Filtra el DataFrame para mantener solo las variables necesarias para el modelo especificado.
        
        Args:
            df (pd.DataFrame): DataFrame con todas las variables (derivadas y macro)
            model_name (str): Nombre del modelo ('annual_model' o 'monthly_model')
        
        Returns:
            pd.DataFrame: DataFrame filtrado con solo las variables necesarias
        """
        if model_name == 'annual_model':
            required_features = self.annual_features
        elif model_name == 'monthly_model':
            required_features = self.derived_features['monthly']
        else:
            raise ValueError(f"Modelo '{model_name}' no soportado")

        # Verificar que todas las variables requeridas estén presentes
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Faltan las siguientes variables requeridas: {missing_features}")
        
        # Mantener solo las columnas necesarias en el orden correcto
        filtered_df = df[required_features].copy()
        
        return filtered_df
    
    def remove_atypical_years(self, df, atypical_years, year_column='ano'):
        """
        Removes rows from a DataFrame where the year is in the specified atypical_years array.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame containing the year column
        atypical_years : list or array-like
            List of years to remove from the DataFrame
        year_column : str, default='ano'
            Name of the column containing the years
            
        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with the atypical years removed
            
        Raises:
        -------
        ValueError
            If the year_column doesn't exist in the DataFrame
            If atypical_years is empty or contains non-numeric values
        """
        try:
            # Input validation
            if df.index.name == year_column:
                mask = ~df.index.isin(atypical_years)
                filtered_df = df[mask].copy()

            elif year_column not in df.columns:
                raise ValueError(f"Column '{year_column}' not found in DataFrame. "
                                f"Available columns are: {', '.join(df.columns)}")
            else:
                # Create mask for filtering
                mask = ~df[year_column].isin(atypical_years)
                
                # Apply filter and create new DataFrame
                filtered_df = df[mask].copy()
            
            # Log information about removed data
            removed_count = len(df) - len(filtered_df)
            print(f"Removed {removed_count} rows containing years: {sorted(atypical_years)}")
            print(f"Remaining data shape: {filtered_df.shape}")
        except Exception as e:
            print("Error durante el filtrado de años atipicos", str(e))
            raise e
        return filtered_df

    def predict_future_arima(self,df, forecast_periods=12):
        """
        Realiza predicciones usando un modelo ARIMA previamente entrenado.
        
        Args:
            df (pd.DataFrame): DataFrame con columnas ['ano', 'mes', 'total']
            forecast_periods (int): Número de meses a predecir
            
        Returns:
            pd.DataFrame: Predicciones con fechas futuras
        """
       
        try:
            # 1. Cargar el modelo
            model = self.arima_model
            
            
            # 3. Hacer predicciones
            forecast = model.forecast(steps=forecast_periods)
            
            # 4. Crear fechas futuras
            last_date = df['fecha'].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_periods,
                freq='M'
            )
            
            # 5. Crear DataFrame de resultados
            predictions = pd.DataFrame({
                'fecha': future_dates,
                'ano': future_dates.year,
                'mes': future_dates.month,
                'prediccion': forecast
            })
            
            return predictions
            
        except Exception as e:
            print(f"Error al realizar predicciones: {str(e)}")
            raise e
        

    def predict_future_xgrf(self, historical_annual_data, historical_monthly_data, 
                    forecast_years=1, future_macro_vars=None, atypical_years=None):
        """
        Realiza predicciones futuras para los próximos años de forma secuencial.
        """
        if future_macro_vars is None:
            raise ValueError("Debes proporcionar las variables macroeconómicas futuras")
            
        try:
            # Preparar datos anuales futuros
            # historical_annual_data = self.remove_atypical_years(historical_annual_data, atypical_years)
            # print("con annual no es")
            print(historical_annual_data.head())

            
            # Preparar datos históricos con log
            hist_data = historical_annual_data.copy()
            hist_data['log_total'] = np.log1p(hist_data['total'])
            
            # Variables requeridas
            required_features = [
                'ipc',
                'Crecimiento de la población (%) anual',
                'Inflación, deflactor del PIB: series vinculadas (% anual)',
                'Comercio de mercaderías (% del PIB)',
                'Importaciones de bienes (balanza de pagos, US$ a precios actuales)',
                'Empleo de tiempo parcial, total (% del total de empleo)',
                'lag_1',
                'lag_2',
                'lag_3',
                'growth_rate',
                'year_index',
                'year_index_norm',
                'trend_squared',
                'growth_momentum',
                'ma_2'
            ]
            
            # Inicializar listas para predicciones
            annual_predictions = []
            
            # Obtener los últimos valores para inicializar los lags
            last_log_totals = hist_data['log_total'].values[-3:].tolist()  # Últimos 3 valores como lista
            last_growth = hist_data['log_total'].diff().iloc[-1]
            
            # Realizar predicciones secuencialmente
            for year_idx in range(forecast_years):
                print(f"\n=== Prediciendo año {future_macro_vars.index[year_idx]} ===")
                
                # Crear DataFrame para el año actual
                current_year_data = pd.DataFrame(
                    future_macro_vars.iloc[year_idx]).T
                
                # ===== CALCULAR FEATURES DE LAGS =====
                current_year_data['lag_1'] = last_log_totals[-1]
                current_year_data['lag_2'] = last_log_totals[-2]
                current_year_data['lag_3'] = last_log_totals[-3]
                current_year_data['growth_rate'] = last_growth
                
                # ===== CALCULAR FEATURES DE TENDENCIA (NUEVO) =====
                # Calcular year_index: continúa desde donde quedó el histórico
                current_year_number = future_macro_vars.index[year_idx]
                years_from_start = current_year_number - hist_data.index[0]  # Años desde el inicio del histórico
                current_year_data['year_index'] = years_from_start
                
                # Calcular year_index_norm: normalizado respecto al total de años disponibles
                # (histórico + años futuros ya predichos + año actual)
                total_years_so_far = len(hist_data) + year_idx + 1
                current_year_data['year_index_norm'] = current_year_data['year_index'] / total_years_so_far
                
                # Calcular trend_squared
                current_year_data['trend_squared'] = current_year_data['year_index_norm'] ** 2
                
                # Calcular growth_momentum: usar último valor disponible
                if year_idx == 0:
                    # Primer año futuro: usar último momentum del histórico
                    last_growth_values = hist_data['log_total'].diff().tail(3)
                    current_year_data['growth_momentum'] = last_growth_values.mean()
                else:
                    # Años subsiguientes: promedio de últimos crecimientos (incluyendo predichos)
                    current_year_data['growth_momentum'] = last_growth
                
                # Calcular ma_2: promedio móvil de últimos 2 log_totals
                if year_idx == 0:
                    # Primer año: usar último valor histórico
                    last_two = hist_data['log_total'].tail(2).mean()
                    current_year_data['ma_2'] = last_two
                else:
                    # Años subsiguientes: promedio de últimos dos valores (incluyendo predichos)
                    current_year_data['ma_2'] = (last_log_totals[-1] + last_log_totals[-2]) / 2
                
                # ===== LOG DE DEBUG =====
                print(f"Lags para año {future_macro_vars.index[year_idx]}:")
                print(f"  lag_1: {current_year_data['lag_1'].values[0]:.4f}")
                print(f"  lag_2: {current_year_data['lag_2'].values[0]:.4f}")
                print(f"  lag_3: {current_year_data['lag_3'].values[0]:.4f}")
                print(f"  growth_rate: {current_year_data['growth_rate'].values[0]:.4f}")
                print(f"Features de tendencia:")
                print(f"  year_index: {current_year_data['year_index'].values[0]:.0f}")
                print(f"  year_index_norm: {current_year_data['year_index_norm'].values[0]:.4f}")
                print(f"  trend_squared: {current_year_data['trend_squared'].values[0]:.4f}")
                print(f"  growth_momentum: {current_year_data['growth_momentum'].values[0]:.4f}")
                print(f"  ma_2: {current_year_data['ma_2'].values[0]:.4f}")
                
                # Verificar y ordenar columnas
                for feature in required_features:
                    if feature not in current_year_data.columns:
                        current_year_data[feature] = np.nan
                
                # Ordenar columnas según el orden requerido
                X_current = current_year_data[required_features]
                
                # Filtrar características
                X_current_filtered = self.filter_features(X_current, 'annual_model')
                
                # Realizar predicción
                X_current_scaled = self.scaler_x.transform(X_current_filtered)
                annual_pred_scaled = self.annual_model.predict(X_current_scaled)
                annual_pred_value = np.expm1(
                    self.scaler_y.inverse_transform(annual_pred_scaled.reshape(-1, 1))
                )[0][0]
                
                print(f"Predicción para año {future_macro_vars.index[year_idx]}: {annual_pred_value:,.0f}")
                prev_value = float(np.expm1(last_log_totals[-1]))  # último total (histórico o predicho)
                if prev_value > 0:
                    implied_growth = (annual_pred_value - prev_value) / prev_value
                    if implied_growth < -0.30:  # caída interanual > 30% = sospechosa
                        hist_log_growth = hist_data['log_total'].diff().dropna()
                        avg_log_growth = float(hist_log_growth.tail(5).mean()) if len(hist_log_growth) else 0.0
                        avg_log_growth = max(0.0, avg_log_growth)  # no arrastrar tendencias negativas espurias
                        sane_value = float(np.expm1(last_log_totals[-1] + avg_log_growth))
                        print(f"⚠️ Predicción degenerada para {future_macro_vars.index[year_idx]}: "
                              f"{annual_pred_value:,.0f} (growth {implied_growth:.1%}). "
                              f"Corrigiendo a {sane_value:,.0f} (growth log medio {avg_log_growth:.4f}).")
                        annual_pred_value = sane_value
                
                # Guardar predicción
                annual_predictions.append(annual_pred_value)
                
                # IMPORTANTE: Actualizar lags para la siguiente predicción
                # Calcular el log de la predicción actual
                current_log_pred = np.log1p(annual_pred_value)
                
                # Actualizar la lista de logs (eliminar el más antiguo y agregar el nuevo)
                last_log_totals.pop(0)  # Eliminar el más antiguo
                last_log_totals.append(current_log_pred)  # Agregar el nuevo
                
                # Actualizar growth rate
                last_growth = current_log_pred - last_log_totals[-2]  # Diferencia con el valor anterior
                
                print(f"Actualizando para siguiente predicción:")
                print(f"  Nuevos últimos 3 logs: {[f'{x:.4f}' for x in last_log_totals]}")
                print(f"  Nuevo growth_rate: {last_growth:.4f}")
            
            # Convertir a array numpy
            annual_pred = np.array(annual_predictions).reshape(-1, 1)
            
            # Preparar datos mensuales futuros
            X_future_monthly = self.prepare_monthly_features_future(
                historical_monthly_data,
                forecast_years * 12
            )
            
            X_future_monthly = self.filter_features(X_future_monthly, 'monthly_model')
            
            # Predecir porcentajes mensuales EN LOTE (como en el script de entrenamiento)
            monthly_predictions = []

            # Obtener datos históricos de porcentajes
            hist_monthly = historical_monthly_data.copy()
            hist_monthly['yearly_total'] = hist_monthly.groupby('ano')['total'].transform('sum')
            hist_monthly['percentage'] = hist_monthly['total'] / hist_monthly['yearly_total']

            for year_idx in range(forecast_years):
                year_total = annual_pred[year_idx][0]
                current_year = hist_monthly['ano'].max() + 1 + year_idx

                # Preparar features para TODO EL AÑO de una vez (como en training script)
                year_features = self.prepare_year_monthly_features(
                    hist_monthly, current_year
                )

                # Filtrar features según el modelo
                year_features_filtered = self.filter_features(year_features, 'monthly_model')

                # Predecir porcentajes para TODO EL AÑO
                year_percentages = self.monthly_model.predict(year_features_filtered)

                # Normalizar porcentajes para que sumen 1
                year_percentages = np.maximum(year_percentages, 0)  # Evitar negativos
                year_percentages = year_percentages / year_percentages.sum()

                # Calcular valores mensuales
                year_monthly_values = year_total * year_percentages
                monthly_predictions.extend(year_monthly_values)

                # Agregar este año predicho a los datos históricos para el siguiente año
                for month in range(1, 13):
                    new_row = pd.DataFrame({
                        'ano': [current_year],
                        'mes': [month],
                        'total': [year_monthly_values[month-1]],
                        'fecha': [pd.Timestamp(current_year, month, 1)],
                        'yearly_total': [year_total],
                        'percentage': [year_percentages[month-1]]
                    })
                    hist_monthly = pd.concat([hist_monthly, new_row], ignore_index=True)
            
            # Crear DataFrame de resultados
            last_date = pd.to_datetime(historical_monthly_data['fecha'].max())
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=len(monthly_predictions),
                freq='M'
            )
            
            results = pd.DataFrame({
                'fecha': future_dates,
                'prediccion': monthly_predictions
            })
            
            return annual_pred, results
            
        except Exception as e:
            print("Error durante la prediccion de xgrf", str(e))
            raise e

    def predict_future_sarimax(self,df_demanda,future_macro_vars, forecast_periods=12,atypical_years=None):
        """
        Realiza predicciones usando el modelo SARIMAX guardado
        
        Args:
            df (pd.DataFrame): DataFrame con columnas año, mes, total y variables macro
            forecast_periods (int): Períodos a predecir
        """
        df_demanda=self.remove_atypical_years(df_demanda, atypical_years)
        # Cargar modelo y componentes
        model = self.sarimax_model
        scaler = self.scaler
        info = self.info
        vars_macro = info['vars_macro']
         # Asegurar que df_demanda tiene fecha como índice
        if 'fecha' in df_demanda.columns:
            df_demanda = df_demanda.set_index('fecha')
    
        historical_data = pd.merge(
            df_demanda,
            future_macro_vars,
            left_on='ano',
            right_index=True,
            how='left'
        )
        
        y_log = np.log1p(historical_data['total'])
        exog_vars = historical_data[vars_macro]
        exog_scaled = scaler.transform(exog_vars)
        
        # Crear fechas futuras correctamente
        last_date = df_demanda.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='M'
        )
        
        future_exog = pd.DataFrame(index=future_dates)
        
        for column in vars_macro:
            future_exog[column] = future_macro_vars[column].iloc[-1]
        
        future_exog_scaled = scaler.transform(future_exog)
        
        forecast_log = model.forecast(steps=forecast_periods, exog=future_exog_scaled)
        forecast = np.expm1(forecast_log)

        print("La predicción de SARIMAX fue exitosa")
        
        return pd.DataFrame({
            'fecha': future_dates,
            'prediccion': forecast
        })

    def _last_real_year(self, atypical_years=None):
        """Último año real disponible en self.climate_hist_annual (cargado
        íntegro en __init__, sin truncar por cutoff), respetando el mismo
        filtro de atypical_years que predict_future_climate aplica
        internamente antes de determinar last_hist_year."""
        df = self.climate_hist_annual
        if atypical_years:
            df = df[~df['year'].isin(atypical_years)]
        if df.empty:
            raise ValueError("No hay histórico climático real disponible (tras filtrar atypical_years).")
        return int(df['year'].max())

    def predict_future_climate(
        self,
        forecast_years=1,
        n_years_clim=10,
        n_years_shape=2,
        atypical_years=None,
        apply_corrector=False,
        cutoff_year: int | None = None,
        enso_phase_by_month: dict[str, str] | None = None,
    ):
        """
        Predicción futura del modelo id=4 (clima + ONI).

        Replica la estructura secuencial anual + mensual de
        `predict_future_xgrf`, pero SIN requerir variables macro futuras
        suministradas por el usuario: el clima y el ONI futuros se generan
        internamente a partir de `self.climate_hist_monthly` /
        `self.climate_hist_annual` (histórico real cargado en __init__),
        usando climatología estacional simple (promedio histórico del mismo
        mes de los últimos `n_years_clim` años) para temperatura/humedad/
        viento/lluvia/días calurosos, y `features.project_oni_future` para
        el ONI.

        Si `cutoff_year` se indica, el histórico se recorta a
        `year <= cutoff_year` antes de proyectar. Así, para predecir el
        año Y del usuario basta pasar `cutoff_year=Y-1` (hindcast cuando
        Y está en el pasado; forecast normal cuando Y está después de
        los datos disponibles). Los años emitidos son
        `[cutoff_year+1 .. cutoff_year+forecast_years]` con etiquetas
        reales (sin remapear).

        El reparto mensual (participación de cada mes en el total del año) NO
        se predice con un modelo -- se calcula UNA SOLA VEZ como el promedio
        histórico real de `demand_percentage` de cada mes calendario sobre
        los últimos `n_years_shape` años completos disponibles (relativo a
        `cutoff_year`), y se aplica igual a TODOS los años pronosticados. Se
        abandonó un XGBoost recursivo (con `demand_percentage_lag_1`/`lag_12`
        encadenados año a año) porque, una vez que el clima/ONI convergen a
        climatología pura (~2 años tras el cutoff), la recursión deriva de
        forma no autocorregible (mayo pasaba de 8.51% a 8.22% mientras junio
        se quedaba fijo en ~8.49%, invirtiendo la relación mayo>junio
        cumplida en 14 de los últimos 15 años reales). El nivel/tendencia
        anual lo sigue aportando el modelo Ridge (paso 3); el resultado se
        normaliza post-hoc para que la suma mensual del año cuadre
        exactamente con el total anual predicho.

        Si `enso_phase_by_month` es un dict no vacío con claves `YYYY-MM` y
        valores NORMAL/NIÑO/NIÑA, (A) sustituye `oni_actual` proyectado por
        la mediana histórica de esa fase (`ENSO_PHASE_ONI`) en esos meses
        -- lo cual alimenta `oni_mean_anio` del Ridge anual -- y (B) para
        esos meses usa un share condicional a la misma fase ONI en el
        histórico -- shrinkage LEVE ponderado por tamaño de muestra hacia
        el share incondicional seguido de PAVA (pool-adjacent-violators)
        ponderado por `n`, que garantiza estructuralmente el orden físico
        NIÑA <= NORMAL <= NIÑO entre las 3 fases del mismo mes (ver
        `_phase_conditional_month_share` / `_isotonic_phase_month_shares` /
        `PHASE_SHARE_LIGHT_SHRINKAGE_K`) --, renormalizando a 1.0 por año de
        pronóstico. Con None/{} el comportamiento es idéntico al anterior.

        El corrector post-predicción (`postproc_corrector_v1.json`) se
        calcula SIEMPRE para cada mes (columnas `prediccion_cruda` /
        `prediccion_ajustada` del DataFrame de resultados, para trazabilidad
        y comparación), pero por defecto NO se usa para la predicción
        principal (`prediccion` / `annual_pred`) -- validado con datos
        reales (backtest retrospectivo 2025) que el corrector, incluso ya
        recalibrado y sin las contaminaciones encontradas, no mejora sobre
        la predicción cruda (MAPE ajustado 2.32% vs. crudo 2.10%; ver
        `training/forecast_climate/artifacts/reporte_decision.md`,
        secciones "Corrección post-revisión 3" y "Experimento (rechazado):
        calibrar con walk-forward genuino"). Pasar `apply_corrector=True`
        para usar la predicción ajustada como principal si se recalibra en
        el futuro con más datos y se confirma que sí ayuda.

        Returns:
            (annual_pred, results): annual_pred es un np.ndarray de forma
            (forecast_years, 1) con el total anual predicho (suma de los 12
            meses de `prediccion`, cruda o ajustada según `apply_corrector`).
            results es un DataFrame con columnas ['fecha', 'prediccion',
            'prediccion_cruda', 'prediccion_ajustada'].
        """
        if climate_features is None or climate_apply_correctors is None:
            raise ValueError(
                "El soporte del modelo climático (training.forecast_climate) no está disponible en este entorno."
            )

        try:
            enso_override = enso_phase_by_month or {}
            use_enso_override = bool(enso_override)

            hist_monthly = self.climate_hist_monthly.copy().sort_values(['year', 'month']).reset_index(drop=True)
            hist_annual = self.climate_hist_annual.copy().sort_values('year').reset_index(drop=True)

            if atypical_years:
                hist_annual = hist_annual[~hist_annual['year'].isin(atypical_years)].reset_index(drop=True)
                hist_monthly = hist_monthly[~hist_monthly['year'].isin(atypical_years)].reset_index(drop=True)

            if cutoff_year is not None:
                hist_annual = hist_annual[hist_annual['year'] <= cutoff_year].reset_index(drop=True)
                hist_monthly = hist_monthly[hist_monthly['year'] <= cutoff_year].reset_index(drop=True)
                if hist_annual.empty or hist_monthly.empty:
                    raise ValueError(
                        f"No hay histórico climático con year <= cutoff_year={cutoff_year}. "
                        f"No se puede generar el pronóstico solicitado."
                    )

            last_hist_year = int(hist_annual['year'].max())

            # El último mes real disponible puede no ser diciembre (año en
            # curso incompleto). Los años objetivo del pronóstico anual son
            # siempre [last_hist_year+1 .. last_hist_year+forecast_years]
            # completos (12 meses cada uno), así que hay que proyectar
            # primero los meses "de relleno" que faltan para completar
            # last_hist_year antes de contar los forecast_years*12 meses
            # objetivo -- si no, project_oni_future (que arranca justo
            # después del último dato real) desalinea el último año
            # pronosticado con menos de 12 meses.
            last_hist_row = hist_monthly.iloc[-1]
            last_actual_month = int(last_hist_row['month'])
            filler_months = 12 - last_actual_month if last_actual_month < 12 else 0
            n_months_future = filler_months + forecast_years * 12

            # ===== 1. Proyectar ONI futuro (sin supuestos manuales) =====
            oni_future = climate_features.project_oni_future(
                hist_monthly, meses_futuros=n_months_future,
                columna_oni='oni_actual', year_col='year', month_col='month',
                n_years_clim=n_years_clim,
            )

            # ===== 2. Climatología estacional mensual (temp/humedad/viento/lluvia/dias_calurosos) =====
            def climatologia_mensual(col, mes, hasta_year):
                ventana = hist_monthly[
                    (hist_monthly['month'] == mes)
                    & (hist_monthly['year'] <= hasta_year)
                    & (hist_monthly['year'] > hasta_year - n_years_clim)
                ]
                if ventana.empty:
                    ventana = hist_monthly[hist_monthly['month'] == mes]
                return float(ventana[col].mean())

            climate_cols = ['temp_mean_mes', 'humidity_mean_mes', 'wind_speed_mean_mes',
                             'rain_sum_mes', 'dias_calurosos_mes']
            future_monthly_rows = []
            for _, row in oni_future.iterrows():
                y, m = int(row['year']), int(row['month'])
                oni_val = float(row['oni_proyectado'])
                # Step A: override ONI proyectado con mediana de fase si el
                # cliente forzó NORMAL/NIÑO/NIÑA para este YYYY-MM.
                if use_enso_override:
                    _enso_key = f"{y}-{m:02d}"
                    if _enso_key in enso_override:
                        oni_val = float(ENSO_PHASE_ONI[enso_override[_enso_key]])
                row_data = {'year': y, 'month': m, 'oni_actual': oni_val}
                for col in climate_cols:
                    row_data[col] = climatologia_mensual(col, m, last_hist_year)
                future_monthly_rows.append(row_data)
            future_monthly_df = pd.DataFrame(future_monthly_rows)

            # delta_temp_interanual futuro: temp(mes,y) - temp(mismo mes, y-1),
            # usando histórico + meses futuros ya proyectados (secuencial).
            combined_temp = pd.concat([
                hist_monthly[['year', 'month', 'temp_mean_mes']],
                future_monthly_df[['year', 'month', 'temp_mean_mes']],
            ], ignore_index=True)
            temp_prev_year = combined_temp.set_index(['year', 'month'])['temp_mean_mes']

            def delta_temp_interanual(y, m, temp_actual):
                prev = temp_prev_year.get((y - 1, m))
                if prev is None or pd.isna(prev):
                    return 0.0
                return float(temp_actual - prev)

            future_monthly_df['delta_temp_interanual'] = future_monthly_df.apply(
                lambda r: delta_temp_interanual(r['year'], r['month'], r['temp_mean_mes']), axis=1
            )

            # oni_lag_1 / oni_lag_3 secuenciales, combinando histórico + proyectado
            combined_oni = pd.concat([
                hist_monthly[['year', 'month', 'oni_actual']],
                future_monthly_df[['year', 'month', 'oni_actual']],
            ], ignore_index=True).sort_values(['year', 'month']).reset_index(drop=True)
            combined_oni['oni_lag_1'] = combined_oni['oni_actual'].shift(1)
            combined_oni['oni_lag_3'] = combined_oni['oni_actual'].shift(3)
            future_monthly_df = future_monthly_df.merge(
                combined_oni[['year', 'month', 'oni_lag_1', 'oni_lag_3']],
                on=['year', 'month'], how='left'
            )

            # mes_sin / mes_cos (mismo criterio que el resto del sistema)
            future_monthly_df['mes_sin'] = np.sin(2 * np.pi * future_monthly_df['month'] / 12)
            future_monthly_df['mes_cos'] = np.cos(2 * np.pi * future_monthly_df['month'] / 12)

            # ===== 3. Predicción anual secuencial (Ridge sobre log1p(demand_total)) =====
            annual_feature_cols = ['temp_mean_anio', 'temp_max_anio', 'temp_min_anio',
                                    'humidity_mean_anio', 'wind_speed_mean_anio', 'rain_sum_anio']

            # El último año del histórico puede estar incompleto (año en
            # curso, con menos de 12 meses de datos reales). Usar su total
            # crudo como lag_1/lag_2 distorsionaría gravemente la secuencia
            # (se vería como una caída brusca de demanda). Se anualiza ese
            # total parcial (total_parcial / (meses_disponibles/12)) solo
            # para efectos de los lags/growth del modelo anual; el resto de
            # la lógica (proyección mensual, climatología) sigue usando los
            # meses reales tal cual.
            months_per_year = hist_monthly.groupby('year')['month'].nunique()
            hist_annual_adj = hist_annual.copy()
            for idx, row in hist_annual_adj.iterrows():
                y = int(row['year'])
                n_months = int(months_per_year.get(y, 12))
                if n_months < 12 and n_months > 0:
                    hist_annual_adj.loc[idx, 'demand_total'] = row['demand_total'] * (12.0 / n_months)

            demand_log_hist = np.log1p(hist_annual_adj['demand_total']).tolist()
            last_log_1 = demand_log_hist[-1]
            last_log_2 = demand_log_hist[-2] if len(demand_log_hist) > 1 else demand_log_hist[-1]
            # Seed growth_rate_lag1 with the mean of the last up to 3 log-growths
            # (when history allows). A single last step can be a post-boom spike;
            # growth_rate_lag1 has a negative Ridge coef, so that spike mean-reverts
            # into systematic overprediction. Averaging dampens that.
            if len(demand_log_hist) >= 4:
                last_growth = float(np.mean([
                    demand_log_hist[-1] - demand_log_hist[-2],
                    demand_log_hist[-2] - demand_log_hist[-3],
                    demand_log_hist[-3] - demand_log_hist[-4],
                ]))
            elif len(demand_log_hist) > 1:
                last_growth = demand_log_hist[-1] - demand_log_hist[-2]
            else:
                last_growth = 0.0

            # Denominator matches train_candidates.prepare_annual_features:
            # year_index = 0..n-1 on hist, year_index_norm = year_index / max(n-1, 1)
            # so the last historical year is 1.0.
            year_index_norm_denom = max(len(hist_annual) - 1, 1)
            annual_predictions_raw = []

            for i in range(forecast_years):
                target_year = last_hist_year + 1 + i
                months_this_year = future_monthly_df[future_monthly_df['year'] == target_year]

                annual_row = {}
                # temp/humidity/wind_speed son promedios mensuales -> el promedio
                # de los 12 promedios mensuales aproxima bien el promedio anual.
                # rain_sum_mes es una SUMA mensual (mm acumulados del mes), así
                # que la agregación a anual debe ser una SUMA de los 12 meses,
                # no un promedio -- usar .mean() aquí entendía "rain_sum_anio"
                # como ~1/12 de su valor real (bug: dejaba esa feature ~12x por
                # debajo de la escala vista en entrenamiento, sesgando fuerte
                # el nivel anual predicho por Ridge aunque el reparto mensual
                # de XGBoost, que no usa esta feature, quedaba correcto).
                for base_col, mes_col in zip(
                    ['temp_mean_anio', 'humidity_mean_anio', 'wind_speed_mean_anio'],
                    ['temp_mean_mes', 'humidity_mean_mes', 'wind_speed_mean_mes'],
                ):
                    if not months_this_year.empty:
                        annual_row[base_col] = float(months_this_year[mes_col].mean())
                    else:
                        annual_row[base_col] = float(hist_annual[base_col].tail(n_years_clim).mean())

                if not months_this_year.empty:
                    annual_row['rain_sum_anio'] = float(months_this_year['rain_sum_mes'].sum())
                else:
                    annual_row['rain_sum_anio'] = float(hist_annual['rain_sum_anio'].tail(n_years_clim).mean())

                # temp_max_anio / temp_min_anio: no hay proyección mensual de
                # max/min, se usa climatología anual (promedio de los últimos
                # n_years_clim años reales).
                annual_row['temp_max_anio'] = float(hist_annual['temp_max_anio'].tail(n_years_clim).mean())
                annual_row['temp_min_anio'] = float(hist_annual['temp_min_anio'].tail(n_years_clim).mean())

                annual_row['oni_mean_anio'] = (
                    float(months_this_year['oni_actual'].mean()) if not months_this_year.empty
                    else float(hist_annual['oni_mean_anio'].tail(n_years_clim).mean())
                )

                annual_row['lag_1'] = last_log_1
                annual_row['lag_2'] = last_log_2
                # Freeze first forecast year at the last historical index (n-1 →
                # norm=1.0). Using n+i extrapolated year_index_norm > 1 and caused
                # ~+3% overprediction. Second forecast year gets n, etc.
                year_index = (len(hist_annual) - 1) + i
                annual_row['year_index'] = year_index
                annual_row['year_index_norm'] = year_index / year_index_norm_denom
                annual_row['growth_rate_lag1'] = last_growth

                X_annual = pd.DataFrame([annual_row])[self.climate_annual_features]
                X_annual_scaled = self.climate_annual_scaler_x.transform(X_annual)
                annual_pred_scaled = self.climate_annual_model.predict(X_annual_scaled)
                annual_pred_value = np.expm1(
                    self.climate_annual_scaler_y.inverse_transform(annual_pred_scaled.reshape(-1, 1))
                )[0][0]
                annual_pred_value = max(float(annual_pred_value), 0.0)

                annual_predictions_raw.append(annual_pred_value)

                current_log_pred = np.log1p(annual_pred_value)
                last_growth = current_log_pred - last_log_1
                last_log_2 = last_log_1
                last_log_1 = current_log_pred

            # ===== 4. Reparto mensual: promedio histórico real de los últimos
            # n_years_shape años completos por mes calendario (sin modelo, sin
            # recursión). El total anual (paso 3) sigue variando por año; solo el
            # reparto entre los 12 meses se mantiene fijo, calculado UNA VEZ fuera
            # del loop de años -- esto es lo que elimina la deriva año-a-año que
            # tenía el XGBoost recursivo (ver plan de la sesión: mayo pasaba de
            # 8.51% a 8.22% mientras junio se quedaba fijo en ~8.49%, invirtiendo
            # la relación mayo>junio que se cumple en 14 de los últimos 15 años
            # reales).
            hist_pct_monthly = hist_monthly.copy()
            hist_pct_monthly['yearly_total'] = hist_pct_monthly.groupby('year')['demand_value'].transform('sum')
            hist_pct_monthly['demand_percentage'] = hist_pct_monthly['demand_value'] / hist_pct_monthly['yearly_total']

            _month_counts = hist_pct_monthly.groupby('year')['month'].transform('count')
            full_years_pct = hist_pct_monthly.loc[_month_counts == 12]
            if full_years_pct.empty:
                full_years_pct = hist_pct_monthly

            full_years_sorted = sorted(full_years_pct['year'].unique())
            recent_full_years = full_years_sorted[-n_years_shape:] if full_years_sorted else []
            shape_src = full_years_pct[full_years_pct['year'].isin(recent_full_years)] if recent_full_years else full_years_pct

            month_avg_share = {}
            for _m in range(1, 13):
                _series = shape_src.loc[shape_src['month'] == _m, 'demand_percentage']
                if len(_series) == 0:
                    _series = hist_pct_monthly.loc[hist_pct_monthly['month'] == _m, 'demand_percentage']
                month_avg_share[_m] = float(_series.mean()) if len(_series) else 1.0 / 12

            # Fuente para shares condicionales a fase ENSO: todos los años
            # completos (no solo n_years_shape), para maximizar muestras por
            # (mes, fase). El share incondicional sigue usando shape_src.
            phase_shape_src = full_years_pct if 'oni_actual' in full_years_pct.columns else hist_pct_monthly

            monthly_records = []
            for i in range(forecast_years):
                target_year = last_hist_year + 1 + i
                year_total_raw = annual_predictions_raw[i]

                if use_enso_override:
                    # Step B: share condicional por fase cuando hay override
                    # para ese YYYY-MM; si no, share incondicional. Luego
                    # renormalizar a 1.0 dentro del año (mezcla de fuentes).
                    year_shares = {}
                    for m in range(1, 13):
                        _enso_key = f"{target_year}-{m:02d}"
                        if _enso_key in enso_override:
                            year_shares[m] = _phase_conditional_month_share(
                                phase_shape_src,
                                month=m,
                                phase=enso_override[_enso_key],
                                fallback=month_avg_share[m],
                            )
                        else:
                            year_shares[m] = month_avg_share[m]
                    share_sum = sum(year_shares.values())
                    if share_sum > 0:
                        year_shares = {m: s / share_sum for m, s in year_shares.items()}
                    else:
                        year_shares = {m: 1.0 / 12 for m in range(1, 13)}
                    for m in range(1, 13):
                        monthly_records.append({
                            'year': target_year, 'month': m,
                            'demand_value_cruda': year_shares[m] * year_total_raw,
                        })
                else:
                    for m in range(1, 13):
                        monthly_records.append({
                            'year': target_year, 'month': m,
                            'demand_value_cruda': month_avg_share[m] * year_total_raw,
                        })

            monthly_df = pd.DataFrame(monthly_records)

            # ===== 5. Normalizar para que la suma mensual cuadre con el total anual =====
            # Tras el clip+renorm del paso 4, demand_value_cruda ya suma
            # year_total_raw (scale ≈ 1); se conserva por coherencia.
            monthly_df['prediccion_cruda'] = monthly_df['demand_value_cruda']
            for i in range(forecast_years):
                target_year = last_hist_year + 1 + i
                mask = monthly_df['year'] == target_year
                year_total_raw_sum = monthly_df.loc[mask, 'demand_value_cruda'].sum()
                year_total_model = annual_predictions_raw[i]
                if year_total_raw_sum > 0:
                    scale = year_total_model / year_total_raw_sum
                else:
                    scale = 1.0
                monthly_df.loc[mask, 'prediccion_cruda'] = monthly_df.loc[mask, 'demand_value_cruda'] * scale

            # ===== 6. Aplicar corrector post-predicción (trazabilidad cruda vs ajustada) =====
            prediccion_ajustada = []
            prediccion_final_col = []
            if self.climate_corrector is not None:
                bias_corrector = self.climate_corrector['bias_corrector']
                seasonal_corrector = self.climate_corrector['seasonal_corrector']
                for _, mrow in monthly_df.iterrows():
                    resultado = climate_apply_correctors(
                        prediccion_cruda=mrow['prediccion_cruda'],
                        mes=mrow['month'],
                        bias_corrector=bias_corrector,
                        seasonal_corrector=seasonal_corrector,
                    )
                    prediccion_ajustada.append(resultado['prediccion_final'])
            else:
                print("[climate_model] Corrector post-predicción no disponible; se usa la predicción cruda sin ajustar.")
                prediccion_ajustada = monthly_df['prediccion_cruda'].tolist()

            monthly_df['prediccion_ajustada'] = prediccion_ajustada
            # Por defecto se sirve la predicción CRUDA (validado con datos
            # reales: el corrector no mejora sobre la cruda, ver docstring).
            # `prediccion_ajustada` queda calculada igual para trazabilidad.
            monthly_df['prediccion'] = (
                monthly_df['prediccion_ajustada'] if apply_corrector else monthly_df['prediccion_cruda']
            )

            monthly_df['fecha'] = pd.to_datetime(
                monthly_df['year'].astype(str) + '-' + monthly_df['month'].astype(str) + '-01'
            )
            monthly_df = monthly_df.sort_values('fecha').reset_index(drop=True)

            results = monthly_df[['fecha', 'prediccion', 'prediccion_cruda', 'prediccion_ajustada']]

            # Total anual final (suma de los 12 meses de `prediccion`, sea
            # cruda o ajustada segun `apply_corrector`), para mantener
            # coherencia entre annual_pred y monthly results.
            annual_pred_final = []
            for i in range(forecast_years):
                target_year = last_hist_year + 1 + i
                total_year = monthly_df.loc[monthly_df['year'] == target_year, 'prediccion'].sum()
                annual_pred_final.append(total_year)

            annual_pred = np.array(annual_pred_final).reshape(-1, 1)

            return annual_pred, results

        except Exception as e:
            print("Error durante la prediccion del modelo climatico (id=4)", str(e))
            raise e

def check_current_year(start_date, end_date):
    """
    Verifica si la fecha actual está dentro del rango de fechas.
    Si esta dentro del rango levanta un error
    """
    current_date = pd.to_datetime('today').date()
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    if start_date <= current_date <= end_date:
        raise ValueError("La fecha actual está dentro del rango de fechas. No se puede seguir con la predicción.")


#Start of endpoints 
class ForecastService:
    def predict(self, search: ForecastSchema, db: Session):
        try:
            # Initialize updated_variables early to avoid reference errors in exception handlers
            updated_variables = {}
            
            demand = DemandService()
            start_date = pd.to_datetime(search.start_date)
            end_date = pd.to_datetime(search.end_date)

            # Query monthly demand data up to the start date year
            # Query monthly demand data up to the start date (using both year and month)
            historic_demands = db.query(MonthlyDemand.value, MonthlyDemand.year, MonthlyDemand.month)\
                .filter(
                    # Exclude data from after start date (including specific month)
                    ((MonthlyDemand.year < start_date.year) |
                     ((MonthlyDemand.year == start_date.year) & (MonthlyDemand.month < start_date.month)))
                )\
                .order_by(MonthlyDemand.year, MonthlyDemand.month)\
                .all()
                
            # Create separate arrays for demands and dates while preserving their order
            demand_values_for_table = []
            date_values_for_table = []

            for record in historic_demands:
                demand_values_for_table.append(record.value)
                # Format date as 'YYYY-MM-01'
                date_values_for_table.append(f"{record.year}-{record.month:02d}-01")

  
            historic_demands_for_table = historic_demands


            # CAMBIO 1: Eliminamos la validación de fecha inicial
            # Ya no verificamos si start_date > last_date_ym
            
            # CAMBIO 2: Calcular la fecha límite para datos históricos (último día del mes anterior)
            # Si start_date es 2025-01-01, queremos 2024-12-31 (último día de diciembre)
            historical_limit_date = start_date.replace(day=1) - pd.DateOffset(days=1)
            historical_limit_date_str = historical_limit_date.strftime('%Y-%m-%d')
            
            # Obtener el rango completo de datos disponibles para determinar el inicio
            year_range = demand.get_first_and_last_date(db)
            print("year_range", year_range)
            first_date = year_range["first_date"].strftime('%Y-%m-%d')
            # Usamos la fecha límite calculada en lugar de la última fecha disponible
            last_date = historical_limit_date_str
            first_year = int(pd.to_datetime(first_date).year)
            last_year = int(pd.to_datetime(last_date).year)
            
            print(f"Usando datos históricos desde {first_date} hasta {last_date}")
            print(f"Predicción solicitada desde {search.start_date} hasta {search.end_date}")
            
            # forecast_years/years_list/start_year no dependen de las variables
            # macro: se calculan siempre, incluso para el modelo climático
            # (id=4), que no necesita variables macro futuras.
            if end_date.to_period('M') < start_date.to_period('M'):
                raise ValueError("end_date debe ser posterior o igual a start_date")

            # Modelos v2 (id 1/2/3) no usan variables macro; id 4 tampoco. Solo el camino legacy las pide.
            use_v2 = search.id in (1, 2, 3) and not _use_legacy_models()
            # División de mercado de la base de la petición (misma fuente y semántica que el front).
            split_config = DemandConfigService().get_config(db)
            division = division_from_config(split_config)
            model_info = {"id": int(search.id), "version": "v2" if (use_v2 or search.id == 4) else "legacy"}
            v2_estimated_years = []
            # Rama v2: parte de cada total anual pedido que son reales previos a start (no la escala el IPC).
            v2_fixed_by_year = None
            v2_start_m = None

            print(f"Calculando años de pronóstico. start_date: {start_date}, end_date: {end_date}")
            forecast_years = end_date.year - start_date.year + 1
            years_list = []
            for i in range(forecast_years):
                years_list.append(start_date.year + i)
            start_year = min(years_list)
            print(f"Años a pronosticar: {years_list}, start_year: {start_year}")

            # cutoff_year: para predecir los años del usuario solo se usa
            # histórico con year <= start_year - 1 (hindcast / forecast
            # honesto; sin remapeo de etiquetas).
            cutoff_year = start_year - 1

            # Define the mapping between API variable names and model feature names
            macroeconomic_variables = {
                'ipc': 'ipc',
                'Crecimiento': 'Crecimiento de la población (%) anual',
                'Inflación': 'Inflación, deflactor del PIB: series vinculadas (% anual)',
                'Comercio': 'Comercio de mercaderías (% del PIB)',
                'Importaciones': 'Importaciones de bienes (balanza de pagos, US$ a precios actuales)',
                'Empleo': 'Empleo de tiempo parcial, total (% del total de empleo)'
            }

            # El modelo climático (id=4) NO requiere variables macroeconómicas
            # futuras suministradas por el usuario (genera su propio clima/ONI
            # futuro internamente) -- se omite todo este bloque para ese caso,
            # sin alterar el comportamiento para id=1/2/3.

            if search.id != 4 and not use_v2:
                try:
                    print("Creando MacroeconomicService...")
                    macroeconomic = MacroeconomicService()

                    # Crear esquema de búsqueda
                    print(f"Creando esquema de búsqueda con ano_inicio={start_year}, ano_fin={end_date.year}")
                    try:
                        macro_Search = SearchAllMacroeconomicSchema(
                            ano_inicio=str(start_year),
                            ano_fin=str(end_date.year)
                        )
                        print(f"Esquema creado: {macro_Search}")
                    except Exception as e:
                        print(f"Error al crear el esquema SearchAllMacroeconomicSchema: {str(e)}")
                        return {"message": f"Error al crear esquema de búsqueda: {str(e)}", "status": "error"}

                    # Obtener variables macroeconómicas
                    print("Obteniendo variables macroeconómicas...")
                    macro_variables = macroeconomic.get_all_variables(db, macro_Search)

                    # Verificar si hay variables disponibles
                    if not macro_variables:
                        return {
                            "message": f"No hay variables macroeconómicas disponibles para el período {start_year} - {end_date.year}. Por favor, utilice un rango de fechas para el que existan datos históricos completos.",
                            "status": "error"
                        }
                    print("macro_variables", macro_variables)

                except Exception as e:
                    print(f"Error inesperado en el bloque de variables macroeconómicas: {str(e)}")
                    return {"message": f"Error al procesar variables macroeconómicas: {str(e)}", "status": "error"}

                # Verificar que todas las variables necesarias existan para TODOS los años
                missing_vars = []
                for api_name, model_name in macroeconomic_variables.items():
                    print(f"Verificando variable: {api_name} -> {model_name}")

                    # Verificar si la variable existe
                    if api_name not in macro_variables:
                        print(f"Error: La variable macroeconómica '{api_name}' no fue encontrada.")
                        missing_vars.append(api_name)
                        continue

                    # Verificar que haya suficientes valores para cada año
                    variable_values = macro_variables[api_name]
                    if len(variable_values) < len(years_list):
                        print(f"Error: La variable '{api_name}' no tiene datos para todos los años requeridos.")
                        print(f"  - Años requeridos: {len(years_list)}, Datos disponibles: {len(variable_values)}")
                        missing_vars.append(f"{api_name} (faltan datos para algunos años)")
                        continue

                    # Guardar los valores sin rellenar
                    updated_variables[model_name] = variable_values[:len(years_list)]

                # Si falta alguna variable, no continuar con la predicción
                if missing_vars:
                    return {
                        "message": f"No se puede realizar la predicción porque faltan las siguientes variables macroeconómicas o no están completas para todo el período: {', '.join(missing_vars)}. Por favor, complete estos datos antes de continuar.",
                        "status": "error"
                    }

                print("updated_variables", updated_variables)

                # Verificar que updated_variables tenga todas las variables necesarias
                expected_vars = set(macroeconomic_variables.values())
                actual_vars = set(updated_variables.keys())
                if expected_vars != actual_vars:
                    missing = expected_vars - actual_vars
                    return {
                        "message": f"Faltan las siguientes variables macroeconómicas requeridas: {', '.join(missing)}",
                        "status": "error"
                    }

            print("Creando esquemas de búsqueda para datos históricos...")
            try:
                search_schema_annual = DemandSearchSchema(
                    tipo=3,
                    ano_inicio=str(first_year),
                    ano_fin=str(last_year)
                )
                search_schema_monthly = DemandSearchSchema(
                    tipo=2,
                    fecha_inicio=str(first_date),
                    fecha_fin=str(last_date)
                )
                print("Esquemas creados correctamente")
            except Exception as e:
                print(f"Error al crear los esquemas de búsqueda: {str(e)}")
                return {"message": f"Error al preparar los esquemas de búsqueda: {str(e)}", "status": "error"}
                
            try:
                print("Obteniendo años atípicos...")
                atypical_years = db.query(TypeYear.year).filter_by(type="ATIPICO").all()
                atypical_years = [year[0] for year in atypical_years]
                print("atypical_years", atypical_years)
            except Exception as e:
                print(f"Error al obtener años atípicos: {str(e)}")
                return {"message": f"Error al obtener años atípicos: {str(e)}", "status": "error"}
            
            # Obtener datos anuales y mensuales
            try:
                print("Obteniendo datos anuales históricos...")
                annual_data_raw = demand.get_demands(db, search_schema_annual)
                print(f"Datos anuales recibidos: {type(annual_data_raw)}")

                print("Obteniendo datos mensuales históricos...")
                monthly_data_raw = demand.get_demands(db, search_schema_monthly)
                print(f"Datos mensuales recibidos: {type(monthly_data_raw)}")
            except Exception as e:
                print(f"Error al obtener datos históricos: {str(e)}")
                return {"message": f"Error al obtener datos históricos: {str(e)}", "status": "error"}
            
            # Verificar que los datos no estén vacíos
            if not annual_data_raw or not annual_data_raw.get("lista_demandas") or not annual_data_raw.get("lista_fechas"):
                raise ValueError("No se encontraron datos anuales para el periodo solicitado")
            
            if not monthly_data_raw or not monthly_data_raw.get("lista_demandas") or not monthly_data_raw.get("lista_fechas"):
                raise ValueError("No se encontraron datos mensuales para el periodo solicitado")
            
            # Formatear datos anuales
            annual_data = pd.DataFrame({
                'total': annual_data_raw["lista_demandas"],
                'ano': annual_data_raw["lista_fechas"]
            }, index=annual_data_raw["lista_fechas"])
            
            # Formatear datos mensuales
            dates = pd.to_datetime(monthly_data_raw["lista_fechas"])
            monthly_data = pd.DataFrame({
                'ano': dates.year,
                'mes': dates.month,
                'total': monthly_data_raw["lista_demandas"],
                'fecha': dates
            })
            
            print("\nRevisión de datos cargados:")
            print("Datos anuales:", annual_data.head())
            print("Datos mensuales:", monthly_data.head())
            
            # Asegurarse que los datos mensuales tengan el formato correcto
            monthly_data['total'] = monthly_data['total'].astype(float)

            print("years_list", years_list)
            

            # Preparar totales anuales historicos ordenados (anio mas viejo primero)
            historical_annual_totals = []
            try:
                historical_years = [int(y) for y in annual_data_raw["lista_fechas"]]
                historical_totals = annual_data_raw["lista_demandas"]
                historical_pairs = sorted(zip(historical_years, historical_totals), key=lambda x: x[0])
                historical_annual_totals = [float(total) for _, total in historical_pairs]
            except Exception as e:
                print(f"Error al preparar totales anuales historicos: {str(e)}")
                historical_annual_totals = [float(total) for total in annual_data_raw.get("lista_demandas", [])]

            # Crear DataFrame para variables macroeconómicas futuras
            # (id=4 no las necesita: se deja None, no se usa en esa rama)
            future_macro_vars = None
            _SARIMAX_MACRO_FALLBACK = False
            if search.id != 4 and not use_v2:
                future_macro_vars = pd.DataFrame(
                    updated_variables,
                    index=years_list
                )
                if search.id == 3:
                    # Como el original (GMR Atenea): SARIMAX usa la macro del
                    # último año conocido sostenida constante (api/data/
                    # gmr_annual_es.csv) en vez de las proyecciones de la BD.
                    # Si falla, se queda con las de la BD y se avisa en la respuesta.
                    try:
                        future_macro_vars = sarimax_macro_constante.build_future_macro_vars(
                            n_years=len(years_list), start_year=start_year
                        )
                    except Exception as e:
                        print(f"[SARIMAX] AVISO: no se pudo construir la macro constante, "
                              f"usando proyecciones de BD: {str(e)}")
                        _SARIMAX_MACRO_FALLBACK = True

            yearly_demands = db.query(YearlyDemand).all()
            years = {}
            for yearly_demand in yearly_demands:  # Cambié 'demand' por 'yearly_demand'
                years[yearly_demand.year] = yearly_demand.demand

            # IMPORTANTE: Guardar los años que realmente queremos predecir
            years_to_predict = years_list.copy()
            
            # Guardar el número de años que queremos predecir ANTES de llamar al modelo
            num_years_to_predict = len(years_to_predict)

            if use_v2:
                # Modelos v2 (ver docs/work/02_rfxgb y 03_sarimax): historia limpia de la base de la
                # petición, cortada en la división de mercado, pronóstico en escala de mercado completo.
                # Totales anuales: reales previos a start (+ pronóstico de huecos internos) + pronóstico del
                # resto; años previos incompletos se completan con pronóstico solo para `growth` (no se toca
                # annual_predictions histórico de la BD). Ver run_v2_forecast / v2_annual_totals.
                hist = load_clean_history_session(db, division)
                v2_start_m = start_date.to_period('M').to_timestamp()
                v2 = run_v2_forecast(search.id, hist.serie, hist.corte, v2_start_m, years_to_predict, years)
                model_info.update(v2["info"])
                years = v2["years"]
                v2_fixed_by_year = v2["fixed_by_year"]
                v2_estimated_years.extend(v2["estimated_years"])
                fc = v2["fc"]
                monthly_predictions = pd.DataFrame({'fecha': fc.index, 'prediccion': fc.to_numpy()})

                # (id 1/2: apply_ipc_adjustment lo recalcula más abajo sobre los totales ajustados)
                years_list_full = sorted(list(years.keys()))
                growth = {}
                for i in range(1, len(years_list_full)):
                    prev_y, cur_y = years_list_full[i - 1], years_list_full[i]
                    if years.get(prev_y):
                        growth[cur_y] = (years[cur_y] - years[prev_y]) / years[prev_y]

            elif search.id == 1 or search.id == 2:
                if search.id == 1:
                    model_path_name = 'api/models/rf_model/'
                else:   
                    model_path_name = 'api/models/xgb_model/'

                predictor = DemandPredictor(model_path=model_path_name)
            
                # Realizar predicción
                annual_predictions, monthly_predictions = predictor.predict_future_xgrf(
                    historical_annual_data=annual_data,
                    historical_monthly_data=monthly_data,
                    forecast_years=num_years_to_predict,  # Usar el número guardado
                    future_macro_vars=future_macro_vars, 
                    atypical_years=atypical_years
                )
                
                # VALIDACIÓN: Verificar que tengamos el número correcto de predicciones
                if len(annual_predictions) != num_years_to_predict:
                    print(f"ADVERTENCIA: Se esperaban {num_years_to_predict} predicciones pero se obtuvieron {len(annual_predictions)}")
                    # Si hay menos predicciones, podríamos necesitar ajustar o lanzar un error
                    if len(annual_predictions) < num_years_to_predict:
                        raise ValueError(f"El modelo devolvió menos predicciones ({len(annual_predictions)}) que los años solicitados ({num_years_to_predict})")

                # Calcular crecimiento usando los años correctos
                for i, year in enumerate(years_to_predict):
                    if i < len(annual_predictions):
                        years[year] = int(annual_predictions[i])
                    else:
                        print(f"ADVERTENCIA: No hay predicción para el año {year}")
                        
                print("years", years)

                years_list_full = sorted(list(years.keys()))
                print("years_list_full", years_list_full)
                growth = {}
                for i in range(1, len(years_list_full)):
                    if years_list_full[i-1] in years and years_list_full[i] in years:
                        growth[years_list_full[i]] = (years[years_list_full[i]] - years[years_list_full[i-1]]) / years[years_list_full[i-1]]
                print("growth", growth)
                
                # Convertir predicciones a lista simple
                if isinstance(annual_predictions, np.ndarray):
                    annual_predictions = annual_predictions.tolist()
                    if len(annual_predictions) > 0 and isinstance(annual_predictions[0], list):
                        annual_predictions = [pred[0] for pred in annual_predictions]
                print("annual_predictions", annual_predictions)

            elif search.id == 3:
                try:
                    model_path_name = 'api/models/sarimax_model/'
                    predictor = DemandPredictor(model_path=model_path_name)
                    
                    # Para SARIMAX, calcular el número de períodos mensuales
                    forecast_periods = num_years_to_predict * 12
                    
                    monthly_predictions = predictor.predict_future_sarimax(
                        monthly_data,
                        future_macro_vars,
                        atypical_years=atypical_years, 
                        forecast_periods=forecast_periods
                    )
                    print("monthly_predictions", monthly_predictions)
                    
                    # Verificar que monthly_predictions contiene los datos esperados
                    if 'prediccion' not in monthly_predictions or 'fecha' not in monthly_predictions:
                        raise ValueError("La predicción SARIMAX no retornó los campos esperados")
                        
                    # Agrupar predicciones mensuales por año
                    annual_predictions = monthly_predictions['prediccion'].groupby(monthly_predictions['fecha'].dt.year).sum().tolist()
                    
                    # VALIDACIÓN: Verificar que tengamos el número correcto de predicciones anuales
                    if len(annual_predictions) != num_years_to_predict:
                        print(f"ADVERTENCIA SARIMAX: Se esperaban {num_years_to_predict} predicciones anuales pero se obtuvieron {len(annual_predictions)}")
                    
                    # Calcular crecimiento
                    for i, year in enumerate(years_to_predict):
                        if i < len(annual_predictions):
                            years[year] = int(annual_predictions[i])
                            
                    years_list_full = sorted(list(years.keys()))
                    growth = {}
                    for i in range(1, len(years_list_full)):
                        if years_list_full[i-1] in years and years_list_full[i] in years:
                            growth[years_list_full[i]] = (years[years_list_full[i]] - years[years_list_full[i-1]]) / years[years_list_full[i-1]]
                    print("growth", growth)

                except Exception as e:
                    print("Error en la predicción SARIMAX", str(e))
                    raise ValueError(f"Error en la predicción SARIMAX: {str(e)}")

            elif search.id == 4:
                try:
                    model_path_name = 'api/models/climate_model/'
                    predictor = DemandPredictor(model_path=model_path_name)

                    # Como el original (GMR Atenea): NO excluir años atípicos (GMR_type_year,
                    # ATIPICO 2020/2021) del histórico que ancla la inferencia: el lag-1 y el
                    # reparto mensual necesitan la serie completa, y quitarlos desfasaba el
                    # resultado. (El ajuste de coeficientes v2 ya omite 2020-2022 por su cuenta:
                    # training/forecast_climate/clima_oni_v2_spec.json, exclude_years.)
                    atypical_years = []

                    # Reparto mensual: promedio de los últimos 2 años completos (default de
                    # predict_future_climate). El parche nys=5 se descartó: empata en el walk-forward total y es
                    # peor en los orígenes posteriores a la selección (ver docs/work/04_clima/REPORTE.md y
                    # docs/work/04_clima/forecast_clima_patch.md).
                    climate_n_years_shape = 2
                    model_info.update({"nombre": "clima_oni_v2", "n_years_shape": climate_n_years_shape})

                    # El histórico climático (CSV) llega hasta last_real_year; el modelo emite los años
                    # cutoff_year+1.. con cutoff_year <= last_real_year. Si el primer año pedido está más allá
                    # de last_real_year+1 las etiquetas no corresponderían a los años pedidos.
                    last_real_year = predictor._last_real_year(atypical_years=atypical_years)
                    if cutoff_year > last_real_year:
                        raise ValueError(
                            f"El modelo climático (id=4) solo puede pronosticar a partir de {last_real_year + 1} "
                            f"(su histórico llega a {last_real_year}); el año de inicio pedido es {start_year}. "
                            f"Use una fecha de inicio <= {last_real_year + 1}-12-31."
                        )

                    # Hindcast/forecast honesto: predecir los años del
                    # usuario usando solo histórico con year <= start_year-1.
                    # Si el CSV local no alcanza el cutoff, no hay forma
                    # de anclar el primer año pedido.

                    # A diferencia de id=1/2/3, no se le pasan datos
                    # históricos ni variables macro futuras: el modelo
                    # climático genera internamente su propio clima/ONI
                    # futuro a partir de su histórico real cargado en
                    # __init__, recortado por cutoff_year.
                    annual_predictions, monthly_predictions = predictor.predict_future_climate(
                        forecast_years=num_years_to_predict,
                        atypical_years=atypical_years,
                        cutoff_year=cutoff_year,
                        enso_phase_by_month=getattr(search, 'enso_phase_by_month', None),
                        n_years_shape=climate_n_years_shape,
                    )

                    # VALIDACIÓN: Verificar que tengamos el número correcto de predicciones
                    if len(annual_predictions) != num_years_to_predict:
                        print(f"ADVERTENCIA: Se esperaban {num_years_to_predict} predicciones pero se obtuvieron {len(annual_predictions)}")
                        if len(annual_predictions) < num_years_to_predict:
                            raise ValueError(f"El modelo devolvió menos predicciones ({len(annual_predictions)}) que los años solicitados ({num_years_to_predict})")

                    # Bug fix: cutoff_year = start_year - 1 descarta histórico
                    # real innecesariamente cuando start_year <= último año
                    # real disponible, contaminando el ancla de los años
                    # genuinamente futuros (>last_real_year) y haciendo que su
                    # predicción varíe según el start_date elegido por el
                    # usuario. Si el rango pedido cruza esa frontera, se hace
                    # una segunda llamada "genuina" anclada en last_real_year
                    # solo para los años > last_real_year, y se empalma su
                    # resultado sobre el de la llamada hindcast de arriba.
                    needs_genuine_forecast_splice = (
                        start_year <= last_real_year and max(years_to_predict) > last_real_year
                    )

                    if needs_genuine_forecast_splice:
                        genuine_forecast_years = max(years_to_predict) - last_real_year
                        genuine_annual_predictions, genuine_monthly_predictions = predictor.predict_future_climate(
                            forecast_years=genuine_forecast_years,
                            atypical_years=atypical_years,
                            cutoff_year=last_real_year,
                            enso_phase_by_month=getattr(search, 'enso_phase_by_month', None),
                            n_years_shape=climate_n_years_shape,
                        )

                    # Calcular crecimiento usando los años correctos (annual_pred es (n, 1): se aplana para
                    # convertir cada total a escalar sin DeprecationWarning de numpy).
                    annual_flat = np.asarray(annual_predictions, dtype=float).ravel()
                    for i, year in enumerate(years_to_predict):
                        if i < len(annual_flat):
                            years[year] = int(annual_flat[i])
                        else:
                            print(f"ADVERTENCIA: No hay predicción para el año {year}")

                    if needs_genuine_forecast_splice:
                        genuine_flat = np.asarray(genuine_annual_predictions, dtype=float).ravel()
                        genuine_years_to_predict = [y for y in years_to_predict if y > last_real_year]
                        for i, year in enumerate(genuine_years_to_predict):
                            if i < len(genuine_flat):
                                years[year] = int(genuine_flat[i])

                        genuine_monthly = genuine_monthly_predictions[['fecha', 'prediccion']].copy()
                        keep_mask = monthly_predictions['fecha'].dt.year <= last_real_year
                        monthly_predictions = pd.concat(
                            [monthly_predictions.loc[keep_mask, ['fecha', 'prediccion']], genuine_monthly],
                            ignore_index=True,
                        ).sort_values('fecha').reset_index(drop=True)

                    # Año previo al pedido incompleto en la base (p. ej. 2026 con sep-oct post-división y
                    # nov preliminar cuando start=2027-01): `years` trae el total parcial de la BD y el
                    # `growth` del primer año pedido salía ~+45 %. Se estima en escala de mercado completo con
                    # los reales limpios observados / participación de esos meses en el reparto del modelo.
                    # Si el año previo no tiene ningún real usable (todo post-división o sin datos), se usa el
                    # pronóstico completo del modelo para ese año (como la rama v2).
                    try:
                        prev_year = start_year - 1
                        if prev_year in years and prev_year not in years_to_predict:
                            hist_c = load_clean_history_session(db, division).serie
                            obs = hist_c[hist_c.index.year == prev_year]
                            if len(obs) == 0:
                                prev_annual, _ = predictor.predict_future_climate(
                                    forecast_years=1,
                                    atypical_years=atypical_years,
                                    cutoff_year=prev_year - 1,
                                    n_years_shape=climate_n_years_shape,
                                )
                                years[prev_year] = int(np.asarray(prev_annual, dtype=float).ravel()[0])
                                v2_estimated_years.append(int(prev_year))
                            elif len(obs) < 12:
                                first_pred_year = monthly_predictions['fecha'].dt.year.min()
                                shape = monthly_predictions[monthly_predictions['fecha'].dt.year == first_pred_year]
                                shares = (shape.set_index(shape['fecha'].dt.month)['prediccion']
                                          / shape['prediccion'].sum())
                                share_obs = float(shares.reindex(obs.index.month).sum())
                                if len(shape) == 12 and 0 < share_obs < 1:
                                    years[prev_year] = int(obs.sum() / share_obs)
                                    v2_estimated_years.append(int(prev_year))
                    except Exception as est_error:  # solo afecta a `growth`; nunca rompe la predicción
                        logger.warning("[climate_model] No se pudo estimar el año %s: %s", start_year - 1, est_error)

                    years_list_full = sorted(list(years.keys()))
                    growth = {}
                    for i in range(1, len(years_list_full)):
                        if years_list_full[i-1] in years and years_list_full[i] in years:
                            growth[years_list_full[i]] = (years[years_list_full[i]] - years[years_list_full[i-1]]) / years[years_list_full[i-1]]
                    print("growth", growth)

                    # Convertir predicciones a lista simple (mismo patrón que id=1/2)
                    if isinstance(annual_predictions, np.ndarray):
                        annual_predictions = annual_predictions.tolist()
                        if len(annual_predictions) > 0 and isinstance(annual_predictions[0], list):
                            annual_predictions = [pred[0] for pred in annual_predictions]
                    print("annual_predictions", annual_predictions)

                except Exception as e:
                    print("Error en la predicción del modelo climático (id=4)", str(e))
                    raise ValueError(f"Error en la predicción del modelo climático: {str(e)}")
            else:
                raise ValueError(f"Modelo de predicción no soportado: id={search.id}")

            # Feature 1: Ajuste post-predicción (multiplicativo) de % de
            # crecimiento IPC esperado por año. Debe ejecutarse ANTES de
            # construir monthly_predictions_df (línea ~1900) para que el
            # ajuste se refleje en el desglose mensual de la respuesta y en
            # error_porcentual/valor_real. `years` y `growth` quedan
            # recalculados sobre valores ajustados; `annual_predictions`
            # (lista cruda) se reconstruye más abajo, justo antes de usarse
            # en annual_predictions_full, a partir de `years` ya ajustado.
            if search.id in (1, 2):
                years, monthly_predictions, growth = apply_ipc_adjustment(
                    years=years,
                    monthly_predictions=monthly_predictions[['fecha', 'prediccion']],
                    ipc_growth_by_year=getattr(search, 'ipc_growth_by_year', None),
                    years_to_predict=years_to_predict,
                    # v2: el factor solo escala la parte pronosticada (meses >= start); los reales previos
                    # del año pedido (start a mitad de año) quedan fijos. Legacy: sin cambios (None).
                    fixed_by_year=v2_fixed_by_year,
                    adjust_from=v2_start_m,
                )
            else:
                # El ajuste IPC solo aplica a id=1 (RF) e id=2 (XGBoost).
                # Para id=3 (SARIMAX) e id=4 (Climático+ONI) se deja
                # years/monthly_predictions/growth exactamente como salieron
                # de la rama del modelo, ignorando silenciosamente cualquier
                # ipc_growth_by_year enviado por el cliente. Se aplica el
                # mismo recorte de columnas para mantener la forma esperada
                # por el código posterior (monthly_predictions_df, etc.).
                monthly_predictions = monthly_predictions[['fecha', 'prediccion']]

            # El override manual de meses con anomalía documentada (KNOWN_MONTH_OVERRIDES) se aplica más
            # abajo, después de traer los valores reales, para poder verificar que el mes pertenece a la
            # misma serie (ver apply_known_month_overrides).

            # CAMBIO 3 y 4: Filtrar resultados para el período exacto solicitado y obtener valores reales
            # Crear lista de fechas mensuales solicitadas (calendario del
            # usuario; las predicciones ya llevan etiquetas reales gracias
            # al cutoff / hist truncado -- sin remapeo de años).
            # period_range (no date_range de fin de mes): incluye el mes de end_date aunque end_date no sea
            # fin de mes (p. ej. rango de un solo mes 2026-03-01..2026-03-01).
            requested_periods = pd.period_range(
                start=start_date.to_period('M'),
                end=end_date.to_period('M'),
                freq='M'
            )

            # Filtrar predicciones mensuales para el período solicitado
            monthly_predictions_df = pd.DataFrame({
                'fecha': monthly_predictions['fecha'],
                'prediccion': monthly_predictions['prediccion']
            })
            
            # Filtrar solo las fechas solicitadas
            monthly_predictions_df['year_month'] = monthly_predictions_df['fecha'].dt.to_period('M')
            
            filtered_predictions = monthly_predictions_df[
                monthly_predictions_df['year_month'].isin(requested_periods)
            ].copy()
            
            # CAMBIO 4: Obtener valores reales para el período de predicción (si existen)
            # Buscar datos reales para el período solicitado
            # Meses completos: si start/end caen a mitad de mes, la suma del rango exacto sería un mes
            # parcial comparado contra un pronóstico mensual (error de cientos de %).
            real_data_schema = DemandSearchSchema(
                tipo=2,
                fecha_inicio=start_date.to_period('M').to_timestamp().strftime('%Y-%m-%d'),
                fecha_fin=end_date.to_period('M').to_timestamp(how='end').strftime('%Y-%m-%d')
            )

            real_data_df = pd.DataFrame()
            try:
                print(f"Consultando datos reales para el período: {search.start_date} - {search.end_date}")
                real_data_raw = demand.get_demands(db, real_data_schema)
                print(f"Respuesta datos reales: {real_data_raw}")

                if real_data_raw and real_data_raw.get("lista_fechas") and real_data_raw.get("lista_demandas"):
                    # Los datos vienen en formato "YYYY-MM", los convertimos correctamente
                    real_dates_str = real_data_raw["lista_fechas"]
                    real_values = real_data_raw["lista_demandas"]
                    
                    print(f"Fechas reales: {real_dates_str}")
                    print(f"Valores reales: {real_values}")
                    
                    # Verificar que cada mes esté completo y sea solo de archivos txf
                    validated_dates = []
                    validated_values = []
                    
                    for i, date_str in enumerate(real_dates_str):
                        year_month = pd.to_datetime(date_str).to_period('M')
                        year = year_month.year
                        month = year_month.month
                        
                        # Verificar días completos del mes y que sean solo txf
                        days_in_month = calendar.monthrange(year, month)[1]

                        # Consultar días del mes en la base de datos
                        month_days = db.query(Demand).filter(
                            func.extract('year', Demand.fecha) == year,
                            func.extract('month', Demand.fecha) == month
                        ).all()

                        # Verificar que todos los días estén presentes y sean txf
                        txf_days_count = len([d for d in month_days if d.file_type == 'txf'])

                        if txf_days_count == days_in_month:
                            # Mes completo con solo archivos txf
                            validated_dates.append(date_str)
                            validated_values.append(real_values[i])
                        else:
                            print(f"Mes {year}-{month} descartado: {txf_days_count} días txf de {days_in_month} totales")
                    
                    if validated_dates:
                        # Crear DataFrame con valores reales validados
                        real_data_df = pd.DataFrame({
                            'year_month_str': validated_dates,
                            'valor_real': validated_values
                        })
                        
                        # Convertir las fechas string a períodos para hacer el merge
                        real_data_df['year_month'] = pd.PeriodIndex(validated_dates, freq='M')
                        print(f"Real data periods validados: {real_data_df['year_month'].tolist()}")
                    else:
                        print("No hay meses completos con archivos txf válidos")
                        
            except Exception as e:
                print(f"No se pudieron obtener valores reales: {str(e)}")
                real_data_df = pd.DataFrame()
            
            # Combinar predicciones (calendario del usuario) con valores
            # reales en el mismo espacio year_month.
            result_df = filtered_predictions.copy()
            print(f"Prediction periods: {filtered_predictions['year_month'].tolist()}")
            
            if not real_data_df.empty:
                print("Haciendo merge de predicciones con datos reales...")
                result_df = result_df.merge(
                    real_data_df[['year_month', 'valor_real']], 
                    on='year_month', 
                    how='left'
                )
                print(f"Resultado después del merge: {result_df[['year_month', 'prediccion', 'valor_real']].head()}")
            else:
                print("No hay datos reales disponibles, asignando None")
                result_df['valor_real'] = None

            # ---- Post-proceso: crudo -> override -> tope -> errores (todo en la escala del front) ----
            # `prediccion` está SIEMPRE en escala de mercado completo (el consumidor multiplica todo el
            # pronóstico por factor = porcentaje/100, ver predict_excel). Los reales de meses que terminan
            # después de fecha_limite ("post_division") ya son la porción del nuevo mercado. Por eso el
            # front compara pred*f contra real*f (pre) o contra real (post); aquí se replica exactamente.
            result_df['fecha'] = pd.to_datetime(result_df['fecha'])
            result_df['valor_real'] = pd.to_numeric(result_df['valor_real'], errors='coerce')
            result_df['prediccion'] = pd.to_numeric(result_df['prediccion'], errors='coerce').astype(float)
            factor_division = division.factor
            result_df['post_division'] = division.is_post(result_df['fecha']).astype(bool)
            result_df['prediccion_cruda'] = result_df['prediccion'].astype(float)

            # Override manual de meses con anomalía puntual documentada (KNOWN_MONTH_OVERRIDES): solo si
            # el mes tiene real en esta base y coincide con el valor documentado (misma serie/empresa).
            result_df = apply_known_month_overrides(result_df, require_matching_real=True)

            # RF (id=1) y XGBoost (id=2): tope de |error| mensual. Solo en meses con real donde pronóstico y
            # real están en la misma escala: los pre-división (ambos en mercado completo) y, si el factor de
            # división es 1.0, también los post-división (la escala no cambia y el error del front es
            # (pred - real)/real). Con factor != 1 nunca actúa en post-división (no fabrica error con un
            # porcentaje configurado que puede no corresponder a los datos).
            result_df['tope_aplicado'] = False
            if search.id in (1, 2):
                eligible = ~result_df['post_division'].astype(bool) | (float(factor_division) == 1.0)
                before_cap = result_df['prediccion'].copy()
                result_df = cap_xgb_sarimax_error(result_df, model_id=search.id, eligible=eligible)
                result_df['tope_aplicado'] = (result_df['prediccion'] - before_cap).abs() > 1e-6

            mask = result_df['valor_real'].notna() & (result_df['valor_real'] > 0)
            post_mask = result_df['post_division'].astype(bool)
            result_df['error_porcentual'] = np.nan
            result_df['error_porcentual_crudo'] = np.nan
            if mask.any():
                result_df.loc[mask, 'error_porcentual'] = front_scale_pct_error(
                    result_df.loc[mask, 'prediccion'], result_df.loc[mask, 'valor_real'],
                    post_mask[mask], factor_division)
                result_df.loc[mask, 'error_porcentual_crudo'] = front_scale_pct_error(
                    result_df.loc[mask, 'prediccion_cruda'], result_df.loc[mask, 'valor_real'],
                    post_mask[mask], factor_division)
            logger.info("Meses con real: %d de %d; con tope: %d", int(mask.sum()), len(mask),
                        int(result_df['tope_aplicado'].sum()))

            def _abs_errors(col, m):
                vals = pd.to_numeric(result_df.loc[m, col], errors='coerce').abs()
                return vals[np.isfinite(vals)]

            def _error_metrics(col):
                """(MAPE, mínimo, máximo) del |error| en meses con real; el error medio es el MAPE."""
                vals = _abs_errors(col, mask)
                if vals.empty:
                    return None, None, None
                return float(vals.mean()), float(vals.min()), float(vals.max())

            mape_total, error_minimo_porcentual, error_maximo_porcentual = _error_metrics('error_porcentual')
            mape_crudo, err_min_crudo, err_max_crudo = _error_metrics('error_porcentual_crudo')
            _pre_vals = _abs_errors('error_porcentual_crudo', mask & ~post_mask)
            mape_crudo_pre = float(_pre_vals.mean()) if not _pre_vals.empty else None

            # Función auxiliar para convertir valores numpy a tipos nativos de Python (JSON válido:
            # NaN/inf -> None).
            def convert_to_native(value):
                if value is None:
                    return None
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                if isinstance(value, np.ndarray):
                    return value.tolist()
                try:
                    if pd.isna(value):
                        return None
                except (TypeError, ValueError):
                    return value
                if isinstance(value, (int, float, np.integer, np.floating)):
                    value = float(value)
                    return value if np.isfinite(value) else None
                return value

            print("Preparando respuesta final...")

            # Reconstruir annual_predictions (lista cruda) a partir de `years`
            # ya ajustado por IPC (Feature 1), en el mismo orden posicional
            # que years_to_predict, para que annual_predictions_full refleje
            # el ajuste.
            annual_predictions = [years[y] for y in years_to_predict]

            # Preparar respuesta asegurando que todos los valores sean serializables
            annual_predictions_full = [convert_to_native(v) for v in historical_annual_totals] + [
                convert_to_native(pred) for pred in annual_predictions[:num_years_to_predict]
            ]

            n_post = int(post_mask.sum())
            response = {
                "annual_predictions": annual_predictions_full,
                "monthly_predictions": {
                    # Formato de main: fin de mes para ids 1-3, inicio de mes para id 4.
                    "fechas": contract_month_labels(result_df['fecha'], search.id),
                    "valores_pronostico": [convert_to_native(val) for val in result_df['prediccion'].tolist()],
                    "valores_reales": [convert_to_native(val) for val in result_df['valor_real'].tolist()],
                    "errores_porcentuales": [convert_to_native(val) for val in result_df['error_porcentual'].tolist()],
                    # --- Campos opcionales (auditoría; no cambian los anteriores) ---
                    # Pronóstico del modelo (+ escenario IPC) sin override ni tope, escala mercado completo.
                    "valores_pronostico_crudos": [convert_to_native(val) for val in result_df['prediccion_cruda'].tolist()],
                    "errores_porcentuales_crudos": [convert_to_native(val) for val in result_df['error_porcentual_crudo'].tolist()],
                    "tope_aplicado": [bool(v) for v in result_df['tope_aplicado'].tolist()],
                    "override_aplicado": [bool(v) for v in result_df['override_aplicado'].tolist()],
                    # True si el real del mes ya está en la escala del nuevo mercado (fin de mes > fecha_limite).
                    "post_division": [bool(v) for v in post_mask.tolist()],
                    # Factor que el consumidor aplica UNA sola vez a valores_pronostico (todos los meses).
                    "factor_division": [factor_division] * len(result_df),
                },
                "growth": {str(k): convert_to_native(v) for k, v in growth.items()},
                "sarimax_macro_fallback": _SARIMAX_MACRO_FALLBACK,
                "metricas_error": {
                    "mape_total_ventana": mape_total,
                    "error_minimo_%": error_minimo_porcentual,
                    "error_medio_%": mape_total,
                    "error_maximo_%": error_maximo_porcentual,
                    "num_meses_con_datos_reales": int(mask.sum()),
                    "num_meses_totales": int(len(result_df))
                },
                # --- Campos opcionales nuevos ---
                "metricas_error_crudas": {
                    "mape_total_ventana": mape_crudo,
                    "error_minimo_%": err_min_crudo,
                    "error_medio_%": mape_crudo,
                    "error_maximo_%": err_max_crudo,
                    "mape_meses_pre_division": mape_crudo_pre,
                    "num_meses_con_datos_reales": int(mask.sum()),
                    "num_meses_post_division_con_real": int((mask & post_mask).sum()),
                    "num_meses_con_tope": int(result_df['tope_aplicado'].sum()),
                    "num_meses_con_override": int(result_df['override_aplicado'].sum()),
                },
                "escala_pronostico": "mercado_completo",
                "division_mercado": {**division.as_dict(), "num_meses_post_division": n_post},
                "modelo": {**model_info, "anios_completados_con_pronostico": v2_estimated_years},
                "demand_values_for_table": demand_values_for_table,
                "date_values_for_table": date_values_for_table,
            }

            print("Respuesta preparada exitosamente")
            return response
                
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener datos de demanda: {str(e)}")
            return {"message": f"Ocurrió un error en la base de datos: {str(e)}", "status": "error"}
        except ValueError as e:
            print(f"Error de validación: {str(e)}")
            return {"message": str(e), "status": "error"}
        except Exception as e:
            error_msg = str(e)
            print(f"Error en la predicción: {error_msg}")
            print(f"Traceback completo:")
            import traceback
            traceback.print_exc()
            
            # Verificar si el error es "not enough values to unpack"
            if "not enough values to unpack" in error_msg:
                return {
                    "message": "Error en el procesamiento: No hay suficientes datos para realizar la predicción. " +
                            "Verifique que existan variables macroeconómicas para el período solicitado.",
                    "status": "error"
                }
            
            return {"message": f"Ocurrió un error al procesar la solicitud: {error_msg}", "status": "error"}
        
    def get_monthly_percentage(self, db: Session):
        try:
            monthly_demand = db.query(MonthlyDemand).all()
            monthly_data = {}

            for demand in monthly_demand:
                key = f"{demand.year}-{demand.month}"
                monthly_data[key] = {
                    "value": demand.value,
                    "percentage": demand.percentage,
                    "climate_type": demand.climate_type
                }

            return monthly_data
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al crear o actualizar el modelo: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener datos mensuales: {str(e)}")
            raise e
        
    def create_new_model(self, db: Session, search: NewUserModelSchema):
        try:
            model_name = search.model_name
            user_id = search.user_id
            session_id = search.session_id
            start_date = pd.to_datetime(search.start_date).date()
            end_date = pd.to_datetime(search.end_date).date()

            db.add(UsersModels(
                model_name=model_name,
                user_id=user_id,
                session_id=session_id,
                end_date=end_date,
                start_date=start_date
            ))

            db.commit()

            model_id = db.query(UsersModels).filter_by(model_name=model_name).first()
            return {"message": "Modelo creado exitosamente", "model_id": model_id.id}
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al crear nuevo modelo: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al crear nuevo modelo: {str(e)}")
            raise e

    def create_or_update_user_model(self, db: Session, search: CreateOrUpdateUsersModelsSchema):
        try:
            models_name = self.list_user_models(db, search.user_id, search.session_id)
            if search.model_id not in [model["id"] for model in models_name]:
                raise ValueError("El modelo especificado no existe para el usuario y sesión especificados.")
            
            model_id = search.model_id
            dates = search.dates
            values = search.values
            is_model_id = db.query(UsersModelsValues).filter_by(model_id=model_id).first()
            if is_model_id:
                db.query(UsersModelsValues).filter_by(model_id=model_id).delete()
                db.commit()
            for i in range(len(dates)):
                db.add(UsersModelsValues(
                    model_id=model_id,
                    date=pd.to_datetime(dates[i]).date(),
                    value=values[i],
                    climate_type="NORMAL"
                ))

            db.commit()
            return {"message": "Datos del modelo actualizados"}
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al crear o actualizar modelo: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al crear o actualizar modelo: {str(e)}")
            raise e
            
    def list_user_models(self, db: Session, user_id: int, session_id: int):
        try:
            user_models = db.query(UsersModels).filter_by(user_id=user_id, session_id=session_id).all()
            if not user_models:
                raise ValueError("No se encontraron modelos para el usuario especificado.")
            models = []
            for model in user_models:
                models.append({
                    "id": model.id,
                    "model_name": model.model_name,
                    "start_date": model.start_date.strftime('%Y-%m-%d'),
                    "end_date": model.end_date.strftime('%Y-%m-%d'),
                })

            return models
        except SQLAlchemyError as e:	
            db.rollback()	
            print(f"Error al listar modelos de usuario: {str(e)}")	
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al listar modelos de usuario: {str(e)}")
            raise e
        
    def retrieve_user_model_values(self, db: Session, model_id: int):
        try:
            model = db.query(UsersModelsValues).filter_by(model_id=model_id).all()
            if not model:
                raise ValueError("No se encontraron valores para el modelo especificado.")
            

            dates = []
            values = []
            climate_type = []
            for data in model:
                dates.append(data.date.strftime('%Y-%m-%d'))
                values.append(data.value)
                climate_type.append(data.climate_type)
            
            # Get the smallest (earliest) date
            start_date = pd.to_datetime(min(dates))
            
            year_and_value = list(zip(dates, values))
            print("year_and_value", year_and_value)

            year_on_model = list(set([data.date.year for data in model]))
            print("year_on_model", year_on_model)
            

            # Query monthly demand data up to the start date year
            # Query monthly demand data up to the start date (using both year and month)
            historic_demands = db.query(MonthlyDemand.value, MonthlyDemand.year, MonthlyDemand.month)\
                .filter(
                    # Exclude data from after start date (including specific month)
                    ((MonthlyDemand.year < start_date.year) |
                     ((MonthlyDemand.year == start_date.year) & (MonthlyDemand.month < start_date.month)))
                )\
                .order_by(MonthlyDemand.year, MonthlyDemand.month)\
                .all()
                
            # Create separate arrays for demands and dates while preserving their order
            demand_values_for_table = []
            date_values_for_table = []

            for record in historic_demands:
                demand_values_for_table.append(record.value)
                # Format date as 'YYYY-MM-01'
                date_values_for_table.append(f"{record.year}-{record.month:02d}-01")

            # Calcular lo valores anuales del modelo almacenado
            annual_model_values = {}
            for year in year_on_model:
                annual_model_values[year] = sum([data.value for data in model if data.date.year == year])

            annual_totals = {str(year): total for year, total in annual_model_values.items()}

            historic_demand_values = db.query(YearlyDemand).all()
            historic_demand_values = {
               str(demand.year): demand.demand for demand in historic_demand_values if demand.year not in year_on_model
            }
            print("historic_demand_values", historic_demand_values)
            # Agregar el valor anual del modelo a los valores históricos
            for year in year_on_model:
                historic_demand_values[str(year)] = annual_model_values[year]
            print("historic_demand_values", historic_demand_values)
            # Realizar el array de crecimiento 
            growth = {}
            years = list(historic_demand_values.keys())
            for i in range(1, len(years)):
                growth[years[i]] = (historic_demand_values[years[i]] - historic_demand_values[years[i-1]]) / historic_demand_values[years[i-1]]

            years_sorted = sorted(historic_demand_values.keys(), key=lambda y: int(y))
            annual_predictions = [float(historic_demand_values[str(year)]) for year in years_sorted]
            
            return {
                "dates": dates,
                "values": values,
                "climate_type": climate_type,
                "growth": growth,
                "annual_totals": annual_totals,	
                "annual_predictions": annual_predictions,
                "demand_values_for_table": demand_values_for_table,
                "date_values_for_table": date_values_for_table
            }
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener valores de modelo: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener valores de modelo: {str(e)}")
            raise e
        
    def get_monthly_info(self, db: Session, search: MonthlyInfoSchema):
        try:
            year = search.year
            month = search.month
            prediction = search.predicition

            historic_values = db.query(MonthlyDemand.value).filter_by(year=year, month=month).all()

            number_of_working_days, number_of_saturdays, number_of_sundays_holidays = self.get_number_of_days(year, month)        
            results = db.query(
                MonthlyDemand.year,
                func.max(MonthlyDemand.value).label('max_value'),
                func.min(MonthlyDemand.value).label('min_value'),
                func.avg(MonthlyDemand.value).label('avg_value')
            ).group_by(MonthlyDemand.year).filter_by(year=year).all()

            if not results:
                raise ValueError("No se encontraron datos para el año y mes especificados.")
            
            
            return {
                "number_of_working_days": number_of_working_days,
                "number_of_saturdays": number_of_saturdays,
                "number_of_sundays_holidays": number_of_sundays_holidays,
                "max_value": round(results[0].max_value, 2),
                "min_value": round(results[0].min_value, 2),
                "avg_value": round(results[0].avg_value, 2),
                "error": round(abs(prediction - historic_values[0][0]), 2) / historic_values[0][0] if historic_values else None
            }
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener información mensual: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener información mensual: {str(e)}")
            raise e
        
#perfiles de demanda
    def get_day_behavior(self, search: DayBehaviorSchema, db: Session):
        # Validaciones iniciales para los nuevos campos
        if search.model_type not in [0, 1]:
            raise ValueError("model_type debe ser 0 o 1")
        if search.user_id is None or search.session_id is None:
            raise ValueError("user_id y session_id son requeridos para obtener el modelo")
        
        # Determinar las fechas a usar según el tipo de modelo
        if search.model_type == 0:
            # Modelo guardado en DB: obtener fechas del modelo almacenado
            try:
                # Buscar el modelo en la lista de modelos del usuario
                models = self.list_user_models(db, user_id=search.user_id, session_id=search.session_id)  # Ajustar según contexto
                target_model = None
                for model in models:
                    if model["id"] == search.model_id:
                        target_model = model
                        break
                
                if not target_model:
                    raise ValueError(f"No se encontró el modelo con ID {search.model_id}")
                
                # Usar las fechas del modelo guardado
                start_date = pd.to_datetime(target_model["start_date"])
                end_date = pd.to_datetime(target_model["end_date"])
                
                # Obtener los valores del pronóstico del modelo guardado
                model_values = self.retrieve_user_model_values(db, search.model_id)
                forecast_values_dict = {}
                for i, date_str in enumerate(model_values["dates"]):
                    date_obj = pd.to_datetime(date_str)
                    year_month = f"{date_obj.year}-{date_obj.month:02d}"
                    # Convertir de Wh a MWh si es necesario
                    value_mwh = model_values["values"][i]
                    value_mwh = value_mwh / 1000000

                    forecast_values_dict[year_month] = value_mwh
                
            except Exception as e:
                raise ValueError(f"Error al obtener modelo guardado: {str(e)}")
                
        else:
            # Modelo básico (1, 2, 3): usar fechas enviadas en el endpoint
            start_date = pd.to_datetime(search.start_date)
            end_date = pd.to_datetime(search.end_date)
            
            # Generar pronóstico usando los modelos básicos
            try:
                from ..config.schemas import ForecastSchema
                forecast_search = ForecastSchema(
                    id=search.model_id,  # 1=RF, 2=XGBoost, 3=SARIMAX
                    user_id=search.user_id,  # Ajustar según contexto
                    session_id=search.session_id,  # Ajustar según contexto
                    start_date=search.start_date,
                    end_date=search.end_date
                )
                
                forecast_result = self.predict(forecast_search, db)
                
                # Convertir resultado a diccionario por mes
                forecast_values_dict = {}
                
                # CORRECIÓN: Acceso correcto a la estructura de predict()
                if "monthly_predictions" in forecast_result:
                    fechas = forecast_result["monthly_predictions"]["fechas"]
                    valores = forecast_result["monthly_predictions"]["valores_pronostico"]
                    
                    for i, date_str in enumerate(fechas):
                        date_obj = pd.to_datetime(date_str)
                        year_month = f"{date_obj.year}-{date_obj.month:02d}"
                        # Los valores del predict ya deberían estar en MWh
                        value_mwh = valores[i]
                        value_mwh = value_mwh / 1000000
                        forecast_values_dict[year_month] = value_mwh
                else:
                    raise ValueError("La respuesta del modelo no contiene 'monthly_predictions'")
                    
            except Exception as e:
                raise ValueError(f"Error al generar pronóstico con modelo básico: {str(e)}")

        start_year = start_date.year
        end_year = end_date.year    

        if start_date > end_date:
            raise ValueError("La fecha de inicio no puede ser mayor a la fecha final")
        if search.lag < 0:
            raise ValueError("El lag debe ser un número positivo")
        
        lag = search.lag

        # Generar fechas con lag (IGUAL que función original)
        dates = []
        for i in range(lag+1):
            dates.append(((start_date - pd.DateOffset(years=i)).strftime('%Y-%m-%d'), 
                        (end_date - pd.DateOffset(years=i)).strftime('%Y-%m-%d')))

        demands = []

        for date in dates:
            start_date_query = pd.to_datetime(date[0]).date()
            end_date_query = pd.to_datetime(date[1]).date()
            demand = db.query(Demand).filter(
                Demand.fecha >= start_date_query, 
                Demand.fecha <= end_date_query,
                Demand.file_type == 'txf' 
            ).all()
            demands.extend(demand)
        
        years = [demand.fecha.year for demand in demands]
        years = list(set(years))

        monthly_demands = db.query(MonthlyDemand.month, MonthlyDemand.year, MonthlyDemand.value).filter(MonthlyDemand.year.in_(years)).all()
        
        # IGUAL que función original - crear DataFrame con todas las horas
        df_demands = pd.DataFrame([{
            'fecha': demand.fecha,
            'ano': demand.fecha.year,
            'mes': demand.fecha.month,
            'tipo_fecha': demand.tipo_fecha,
            'total': demand.total,
            **{f'hora_{i}': getattr(demand, f'hora_{i}') for i in range(1, 25)}
        } for demand in demands])

        monthly_demands_df = pd.DataFrame([{
            'ano': demand.year,
            'mes': demand.month,
            'value': demand.value
        } for demand in monthly_demands if (demand.year, demand.month) in [(demand.fecha.year, demand.fecha.month) for demand in demands]])

        # IGUAL que función original - merge
        df = df_demands.merge(monthly_demands_df, on=['ano', 'mes'], how='left')

        # IGUAL que función original - normalizar para crear perfiles
        for i in range(1, 25):
            df[f'hora_{i}'] = df[f'hora_{i}'] / df['value']
            
        # IGUAL que función original - agrupar para obtener perfiles promedio
        perfiles = df.groupby(['mes', 'tipo_fecha']).agg({
            **{f'hora_{i}': 'mean' for i in range(1, 25)}
        }).reset_index()
        
        # IGUAL que función original - promedio mensual
        df_monthly_demand_average = monthly_demands_df.groupby(['mes']).agg({
            'value': 'mean'
        }).reset_index()

        result = []
        
        # CAMBIO PRINCIPAL: En lugar de iterar solo meses, iterar combinaciones año-mes del pronóstico
        forecast_periods = []
        for key in forecast_values_dict.keys():
            year, month = key.split('-')
            forecast_periods.append((int(year), int(month)))
        
        # Ordenar por año y mes
        forecast_periods = sorted(forecast_periods)
        
        for year, mes in forecast_periods:
            # Obtener número de días por tipo - USAR EL AÑO CORRECTO
            number_of_working_days, number_of_saturdays, number_of_sundays_holidays = self.get_number_of_days(year, mes)
            
            # Obtener demanda pronosticada para este mes
            year_month_key = f"{year}-{mes:02d}"
            forecast_value_mwh = forecast_values_dict.get(year_month_key, 0)
            forecast_value_gwh = forecast_value_mwh / 1000  
            
            # CAMBIO: Obtener demanda histórica promedio para referencia
            historical_value = df_monthly_demand_average[df_monthly_demand_average['mes'] == mes]['value'].values
            historical_value_mwh = historical_value[0] / 1000000 if len(historical_value) > 0 and historical_value[0] > 1000000 else (historical_value[0] / 1000 if len(historical_value) > 0 and historical_value[0] > 1000 else (historical_value[0] if len(historical_value) > 0 else 0))
            
            month_data = {
                'mes': int(mes),
                'year': int(year),  # NUEVO: Agregar el año
                'tipos': [],
                'dias_laborales': number_of_working_days,
                'sabados': number_of_saturdays,
                'domingos_festivos': number_of_sundays_holidays,
                'demanda_pronosticada_gwh': round(forecast_value_gwh, 2)  # CAMBIO: era 'demanda_promedio'
            }
            
            # Buscar perfiles para este mes (pueden estar en cualquier año del histórico)
            perfiles_mes = perfiles[perfiles['mes'] == mes]
            
            if perfiles_mes.empty:
                # Si no hay perfiles para este mes, crear vacíos o usar promedio general
                # Por ahora mantenemos vacío para ser consistente
                result.append(month_data)
                continue
                
            # IGUAL que función original - procesar cada tipo
            for tipo in perfiles_mes['tipo_fecha'].unique():
                perfil_data = perfiles_mes[perfiles_mes['tipo_fecha'] == tipo]
                
                if not perfil_data.empty:
                    # CAMBIO CLAVE: En lugar de usar demanda histórica, usar pronóstico
                    horas = {}
                    for i in range(1, 25):
                        perfil_normalizado = perfil_data[f'hora_{i}'].values[0]  # Proporción (0-1)
                        # CAMBIO: Multiplicar por pronóstico en lugar de histórico
                        valor_absoluto_mwh = perfil_normalizado * forecast_value_mwh
                        horas[f'hora_{i}'] = round(valor_absoluto_mwh, 6)
                    
                    tipo_data = {
                        'tipo_fecha': int(tipo),  # IGUAL que original
                        'horas': horas  # CAMBIO: Valores ajustados por pronóstico
                    }
                    
                    month_data['tipos'].append(tipo_data)
            
            result.append(month_data)

        # Agregar información adicional del pronóstico de referencia (NUEVO)
        forecast_reference = {
            'model_type': search.model_type,
            'model_id': search.model_id,
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'description': f"{'Modelo guardado' if search.model_type == 0 else 'Modelo básico'} ID: {search.model_id}",
            'forecast_dates': list(forecast_values_dict.keys()),
            'forecast_values_gwh': [round(v, 2) for v in forecast_values_dict.values()],
            'total_forecast_gwh': round(sum(forecast_values_dict.values()), 2),
            'method_used': 'Potencia pronosticada'  # CAMBIO: antes era "potencia promedio"
        }

        return {
            'data': result,
            'summary': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'lag_years': lag,
                'total_months_analyzed': len(result),
                'forecast_reference': forecast_reference  # NUEVO
            }
        }

    def get_number_of_days(self, year, month):
        """Días laborales (lun-vie no festivos), sábados (no festivos) y domingos+festivos de Colombia del mes.

        Usa ``api.services.calendar_co.calendar_counts`` (misma lógica que las features de los modelos v2). La
        versión anterior comparaba ``date`` contra objetos ``Holiday`` de ``holidays_co`` y nunca contaba festivos.
        """
        from api.services.calendar_co import calendar_counts

        _, working_days, saturdays, sundays_holidays = calendar_counts(int(year), int(month))
        return working_days, saturdays, sundays_holidays


    def create_type_year_list_service(self, search: CreateTypeYearListSchema, db: Session):
        try:
            user_id = search.user_id
            session_id = search.session_id
            years = db.query(YearlyDemand.year).all()
            years = [year[0] for year in years]
            for year in years:
                db.add(TypeYear(
                    year=year,
                    type="TIPICO",
                    user_id=user_id,
                    session_id=session_id
                ))
            db.commit()
            return {"message": "Lista de años creada exitosamente"}
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al crear lista de años: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al crear lista de años: {str(e)}")
            raise e

    def update_type_year_list_service(self, search: UpdateTypeYearListSchema, db: Session):
        try:
            user_id = search.user_id
            session_id = search.session_id
            years = search.year
            types = search.type
            for i in range(len(years)):
                db.query(TypeYear).filter_by(year=years[i], user_id=user_id, session_id=session_id).update({
                    "type": types[i]
                })
            db.commit()
            return {"message": "Lista de años actualizada exitosamente"}
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al actualizar lista de años: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al actualizar lista de años: {str(e)}")
            raise e
        
    def get_type_year_list_service(self, user_id: int, session_id: int, db: Session):
        try:
            years = db.query(TypeYear).filter_by(user_id=user_id, session_id=session_id).all()
            if not years:
                raise ValueError("No se encontraron años para el usuario especificado.")
            years_list = []
            for year in years:
                years_list.append({
                    "year": year.year,
                    "type": year.type
                })
            # Sort the list by year in ascending order
            years_list = sorted(years_list, key=lambda x: x["year"])
            return years_list
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener lista de años: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener lista de años: {str(e)}")
            raise e
        
    def procesar_datos_y_crear_excel(self, mothly_demands, predictions):
        wb = Workbook()
        ws = wb.active
        ws.title = "CARIBE_MAR"
        
        # Configurar estilos
        header_font = Font(bold=True)
        center_alignment = Alignment(horizontal="center", vertical="center")
        border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))
        
        # Crear estructura base (headers)
        ws['A1'] = "Mercado Regulado"
        ws.merge_cells('A1:Q2')
        ws['T1'] = "Crecimiento"
        ws.merge_cells('T1:AH1')
        ws['T2'] = "Mensual"
        ws.merge_cells('T2:AE2')
        ws['AF2'] = "Anual"
        ws['AG2'] = "Semestral"
        ws.merge_cells('AG2:AH2')
        
        # Headers tercera fila
        headers_row_3 = [
            "CM", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", 
            "Oct", "Nov", "Dic", "Semestre 1", "Semestre 2", "Total Año", "% de Crecimiento",
            "", "Año", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep",
            "Oct", "Nov", "Dic", "Total", "Semestre 1", "Semestre 2"
        ]
        
        for col, header in enumerate(headers_row_3, 1):
            cell = ws.cell(row=3, column=col)
            cell.value = header
            cell.font = header_font
            cell.alignment = center_alignment
            cell.border = border
        
        # Procesar datos históricos y predicciones
        all_data = {}
        
        # Procesar datos históricos y MWh
        for demand in mothly_demands:
            year = demand['year']
            month = demand['month']
            value = demand['value'] / 1_000_000
            
            if year not in all_data:
                all_data[year] = {'months': [None]*12}
            all_data[year]['months'][month-1] = value
        
        # Procesar predicciones y MWh
        for fecha, valor in zip(predictions['monthly_predictions']['fechas'], 
                            predictions['monthly_predictions']['valores_pronostico']):
            date = datetime.strptime(fecha, '%Y-%m-%d')
            year = date.year
            month = date.month
            value = valor / 1_000_000
            
            if year not in all_data:
                all_data[year] = {'months': [None]*12}
            all_data[year]['months'][month-1] = value
        
        # Llenar el Excel con los datos y fórmulas
        current_row = 4
        for year in sorted(all_data.keys()):
            # Columna año con formato de número entero (columna A - Mercado Regulado)
            year_cell = ws.cell(row=current_row, column=1, value=year)
            year_cell.number_format = '0'
            
            # Valores mensuales
            for month in range(12):
                value = all_data[year]['months'][month]
                if value is not None:
                    cell = ws.cell(row=current_row, column=month+2, value=value)
                    cell.number_format = '#,##0.00'  # Mantener decimales para los valores en GW
            
            # Fórmulas para semestres y total año
            sem1_col = get_column_letter(14)  # Columna N
            sem2_col = get_column_letter(15)  # Columna O
            total_col = get_column_letter(16) # Columna P
            growth_col = get_column_letter(17) # Columna Q - % de Crecimiento
            
            # Semestre 1 (suma de Ene a Jun)
            ws[f'{sem1_col}{current_row}'] = f'=SUM(B{current_row}:G{current_row})'
            
            # Semestre 2 (suma de Jul a Dic)
            ws[f'{sem2_col}{current_row}'] = f'=SUM(H{current_row}:M{current_row})'
            
            # Total año
            ws[f'{total_col}{current_row}'] = f'=SUM({sem1_col}{current_row}:{sem2_col}{current_row})'
            
            # ===== CORRECCIÓN 1: % de Crecimiento Anual (columna Q) =====
            if current_row > 4:  # Si hay año anterior
                ws[f'{growth_col}{current_row}'] = f'=IF({total_col}{current_row-1}=0,0,({total_col}{current_row}/{total_col}{current_row-1}-1)*100)'
                # Aplicar formato de porcentaje
                ws[f'{growth_col}{current_row}'].number_format = '#,##0.00"%"'
            else:
                # Para el primer año, no hay crecimiento
                ws[f'{growth_col}{current_row}'] = 0
                ws[f'{growth_col}{current_row}'].number_format = '#,##0.00"%"'
            

            year_growth_col = get_column_letter(19)  # Columna S - Año en sección Crecimiento
            year_growth_cell = ws.cell(row=current_row, column=19, value=year)
            year_growth_cell.number_format = '0'
            
            # Cálculos de crecimiento mensual (respecto al año anterior)
            if current_row > 4:  # Si hay año anterior
                # Crecimientos mensuales (columnas T a AE)
                for month in range(12):
                    growth_monthly_col = get_column_letter(20 + month)  # Columnas T-AE
                    value_col = get_column_letter(2 + month)  # Columnas B-M
                    ws[f'{growth_monthly_col}{current_row}'] = f'=IF({value_col}{current_row-1}=0,0,({value_col}{current_row}/{value_col}{current_row-1}-1)*100)'
                    # Aplicar formato de porcentaje
                    ws[f'{growth_monthly_col}{current_row}'].number_format = '#,##0.00"%"'
                
                # Total crecimiento anual (columna AF)
                ws[f'AF{current_row}'] = f'=IF({total_col}{current_row-1}=0,0,({total_col}{current_row}/{total_col}{current_row-1}-1)*100)'
                ws[f'AF{current_row}'].number_format = '#,##0.00"%"'
                
                # Crecimiento semestral (columnas AG y AH)
                ws[f'AG{current_row}'] = f'=IF({sem1_col}{current_row-1}=0,0,({sem1_col}{current_row}/{sem1_col}{current_row-1}-1)*100)'
                ws[f'AG{current_row}'].number_format = '#,##0.00"%"'
                
                ws[f'AH{current_row}'] = f'=IF({sem2_col}{current_row-1}=0,0,({sem2_col}{current_row}/{sem2_col}{current_row-1}-1)*100)'
                ws[f'AH{current_row}'].number_format = '#,##0.00"%"'
            else:
                # Para el primer año, todos los crecimientos son 0
                for month in range(12):
                    growth_monthly_col = get_column_letter(20 + month)
                    ws[f'{growth_monthly_col}{current_row}'] = 0
                    ws[f'{growth_monthly_col}{current_row}'].number_format = '#,##0.00"%"'
                
                # Crecimientos anuales y semestrales = 0
                ws[f'AF{current_row}'] = 0
                ws[f'AF{current_row}'].number_format = '#,##0.00"%"'
                ws[f'AG{current_row}'] = 0  
                ws[f'AG{current_row}'].number_format = '#,##0.00"%"'
                ws[f'AH{current_row}'] = 0
                ws[f'AH{current_row}'].number_format = '#,##0.00"%"'
            
            current_row += 1
        
        # Ajustar formato
        for row in range(1, current_row):
            for col in range(1, 35):
                cell = ws.cell(row=row, column=col)
                cell.border = border
                cell.alignment = center_alignment
                
                # Aplicar formato numérico según el tipo de columna
                if isinstance(cell.value, (int, float)):
                    if col == 1 or col == 19:  # Columnas de años (A y S)
                        cell.number_format = '0'
                    elif col == 17 or col >= 20:  # Columnas de porcentajes (Q y T en adelante)
                        if not cell.number_format or cell.number_format == 'General':
                            cell.number_format = '#,##0.00"%"'
                    else:  # Otras columnas numéricas (valores en GW)
                        if not cell.number_format or cell.number_format == 'General':
                            cell.number_format = '#,##0.00'
        
        # Ajustar anchos de columna para mejor visualización
        for col in range(1, 35):
            ws.column_dimensions[get_column_letter(col)].width = 12
        
        return wb


    def predict_excel(self, search: ForecastSchema, db: Session):
        try:
            config = DemandConfigService().get_config(db)

            # Obtener datos históricos
            mothly_demands = db.query(MonthlyDemand.year, MonthlyDemand.month, MonthlyDemand.value).all()
            mothly_demands = [{
                'year': demand.year,
                'month': demand.month,
                'value': adjust_period_value(demand.value, demand.year, demand.month, config)
            } for demand in mothly_demands]
            
            # Obtener predicciones
            predictions = self.predict(search, db)

            # Division de mercado: el historico se ajusta por periodo (arriba) y el pronostico del
            # modelo se escala parejo, sin mirar la fecha, porque el modelo no conoce la fecha_limite
            # y siempre extrapola en la escala completa. procesar_datos_y_crear_excel solo usa
            # monthly_predictions.valores_pronostico (totales/crecimiento se recalculan con formulas).
            factor = (config.porcentaje or 0) / 100.0
            monthly_predictions = predictions.get('monthly_predictions') or {}
            if monthly_predictions.get('valores_pronostico'):
                monthly_predictions['valores_pronostico'] = [
                    (v * factor) if v is not None else v
                    for v in monthly_predictions['valores_pronostico']
                ]
            
            # Crear el Excel
            excel_workbook = self.procesar_datos_y_crear_excel(mothly_demands, predictions)
            
            # Guardar el Excel en un buffer de memoria
            excel_buffer = io.BytesIO()
            excel_workbook.save(excel_buffer)
            excel_buffer.seek(0)
            
            return excel_buffer
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing Excel: {str(e)}")
        
    def create_day_behavior_excel(self, day_behavior_data, end_year):
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Comportamiento_Diario"
        
        # Configurar estilos
        header_font = Font(bold=True)
        center_alignment = Alignment(horizontal="center", vertical="center")
        border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))
        
        # Headers (IGUAL que función original)
        headers = ["AÑO", "MES", "Dias\\Horas"]
        headers.extend([f"H{str(i).zfill(2)}" for i in range(1, 25)])  # H01 a H24
        headers.append("#dias")
        
        try:
            # Aplicar headers
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col)
                cell.value = header
                cell.font = header_font
                cell.alignment = center_alignment
                cell.border = border
            
            # Llenar datos
            current_row = 2
            # Mapeo correcto de tipos (IGUAL que función original)
            tipos_fecha = {
                0: "LABORAL",
                1: "SABADO",
                2: "DOMINGO/FESTIVO"
            }
            
            # CAMBIO: Acceder a day_behavior_data['data'] en lugar de day_behavior_data directamente
            data_list = day_behavior_data['data'] if isinstance(day_behavior_data, dict) else day_behavior_data
            
            for month_data in data_list:
                mes = month_data['mes']
                
                for tipo in month_data['tipos']:
                    tipo_fecha = tipo['tipo_fecha']
                    
                    # Año, Mes y Tipo de día (IGUAL que función original)
                    ws.cell(row=current_row, column=1, value=end_year)
                    ws.cell(row=current_row, column=2, value=mes)
                    ws.cell(row=current_row, column=3, value=tipos_fecha[tipo_fecha])
                    
                    # Valores por hora (IGUAL que función original)
                    for hora in range(1, 25):
                        value = tipo['horas'][f'hora_{hora}']
                        ws.cell(row=current_row, column=hora + 3, value=value)
                    
                    # Número de días según el tipo correcto (IGUAL que función original)
                    if tipo_fecha == 0:
                        dias = month_data['dias_laborales']
                    elif tipo_fecha == 1:
                        dias = month_data['sabados']
                    else:  # tipo_fecha == 2
                        dias = month_data['domingos_festivos']
                    
                    ws.cell(row=current_row, column=28, value=dias)
                    
                    # Aplicar formato (IGUAL que función original)
                    for col in range(1, 29):
                        cell = ws.cell(row=current_row, column=col)
                        cell.border = border
                        cell.alignment = center_alignment
                        if col > 3 and col < 28:  # Columnas de horas
                            # CAMBIO: Actualizar formato para valores en MWh
                            cell.number_format = '#,##0.000000'  # Mejor formato para MWh
                    
                    current_row += 1
            
            # NUEVO: Agregar hoja adicional con información del pronóstico de referencia
            if isinstance(day_behavior_data, dict) and 'summary' in day_behavior_data:
                ws_summary = wb.create_sheet("Resumen_Pronostico")
                
                # Headers de resumen
                ws_summary.cell(row=1, column=1, value="Información del Pronóstico")
                ws_summary.cell(row=1, column=1).font = Font(bold=True, size=14)
                
                summary = day_behavior_data['summary']
                forecast_ref = summary.get('forecast_reference', {})
                
                # Información básica
                info_rows = [
                    ("Período:", f"{summary.get('start_date', '')} to {summary.get('end_date', '')}"),
                    ("Lag (años):", summary.get('lag_years', 0)),
                    ("Tipo de modelo:", forecast_ref.get('model_type', 'N/A')),
                    ("ID del modelo:", forecast_ref.get('model_id', 'N/A')),
                    ("Descripción:", forecast_ref.get('description', 'N/A')),
                    ("Método usado:", forecast_ref.get('method_used', 'N/A')),
                    ("Total pronóstico (MWh):", forecast_ref.get('total_forecast_mwh', 0))
                ]
                
                row = 3
                for label, value in info_rows:
                    ws_summary.cell(row=row, column=1, value=label).font = Font(bold=True)
                    ws_summary.cell(row=row, column=2, value=value)
                    row += 1
                
                # Detalle mensual del pronóstico
                if 'forecast_dates' in forecast_ref and 'forecast_values_mwh' in forecast_ref:
                    row += 2
                    ws_summary.cell(row=row, column=1, value="Pronóstico Mensual (MWh)").font = Font(bold=True)
                    row += 1
                    
                    ws_summary.cell(row=row, column=1, value="Mes").font = Font(bold=True)
                    ws_summary.cell(row=row, column=2, value="Demanda (MWh)").font = Font(bold=True)
                    row += 1
                    
                    for i, (date, value) in enumerate(zip(forecast_ref['forecast_dates'], forecast_ref['forecast_values_mwh'])):
                        ws_summary.cell(row=row, column=1, value=date)
                        ws_summary.cell(row=row, column=2, value=value)
                        ws_summary.cell(row=row, column=2).number_format = '#,##0.00'
                        row += 1
                
                # Ajustar anchos de columna para la hoja de resumen
                ws_summary.column_dimensions['A'].width = 25
                ws_summary.column_dimensions['B'].width = 20
            
            # Ajustar anchos de columna (IGUAL que función original)
            for col in range(1, 29):
                column_letter = get_column_letter(col)
                ws.column_dimensions[column_letter].width = 12
            
            ws.column_dimensions['C'].width = 15
            
            return wb
        except Exception as e:
            print(f"Error en create_day_behavior_excel: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def day_behavior_excel(self, search: DayBehaviorSchema, db: Session):
        try:
            day_behavior_data = self.get_day_behavior(search, db)
            if not day_behavior_data:
                raise ValueError("No se encontraron datos para el período especificado")
                
            print("Creating Excel workbook...")
            end_date = pd.to_datetime(search.end_date)
            excel_workbook = self.create_day_behavior_excel(day_behavior_data, end_date.year)
            
            print("Saving to buffer...")
            excel_buffer = io.BytesIO()
            excel_workbook.save(excel_buffer)
            excel_buffer.seek(0)
            
            return excel_buffer
        except Exception as e:
            print(f"Error completo: {str(e)}")
            print(f"Tipo de error: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing Excel: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
        
    def update_monthly_type(self, search: UpdateMonthlyTypeSchema, db: Session):
        try:
            monthly_json = search.values 
            for year in monthly_json:
                for month in monthly_json[year]:
                    db.query(MonthlyDemand).filter_by(year=year, month=month).update({
                        "climate_type": monthly_json[year][month]
                    })

            db.commit()
            return {"message": "Datos actualizados exitosamente"}
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al actualizar datos mensuales: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al actualizar datos mensuales: {str(e)}")
            raise e
        

    def get_last_year_on_db(self, db: Session):
        try:
            demand = DemandService()
            year_range = demand.get_first_and_last_date(db)
            last_date = year_range["last_date"].strftime('%Y-%m-%d')
            last_year = int(pd.to_datetime(last_date).year)
            return {
                "last_year": last_year
            }
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener último año: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener último año: {str(e)}")
            raise e

    def list_historic_years(self, db: Session):
        try:
            # Extract all years from December 31st dates
            years = db.query(func.extract('year', Demand.fecha).label('year'))\
                     .filter(func.extract('month', Demand.fecha) == 12,
                             func.extract('day', Demand.fecha) == 31)\
                     .distinct()\
                     .order_by('year')\
                     .all()
            
            return sorted([int(year[0]) for year in years])
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener años históricos: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener años históricos: {str(e)}")
            raise e

    def get_monthly_demand_by_year(self, year: int, db: Session):
        try:
            monthly_demand = db.query(MonthlyDemand).filter_by(year=year).all()
            if not monthly_demand:
                raise ValueError("No se encontraron datos para el año especificado.")
            
            monthly_data = {}
            monthly_data = {}
            
            # Create entries with YYYY-MM format
            for demand in monthly_demand:
                month_str = f"0{demand.month}" if demand.month < 10 else str(demand.month)
                key = f"{year}-{month_str}"
                monthly_data[key] = {
                    "value": demand.value,
                    "percentage": demand.percentage,
                    "climate_type": demand.climate_type
                }
            
            # Sort the dictionary by keys (which are now in YYYY-MM format)
            sorted_data = {k: monthly_data[k] for k in sorted(monthly_data.keys())}
            
            return sorted_data
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al obtener demanda mensual por año: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al obtener demanda mensual por año: {str(e)}")
            raise e
        
    def change_model_monthly_type(self, search: ForecastTypeMonthSchema, db: Session):
        try:
            model_id = search.model_id
            dates = search.dates
            types = search.types
            model = db.query(UsersModelsValues).filter_by(model_id=model_id).first()
            if not model:
                raise ValueError("El modelo especificado no existe.")
            
            def change_type_month_value(value, current_type, new_type):
                if current_type == "NORMAL":
                    if new_type == "NIÑA":
                        return value * 0.9946
                    elif new_type == "NIÑO":
                        return value * 1.0104
                    else:
                        return value
                elif current_type == "NIÑA":
                    if new_type == "NORMAL":
                        return value / 0.9946
                    elif new_type == "NIÑO":
                        return ( value / 1.0104 ) * 0.9946
                    else:
                        return value
                elif current_type == "NIÑO":
                    if new_type == "NORMAL":
                        return value / 1.0104
                    elif new_type == "NIÑA":
                        return ( value / 0.9946 ) * 1.0104
                    else:
                        return value
            
            model_user = db.query(UsersModelsValues).filter_by(model_id=model_id).all()
            model_dates = [model.date.strftime('%Y-%m-%d') for model in model_user]
            model_values = [model.value for model in model_user]
            model_types = [model.climate_type for model in model_user]
            list_model = zip(model_dates, model_values, model_types)
            for item in list_model:
                date = item[0]
                value = item[1]
                current_type = item[2]
                if date in dates:
                    index = dates.index(date)
                    new_type = types[index]
                    new_value = change_type_month_value(value, current_type, new_type)
                    db.query(UsersModelsValues).filter_by(model_id=model_id, date=pd.to_datetime(date).date()).update({
                        "value": new_value,
                        "climate_type": new_type
                    })
            db.commit()

            return self.retrieve_user_model_values(model_id=model_id, db=db)
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al cambiar tipo de modelo mensual: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al cambiar tipo de modelo mensual: {str(e)}")
            raise e
    
    def change_model_based_on_year(self, search: ForecastModelSaveBasedOnYear, db: Session):
        try:
            model_id = search.model_id
            year = search.year
            predict_year = search.predict_year
            model_values = db.query(UsersModelsValues).filter_by(model_id=model_id).all()
            if not model_values:
                raise ValueError("El modelo especificado no existe.")
            year_info = self.get_monthly_demand_by_year(year, db)

            model_values_dates = [model.date for model in model_values if str(predict_year) in str(model.date)]
            if not model_values_dates:
                raise ValueError("No se encontraron fechas para el modelo especificado.")
            months_to_change = [model.date.month for model in model_values if str(predict_year) in str(model.date)]
            model_values = [model.value for model in model_values if str(predict_year) in str(model.date)]

            if len(months_to_change) != 12:
                partial_total = 0
                for month in months_to_change:
                    partial_total += year_info[f"{year}-{str(month).zfill(2)}"]["value"]
                partial_percentage_dict = {}
                for month in months_to_change:
                    partial_percentage_dict[month] = year_info[f"{year}-{str(month).zfill(2)}"]["value"] / partial_total

                percentages = partial_percentage_dict
            else:
                percentages = {}
                for month in months_to_change:
                    percentages[month] = year_info[f"{year}-{str(month).zfill(2)}"]["percentage"]

            total_model = sum(model_values)
            model_to_change = zip(model_values_dates, model_values, months_to_change)
            for month in months_to_change:
                date = model_values_dates[months_to_change.index(month)]
                if isinstance(date, pd.Timestamp):
                    date = date.date()
                elif isinstance(date, datetime):
                    date = date.date()
                # Si ya es date, no pasa nada
                value = model_values[months_to_change.index(month)]
                percentage = percentages[month]
                new_value = total_model * percentage
                db.query(UsersModelsValues).filter_by(model_id=model_id, date=date).update({
                    "value": new_value,
                    "climate_type": year_info[f"{year}-{str(month).zfill(2)}"]["climate_type"]
                })
            db.commit()
            return self.retrieve_user_model_values(model_id=model_id, db=db)
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error al cambiar modelo basado en año: {str(e)}")
            return {"message": "Ocurrió un error al procesar la solicitud.", "status": "error"}
        except Exception as e:
            print(f"Error al cambiar modelo basado en año: {str(e)}")
            raise e
            




            