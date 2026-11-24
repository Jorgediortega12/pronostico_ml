"""Historia mensual limpia de demanda para la inferencia de los modelos v2 (ids 1, 2, 3).

La demanda se lee con la **misma sesión SQLAlchemy de la petición** (la base de la empresa indicada por
``database_url``), nunca con un engine global. La limpieza reutiliza las funciones puras de
``training.common.data`` (el paquete ``training/`` se copia en la imagen Docker), así que train y serve
comparten exactamente las mismas reglas:

* deduplicado por fecha (2020-Q4 tiene filas repetidas a ~1/10 de escala),
* solo meses completos y definitivos (``txf``) como dato; huecos internos con >= 50 % de días se imputan
  (p. ej. 2021-09) solo para usarlos como historia del modelo,
* corte en el quiebre de mercado (``demand_split_config``): los meses posteriores a ``fecha_limite`` están en
  otra escala (porción del nuevo mercado) y no se usan como historia.

Unidad: kWh, escala de mercado completo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from training.common.data import MarketBreak, aggregate_monthly, deduplicate_daily, read_raw_daily, target_series


@dataclass(frozen=True)
class DivisionMercado(MarketBreak):
    """División de mercado de la petición: es el mismo :class:`training.common.data.MarketBreak` (misma
    semántica que ``adjust_period_value`` y el front) con la serialización para la respuesta de la API.

    Un periodo cuyo último día es ``<= fecha_limite`` está en la escala de mercado completo; los posteriores ya
    son la porción del nuevo mercado (``factor`` = ``porcentaje``/100 del mercado completo). El consumidor del
    pronóstico (front / ``predict_excel``) multiplica TODO el pronóstico por ``factor``.
    ``is_post(meses)`` (heredado) devuelve una máscara booleana para un índice/serie de meses.
    """

    fuente: str = "demand_split_config (DemandConfigService)"

    def as_dict(self) -> dict:
        return {
            "fecha_limite": pd.Timestamp(self.fecha_limite).strftime("%Y-%m-%d"),
            "porcentaje": float(self.porcentaje),
            "factor": self.factor,
            "primer_mes_post_division": self.first_post_month.strftime("%Y-%m-%d"),
        }


def division_from_config(config) -> DivisionMercado:
    """Convierte el ``DemandSplitConfig`` de ``DemandConfigService`` (fecha_limite, porcentaje)."""
    return DivisionMercado(fecha_limite=pd.Timestamp(config.fecha_limite).date(), porcentaje=float(config.porcentaje))


def read_raw_daily_session(db: Session) -> pd.DataFrame:
    """Filas crudas de ``PRONOSTICO_demands`` con la sesión de la petición: delega en
    :func:`training.common.data.read_raw_daily` (misma consulta y filtro que el entrenamiento)."""
    return read_raw_daily(connection=db.connection())


def clean_monthly_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Mensual limpio (columnas de ``training.common.data.MONTHLY_COLUMNS``) desde filas diarias crudas."""
    if raw.empty:
        raise ValueError("No hay datos de demanda diaria en la base de la petición")
    daily, _log = deduplicate_daily(raw)
    return aggregate_monthly(daily, impute_incomplete=True)


@dataclass
class HistoriaLimpia:
    """Historia de entrenamiento/inferencia (``serie``, con imputados, cortada en la división) y reales
    observados completos (``reales``, sin imputados, en la escala en que están en la BD)."""

    serie: pd.Series
    reales: pd.Series
    division: Optional[DivisionMercado]

    @property
    def corte(self) -> Optional[pd.Timestamp]:
        return self.division.first_post_month if self.division is not None else None


def load_clean_history_session(db: Session, division: Optional[DivisionMercado]) -> HistoriaLimpia:
    """Historia limpia desde la base de la petición, cortada en el primer mes post-división."""
    monthly = clean_monthly_from_raw(read_raw_daily_session(db))
    serie = target_series(monthly)
    if division is not None:
        serie = serie[serie.index < division.first_post_month]
    reales = target_series(monthly, include_imputed=False)
    return HistoriaLimpia(serie=serie, reales=reales, division=division)
