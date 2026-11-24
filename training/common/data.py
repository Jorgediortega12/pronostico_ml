"""Capa de datos limpia y única para el forecast mensual de demanda.

Lee ``PRONOSTICO_demands`` (una fila por día, kWh) y produce una serie mensual con banderas de calidad:

* **Deduplicado por fecha** (2020-Q4 tiene 2 filas por día, una ~10× menor): se elige la fila cuya escala
  coincide con los días vecinos no duplicados (criterio verificable; ver :func:`deduplicate_daily`).
* **Meses incompletos** (p. ej. 2021-09 con 28/30 días) y **preliminares** (filas no ``txf``, p. ej. 2026-11
  ``tx2``) quedan marcados. ``demanda_kwh`` SOLO tiene valor cuando el mes es usable como target
  (completo y definitivo, o imputado si se pide); la suma parcial queda en ``demanda_observada_kwh``.
* **Nunca se inventa futuro**: la imputación solo rellena huecos internos del histórico.
* **Quiebre de mercado** (``demand_split_config``): se expone como :class:`MarketBreak`, con la misma
  semántica que ``api/services/demand_config_service.adjust_period_value``.

Unidad: kWh en todo el módulo.
"""
from __future__ import annotations

import calendar
import logging
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from training.common import config

logger = logging.getLogger(__name__)

EngineLike = Union[Engine, str, None]

MONTHLY_COLUMNS = [
    "demanda_kwh",
    "demanda_observada_kwh",
    "dias_con_dato",
    "dias_mes",
    "completo",
    "preliminar",
    "imputado",
]


# --------------------------------------------------------------------------------------------------
# Conexión
# --------------------------------------------------------------------------------------------------
def get_engine(engine_or_url: EngineLike = None) -> Engine:
    """Devuelve un ``Engine``: el recibido, uno nuevo desde la URL, o desde ``ML_DATABASE_URL``/.env."""
    if isinstance(engine_or_url, Engine):
        return engine_or_url
    return create_engine(engine_or_url or config.database_url())


# --------------------------------------------------------------------------------------------------
# Diario
# --------------------------------------------------------------------------------------------------
def read_raw_daily(engine_or_url: EngineLike = None, *, connection=None) -> pd.DataFrame:
    """Lee las filas crudas de la tabla de demanda (sin deduplicar; se omiten filas con ``total`` NULL).

    Columnas: ``id``, ``fecha`` (Timestamp a día), ``total`` (kWh, float), ``suma_horas`` (Σ hora_1..24),
    ``file_type``.

    ``connection``: conexión SQLAlchemy ya abierta (p. ej. ``session.connection()`` de la petición en la API);
    si se pasa, se usa en lugar de ``engine_or_url`` y no se cierra.
    """
    hsum = " + ".join(f"coalesce(hora_{i}, 0)" for i in range(1, 25))
    q = text(
        f'SELECT id, fecha::date AS fecha, total, ({hsum}) AS suma_horas, file_type '
        f'FROM "{config.DEMANDS_TABLE}" WHERE total IS NOT NULL ORDER BY fecha, id'
    )
    if connection is not None:
        raw = pd.read_sql(q, connection)
    else:
        with get_engine(engine_or_url).connect() as conn:
            raw = pd.read_sql(q, conn)
    raw["fecha"] = pd.to_datetime(raw["fecha"])
    raw["total"] = pd.to_numeric(raw["total"], errors="coerce").astype(float)
    return raw.dropna(subset=["total"])


def deduplicate_daily(
    raw: pd.DataFrame, window_days: int = config.DEDUP_WINDOW_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deja una sola fila por fecha.

    Criterio (verificable con el log devuelto):

    1. Si en la fecha hay filas del tipo definitivo (``config.FINAL_FILE_TYPE``) se descartan las preliminares.
    2. Si aún quedan varias, se toma como referencia la mediana del ``total`` de los días **no duplicados**
       dentro de ±``window_days`` días (si no hay ninguno, la mediana global de días no duplicados) y se
       elige el candidato con menor ``|log(total / referencia)|``. Es decir, la fila cuya escala es
       coherente con los días vecinos; no depende del orden de ``id`` ni de "la mayor".

    Args:
        raw: filas con al menos ``fecha``, ``total`` y ``file_type`` (``id`` opcional).
        window_days: semiventana de días para la referencia.

    Returns:
        ``(diario, log)``. ``diario`` tiene índice ``fecha`` y columnas ``demanda_kwh``, ``file_type``,
        ``n_filas``. ``log`` tiene una fila por fecha duplicada: ``fecha``, ``candidatos`` (totales),
        ``referencia``, ``elegido``, ``razon_max_min`` y ``criterio``.
    """
    df = raw.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    counts = df.groupby("fecha")["total"].transform("size")
    unique = df.loc[counts == 1].set_index("fecha")["total"].sort_index()
    global_ref = float(unique.median()) if len(unique) else np.nan

    keep_idx: list[int] = df.index[counts == 1].tolist()
    log_rows = []
    win = pd.Timedelta(days=window_days)
    for fecha, grp in df.loc[counts > 1].groupby("fecha"):
        cand = grp
        criterio = "escala_vecinos"
        finals = grp[grp["file_type"] == config.FINAL_FILE_TYPE]
        if 0 < len(finals) < len(grp):
            cand = finals
            criterio = "prefiere_definitivo"
        if len(cand) == 1:
            ref = np.nan
            chosen = cand.index[0]
        else:
            vecinos = unique.loc[fecha - win: fecha + win]
            ref = float(vecinos.median()) if len(vecinos) else global_ref
            if not len(vecinos):
                criterio = "escala_global"
            tot = cand["total"].astype(float)
            dist = np.where(tot > 0, np.abs(np.log(tot.clip(lower=1e-12) / ref)), np.inf)
            chosen = cand.index[int(np.argmin(dist))]
        keep_idx.append(chosen)
        totals = grp["total"].astype(float)
        log_rows.append(
            {
                "fecha": fecha,
                "candidatos": [round(v, 3) for v in totals.tolist()],
                "ids": grp["id"].tolist(),
                "referencia": ref,
                "elegido": float(df.at[chosen, "total"]),
                "id_elegido": df.at[chosen, "id"],
                "razon_max_min": float(totals.max() / totals.min()) if totals.min() > 0 else np.inf,
                "criterio": criterio,
            }
        )
    out = df.loc[sorted(keep_idx)].copy()
    out["n_filas"] = out["fecha"].map(df.groupby("fecha").size())
    out = out.rename(columns={"total": "demanda_kwh"}).set_index("fecha").sort_index()
    log = pd.DataFrame(
        log_rows,
        columns=["fecha", "candidatos", "ids", "referencia", "elegido", "id_elegido", "razon_max_min", "criterio"],
    )
    return out[["demanda_kwh", "file_type", "n_filas"]], log


def load_daily_demand(
    engine_or_url: EngineLike = None, window_days: int = config.DEDUP_WINDOW_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Lee y deduplica la demanda diaria.

    Returns:
        ``(diario, log_dedupe, n_total_vs_horas)`` donde el último es el número de filas crudas con
        ``|total − Σ hora_i| > config.TOTAL_VS_HOURS_TOL_KWH`` (esperado 0: ``total`` es la fuente).
    """
    raw = read_raw_daily(engine_or_url)
    mismatch = int((raw["total"] - raw["suma_horas"]).abs().gt(config.TOTAL_VS_HOURS_TOL_KWH).sum())
    if mismatch:
        warnings.warn(f"{mismatch} filas con total != suma de horas; se usa `total`.", stacklevel=2)
    daily, log = deduplicate_daily(raw, window_days)
    return daily, log, mismatch


# --------------------------------------------------------------------------------------------------
# Mensual
# --------------------------------------------------------------------------------------------------
def aggregate_monthly(
    daily: pd.DataFrame, impute_incomplete: bool = False, min_fraction_to_impute: float = 0.5
) -> pd.DataFrame:
    """Agrega la demanda diaria deduplicada a meses.

    Args:
        daily: salida de :func:`deduplicate_daily` (índice fecha, columnas ``demanda_kwh``, ``file_type``).
        impute_incomplete: si ``True``, los meses incompletos que son **huecos internos** del histórico
            (definitivos, con al menos ``min_fraction_to_impute`` de los días y seguidos por algún mes
            completo) se imputan como promedio diario × días del mes. Nunca se imputa el último mes
            parcial, un mes preliminar, ni meses sin ningún dato.
        min_fraction_to_impute: fracción mínima de días observados para permitir la imputación.

    Returns:
        DataFrame con índice ``mes`` (inicio de mes, frecuencia mensual continua desde el primer al
        último mes con dato) y columnas :data:`MONTHLY_COLUMNS`. ``demanda_kwh`` es NaN cuando el mes no
        es usable como target (incompleto sin imputar o preliminar).
    """
    d = daily.copy()
    d.index = pd.to_datetime(d.index).normalize()
    mes = d.index.to_period("M").to_timestamp()
    g = d.groupby(mes)
    m = pd.DataFrame(
        {
            "demanda_observada_kwh": g["demanda_kwh"].sum(),
            "dias_con_dato": g["demanda_kwh"].size(),
            "preliminar": g["file_type"].agg(lambda s: bool((s != config.FINAL_FILE_TYPE).any())),
        }
    )
    full = pd.date_range(m.index.min(), m.index.max(), freq="MS")
    m = m.reindex(full)
    m.index.name = "mes"
    m["dias_con_dato"] = m["dias_con_dato"].fillna(0).astype(int)
    m["preliminar"] = m["preliminar"].fillna(False).astype(bool)
    m["dias_mes"] = m.index.days_in_month.astype(int)
    m["completo"] = m["dias_con_dato"] == m["dias_mes"]
    usable = m["completo"] & ~m["preliminar"]
    m["demanda_kwh"] = m["demanda_observada_kwh"].where(usable)
    m["imputado"] = False

    if impute_incomplete:
        later_complete = usable[::-1].cummax()[::-1].shift(-1, fill_value=False)
        candidate = (
            ~m["completo"]
            & ~m["preliminar"]
            & (m["dias_con_dato"] >= np.ceil(min_fraction_to_impute * m["dias_mes"]))
            & (m["dias_con_dato"] > 0)
            & later_complete
        )
        m.loc[candidate, "demanda_kwh"] = (
            m.loc[candidate, "demanda_observada_kwh"] / m.loc[candidate, "dias_con_dato"] * m.loc[candidate, "dias_mes"]
        )
        m.loc[candidate, "imputado"] = True
    return m[MONTHLY_COLUMNS]


def load_monthly_demand(
    engine_or_url: EngineLike = None,
    impute_incomplete: bool = False,
    window_days: int = config.DEDUP_WINDOW_DAYS,
) -> pd.DataFrame:
    """Demanda mensual limpia desde la BD (ver :func:`aggregate_monthly` para las columnas).

    En ``df.attrs`` quedan: ``dedup_log`` (DataFrame), ``n_total_vs_horas`` (int) y ``market_break``
    (:class:`MarketBreak` o ``None``). Para el target use :func:`target_series`.
    """
    eng = get_engine(engine_or_url)
    daily, log, mismatch = load_daily_demand(eng, window_days)
    monthly = aggregate_monthly(daily, impute_incomplete=impute_incomplete)
    monthly.attrs["dedup_log"] = log
    monthly.attrs["n_total_vs_horas"] = mismatch
    monthly.attrs["market_break"] = load_market_break(eng)
    return monthly


def target_series(
    monthly: pd.DataFrame,
    include_imputed: bool = True,
    until: Optional[Union[str, pd.Timestamp]] = None,
    market_break: Optional["MarketBreak"] = None,
) -> pd.Series:
    """Serie mensual usable como target (kWh), sin meses incompletos/preliminares.

    Args:
        monthly: salida de :func:`load_monthly_demand` / :func:`aggregate_monthly`.
        include_imputed: ``False`` para usarla como "real" en métricas (solo meses observados completos).
        until: último mes inclusivo (p. ej. un origen).
        market_break: si se pasa, se cortan los meses posteriores al quiebre (escala distinta).

    Puede tener huecos (meses NaN eliminados): el índice es inicio de mes pero no necesariamente continuo.
    """
    s = monthly["demanda_kwh"]
    if not include_imputed:
        s = s.where(~monthly["imputado"])
    if until is not None:
        s = s.loc[: pd.Timestamp(until).to_period("M").to_timestamp()]
    if market_break is not None:
        s = s.loc[s.index < market_break.first_post_month]
    return s.dropna().rename("demanda_kwh")


# --------------------------------------------------------------------------------------------------
# Quiebre de mercado
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketBreak:
    """División de mercado leída de ``demand_split_config``.

    Semántica (igual a ``adjust_period_value`` de la API): los periodos cuyo **último día** es
    ``<= fecha_limite`` están en la escala de mercado completo; los posteriores ya son solo la porción
    del nuevo mercado, que equivale a ``porcentaje`` % del mercado completo.
    """

    fecha_limite: date
    porcentaje: float
    fuente: str = config.SPLIT_TABLE

    @property
    def factor(self) -> float:
        """Fracción (0–1] del mercado completo que representan los meses post-quiebre."""
        return self.porcentaje / 100.0

    @property
    def first_post_month(self) -> pd.Timestamp:
        """Primer mes (inicio de mes) cuyo último día es posterior a ``fecha_limite``."""
        fl = pd.Timestamp(self.fecha_limite)
        last_day = calendar.monthrange(fl.year, fl.month)[1]
        start = fl.to_period("M").to_timestamp()
        # El mes de fecha_limite es pre-quiebre solo si fecha_limite es su último día.
        return start + pd.offsets.MonthBegin(1) if fl.day == last_day else start

    def is_post(self, months: Union[pd.DatetimeIndex, pd.Series, pd.Timestamp, date, str]):
        """Máscara booleana: ``True`` para meses posteriores al quiebre. Con un solo mes (Timestamp, date o
        str) devuelve un ``bool``."""
        if isinstance(months, (pd.Timestamp, date, str, np.datetime64)):
            return bool(pd.Timestamp(months).to_period("M").to_timestamp() >= self.first_post_month)
        idx = pd.DatetimeIndex(months).to_period("M").to_timestamp()
        return np.asarray(idx >= self.first_post_month)


def load_market_break(engine_or_url: EngineLike = None) -> Optional[MarketBreak]:
    """Lee la primera fila de ``demand_split_config``. ``None`` si no existe la tabla o está vacía.

    A diferencia de la API (que cae a hoy/100 %), aquí ``None`` deja explícito que no hay configuración.
    """
    try:
        with get_engine(engine_or_url).connect() as conn:
            row = conn.execute(
                text(f"SELECT fecha_limite, porcentaje FROM {config.SPLIT_TABLE} ORDER BY id ASC LIMIT 1")
            ).first()
    except Exception as exc:  # tabla inexistente / sin permisos
        logger.info("sin %s (%s)", config.SPLIT_TABLE, type(exc).__name__)
        return None
    if not row or row[0] is None or row[1] is None:
        return None
    return MarketBreak(fecha_limite=pd.Timestamp(row[0]).date(), porcentaje=float(row[1]))


def to_full_market_scale(series: pd.Series, market_break: Optional[MarketBreak]) -> pd.Series:
    """Lleva los meses post-quiebre a la escala de mercado completo: ``valor / factor``.

    Límites (leer antes de usar):

    * Usa el ``porcentaje`` **configurado**, nunca uno estimado de los datos: con ~2 meses post-quiebre
      cualquier factor inferido mezcla el quiebre con estacionalidad y crecimiento → inestable.
    * Si la configuración está desactualizada (p. ej. la BD local tiene 100 % mientras los datos muestran
      una caída de ~30 % desde 2026-09) la conversión es un no-op y la serie sigue teniendo el escalón.
      Por eso los modelos deben preferir entrenar con ``target_series(..., market_break=mb)`` (corte en
      el quiebre) y comparar contra reales post-quiebre con :func:`from_full_market_scale`.
    * No corrige diferencias de composición (el nuevo mercado no tiene por qué tener la misma
      estacionalidad ni crecimiento que el total).
    """
    if market_break is None:
        return series.copy()
    if market_break.factor <= 0:
        raise ValueError("porcentaje de división de mercado debe ser > 0")
    if market_break.factor == 1.0:
        warnings.warn("porcentaje = 100 %: to_full_market_scale no cambia la serie.", stacklevel=2)
    out = series.astype(float).copy()
    post = market_break.is_post(out.index)
    out[post] = out[post] / market_break.factor
    return out


def from_full_market_scale(series: pd.Series, market_break: Optional[MarketBreak]) -> pd.Series:
    """Inversa de :func:`to_full_market_scale`: pasa un pronóstico en escala completa a la escala de la BD
    (``valor × factor`` en meses post-quiebre). Útil para comparar contra reales post-quiebre."""
    if market_break is None:
        return series.copy()
    out = series.astype(float).copy()
    post = market_break.is_post(out.index)
    out[post] = out[post] * market_break.factor
    return out
