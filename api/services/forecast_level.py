"""Integracion en la API de la correccion adaptativa de nivel (``training.common.level_adapt``).

Opt-in: solo actua con la variable de entorno ``LEVEL_ADAPT=1`` (por defecto APAGADA; ver
``docs/work/11_nivel_adaptativo/REPORTE.md``). Con el flag apagado ``forecast.py`` solo consulta
:func:`applies_to` y el resultado es identico al de antes de este cambio.

La spec y los modelos (ids) afectados se leen de ``api/models/level_adapt.json``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from training.common.level_adapt import LevelResult, LevelSpec, adjust, apply_level, covid_exclusions

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "models" / "level_adapt.json"
_TRUE = ("1", "true", "si", "sí", "on")


def enabled() -> bool:
    """``LEVEL_ADAPT=1`` activa la correccion (por defecto apagada; ``0``/vacio la apaga)."""
    return os.getenv("LEVEL_ADAPT", "").strip().lower() in _TRUE


def load_config(path: Optional[Path] = None) -> tuple[LevelSpec, frozenset]:
    d = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    return LevelSpec.from_dict(d["spec"]), frozenset(int(x) for x in d.get("models", []))


def applies_to(model_id: int) -> bool:
    if not enabled():
        return False
    try:
        return int(model_id) in load_config()[1]
    except Exception as exc:  # config ausente/corrupta: nunca rompe el pronostico
        logger.warning("level_adapt: no se pudo leer %s (%s); sin ajuste", CONFIG_PATH.name, exc)
        return False


def ref_fn_rfxgb(model_id: int, artifact: dict, fingerprint: str):
    """Pronostico de referencia RF/XGB: reajusta con la historia hasta el origen de referencia (cache existente)."""
    from api.services import forecast_rfxgb

    def ref_fn(hist_ref: pd.Series, h: int) -> pd.Series:
        art = forecast_rfxgb.refit_from_artifact_cached(model_id, hist_ref, artifact, fingerprint=fingerprint)
        return forecast_rfxgb.predict_from_artifact(art, hist_ref, h)

    return ref_fn


def ref_fn_sarimax(corte=None):
    """Pronostico de referencia SARIMAX (parametros re-estimados solo con la historia de referencia)."""
    from api.services import forecast_sarimax

    def ref_fn(hist_ref: pd.Series, h: int) -> pd.Series:
        last = hist_ref.index.max()
        df = forecast_sarimax.predict_monthly(hist_ref, last + pd.offsets.MonthBegin(1), last + pd.offsets.MonthBegin(h),
                                              refit=True, exclude_from=corte)
        return pd.Series(df["prediccion"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["fecha"]))

    return ref_fn


def adjust_forecast(model_id: int, hist: pd.Series, fc: pd.Series, ref_fn) -> tuple[pd.Series, Optional[dict]]:
    """``(pronostico_corregido, info)``; si el flag/modelo no aplica devuelve ``(fc, None)``."""
    if not applies_to(model_id):
        return fc, None
    try:
        spec, _ = load_config()
        res: LevelResult = adjust(hist, fc, ref_fn, spec, covid_exclusions())
        return apply_level(fc, res), res.to_dict()
    except Exception as exc:  # noqa: BLE001 - la correccion es opcional: ante cualquier fallo se sirve el pronostico sin ajustar
        print(f"[level_adapt] correccion omitida (model_id={model_id}): {exc}")
        return fc, None
