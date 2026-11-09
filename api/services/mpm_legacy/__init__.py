"""Motor puro MPM con surrogate climático calibrado."""

from .city_map import LEGACY_CITY_MAP, REQUIRED_TEMP_KEYS
from .climate_adjust import LINEAR_INTERCEPT, LINEAR_SLOPE
from .climate_adjuster import (
    CalibratedCurveAdjuster,
    ClimateAdjuster,
    FallbackChainAdjuster,
    LinearFallbackAdjuster,
    get_default_adjuster,
)
from .climate_fill import SCENARIO_NAMES
from .day_types import TipoDia, day_type, matlab_weekday
from .engine import METHOD_IDS, PredictMonthResult, predict_month
from .prom_dia import PromDia, calculate_prom_dia
from .similar_days import DESV_NO_MATCH, DESV_THRESHOLD

__all__ = [
    "DESV_NO_MATCH",
    "DESV_THRESHOLD",
    "LEGACY_CITY_MAP",
    "LINEAR_INTERCEPT",
    "LINEAR_SLOPE",
    "CalibratedCurveAdjuster",
    "ClimateAdjuster",
    "FallbackChainAdjuster",
    "LinearFallbackAdjuster",
    "METHOD_IDS",
    "PredictMonthResult",
    "PromDia",
    "REQUIRED_TEMP_KEYS",
    "SCENARIO_NAMES",
    "TipoDia",
    "calculate_prom_dia",
    "day_type",
    "get_default_adjuster",
    "matlab_weekday",
    "predict_month",
]
