"""Compatibilidad de ajustes climáticos y búsqueda de referencias legacy."""

from __future__ import annotations

from typing import Sequence

from .climate_adjuster import (
    LINEAR_INTERCEPT,
    LINEAR_SLOPE,
    LinearFallbackAdjuster,
    get_default_adjuster,
)

WH_SCALE = 1e6


def linear_delta_demand(delta_t: float) -> float:
    """dD = 1.099*dT - 0.01963 (resultado en GW; *1e6 → Wh)."""

    return LINEAR_SLOPE * float(delta_t) + LINEAR_INTERCEPT


def apply_linear_baq(base_demand_wh: float, delta_t: float) -> float:
    return float(base_demand_wh) + linear_delta_demand(delta_t) * WH_SCALE


def stub_rn_baq(base_demand_wh: float, delta_t: float) -> float:
    """Wrapper histórico hacia el corrector activo."""

    adjustment = get_default_adjuster().adjust_baq(delta_t)
    return float(base_demand_wh) + adjustment * WH_SCALE


def stub_rn_3c(
    base_demand_wh: float,
    delta_temps: Sequence[float],
) -> float:
    """Wrapper histórico hacia el corrector 3C activo."""

    if not delta_temps:
        return float(base_demand_wh)
    if len(delta_temps) != 3:
        avg_delta = sum(float(value) for value in delta_temps) / len(delta_temps)
        adjustment = LinearFallbackAdjuster().adjust_baq(avg_delta)
    else:
        adjustment = get_default_adjuster().adjust_3c(*delta_temps)
    return float(base_demand_wh) + adjustment * WH_SCALE


def find_climate_reference(
    *,
    target_weekday: int,
    target_tipo: int,
    target_temp_baq: float,
    ref_weekdays: Sequence[int],
    ref_tipos: Sequence[int],
    ref_temps_baq: Sequence[float],
) -> tuple[int, bool] | None:
    """Busca índice de referencia.

    Retorna ``(idx, exact_match)``. Si hay match exacto en (TipoDia, temp),
    el fragmento MATLAB no aplica ajuste; solo lo hace en fallback ordinario.
    """

    exact = [
        idx
        for idx, (tipo, temp) in enumerate(zip(ref_tipos, ref_temps_baq))
        if tipo == target_tipo and temp == target_temp_baq
    ]
    if exact:
        return exact[0], True

    if target_tipo != 0:
        return None

    same_weekday = [
        idx
        for idx, (tipo, weekday) in enumerate(zip(ref_tipos, ref_weekdays))
        if tipo == target_tipo and weekday == target_weekday
    ]
    if same_weekday:
        return same_weekday[-1], False

    same_tipo = [idx for idx, tipo in enumerate(ref_tipos) if tipo == target_tipo]
    if not same_tipo:
        return None

    # MATLAB: pos = f - ProData(i,1), donde f es un índice de fila 1-based
    # sobre RefData y ProData(i,1) es el weekday del día a pronosticar.
    distances = [(idx + 1) - target_weekday for idx in same_tipo]
    preferred = [idx for idx, dist in zip(same_tipo, distances) if dist == -1]
    if preferred:
        return preferred[-1], False
    preferred = [idx for idx, dist in zip(same_tipo, distances) if dist == 1]
    if preferred:
        return preferred[-1], False
    min_abs = min(abs(dist) for dist in distances)
    closest = [
        idx for idx, dist in zip(same_tipo, distances) if abs(dist) == min_abs
    ]
    return closest[-1], False
