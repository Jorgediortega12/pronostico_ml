"""Puerto de correctores residuales climáticos para f5/f6."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


LINEAR_SLOPE = 1.099
LINEAR_INTERCEPT = -0.01963
DT_MIN = -5.0
DT_MAX = 5.0
DD_MIN = -3.0
DD_MAX = 3.0
ARTIFACTS_DIR = Path(__file__).with_name("artifacts")


@runtime_checkable
class ClimateAdjuster(Protocol):
    """Contrato de ajuste; todos los resultados están en GW."""

    def adjust_baq(self, dT: float) -> float: ...

    def adjust_3c(self, dT_baq: float, dT_car: float, dT_sm: float) -> float: ...


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


class LinearFallbackAdjuster:
    """Fallback determinista equivalente al lineal legacy."""

    def adjust_baq(self, dT: float) -> float:
        return LINEAR_SLOPE * float(dT) + LINEAR_INTERCEPT

    def adjust_3c(self, dT_baq: float, dT_car: float, dT_sm: float) -> float:
        mean_dt = (float(dT_baq) + float(dT_car) + float(dT_sm)) / 3.0
        return self.adjust_baq(mean_dt)


def _read_artifact(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("version") != "v1":
        raise ValueError(f"Versión de artefacto no soportada: {path}")
    return artifact


def _validate_piecewise(artifact: dict[str, Any], path: Path) -> None:
    knots = artifact.get("dT_knots")
    values = artifact.get("dD_values")
    if not isinstance(knots, list) or not isinstance(values, list):
        raise ValueError(f"Curva incompleta: {path}")
    if len(knots) < 2 or len(knots) != len(values):
        raise ValueError(f"Dimensiones de curva inválidas: {path}")
    if any(not math.isfinite(float(value)) for value in (*knots, *values)):
        raise ValueError(f"Curva no finita: {path}")
    if any(float(a) >= float(b) for a, b in zip(knots, knots[1:])):
        raise ValueError(f"dT_knots debe ser estrictamente creciente: {path}")
    if any(float(a) > float(b) for a, b in zip(values, values[1:])):
        raise ValueError(f"dD_values debe ser monótono: {path}")


class CalibratedCurveAdjuster:
    """Evalúa artefactos JSON congelados con clipping operativo."""

    def __init__(self, baq_artifact: str | Path, three_c_artifact: str | Path):
        self.baq_path = Path(baq_artifact)
        self.three_c_path = Path(three_c_artifact)
        self.baq = _read_artifact(self.baq_path)
        self.three_c = _read_artifact(self.three_c_path)
        if self.baq.get("type") != "piecewise_linear":
            raise ValueError("rn_baq_v1 debe ser piecewise_linear")
        _validate_piecewise(self.baq, self.baq_path)
        if self.three_c.get("type") == "piecewise_linear":
            _validate_piecewise(self.three_c, self.three_c_path)
        elif self.three_c.get("type") == "linear_3c":
            coefficients = (
                self.three_c.get("coef_baq"),
                self.three_c.get("coef_car"),
                self.three_c.get("coef_sm"),
            )
            if any(value is None or float(value) < 0 for value in coefficients):
                raise ValueError("Los coeficientes 3C deben ser no negativos")
        else:
            raise ValueError("Tipo de artefacto 3C no soportado")

    @staticmethod
    def _interpolate(artifact: dict[str, Any], dT: float) -> float:
        x = _clip(dT, DT_MIN, DT_MAX)
        knots = [float(value) for value in artifact["dT_knots"]]
        values = [float(value) for value in artifact["dD_values"]]
        if x <= knots[0]:
            return _clip(values[0], DD_MIN, DD_MAX)
        if x >= knots[-1]:
            return _clip(values[-1], DD_MIN, DD_MAX)
        right = bisect_right(knots, x)
        left = right - 1
        fraction = (x - knots[left]) / (knots[right] - knots[left])
        result = values[left] + fraction * (values[right] - values[left])
        return _clip(result, DD_MIN, DD_MAX)

    def adjust_baq(self, dT: float) -> float:
        return self._interpolate(self.baq, dT)

    def adjust_3c(self, dT_baq: float, dT_car: float, dT_sm: float) -> float:
        if self.three_c["type"] == "piecewise_linear":
            mean_dt = (
                _clip(dT_baq, DT_MIN, DT_MAX)
                + _clip(dT_car, DT_MIN, DT_MAX)
                + _clip(dT_sm, DT_MIN, DT_MAX)
            ) / 3.0
            return self._interpolate(self.three_c, mean_dt)

        result = (
            float(self.three_c.get("intercept", LINEAR_INTERCEPT))
            + float(self.three_c["coef_baq"]) * _clip(dT_baq, DT_MIN, DT_MAX)
            + float(self.three_c["coef_car"]) * _clip(dT_car, DT_MIN, DT_MAX)
            + float(self.three_c["coef_sm"]) * _clip(dT_sm, DT_MIN, DT_MAX)
        )
        return _clip(result, DD_MIN, DD_MAX)


class FallbackChainAdjuster:
    """Usa el primario solo cuando devuelve un ajuste operativo válido."""

    def __init__(
        self,
        primary: ClimateAdjuster,
        fallback: ClimateAdjuster | None = None,
    ):
        self.primary = primary
        self.fallback = fallback or LinearFallbackAdjuster()

    @staticmethod
    def _valid(value: float) -> bool:
        return math.isfinite(value) and abs(value) <= DD_MAX

    def adjust_baq(self, dT: float) -> float:
        try:
            result = float(self.primary.adjust_baq(dT))
            if self._valid(result):
                return result
        except (ArithmeticError, TypeError, ValueError):
            pass
        return self.fallback.adjust_baq(dT)

    def adjust_3c(self, dT_baq: float, dT_car: float, dT_sm: float) -> float:
        try:
            result = float(self.primary.adjust_3c(dT_baq, dT_car, dT_sm))
            if self._valid(result):
                return result
        except (ArithmeticError, TypeError, ValueError):
            pass
        return self.fallback.adjust_3c(dT_baq, dT_car, dT_sm)


def get_default_adjuster() -> ClimateAdjuster:
    """Selecciona el corrector por entorno; calibrado es el modo preferido."""

    mode = os.getenv("MPM_CLIMATE_ADJUSTER", "calibrated").strip().lower()
    if mode == "linear":
        return LinearFallbackAdjuster()
    if mode != "calibrated":
        raise ValueError("MPM_CLIMATE_ADJUSTER debe ser 'linear' o 'calibrated'")

    baq_path = ARTIFACTS_DIR / "rn_baq_v1.json"
    three_c_path = ARTIFACTS_DIR / "rn_3c_v1.json"
    if not baq_path.is_file() or not three_c_path.is_file():
        return LinearFallbackAdjuster()
    try:
        primary = CalibratedCurveAdjuster(baq_path, three_c_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return LinearFallbackAdjuster()
    return FallbackChainAdjuster(primary, LinearFallbackAdjuster())
