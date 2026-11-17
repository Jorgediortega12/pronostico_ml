"""Tests del puerto de ajuste climático residual."""

import json
import math
from pathlib import Path

import pytest

from api.services.mpm_legacy.climate_adjust import (
    LINEAR_INTERCEPT,
    LINEAR_SLOPE,
    apply_linear_baq,
    find_climate_reference,
    linear_delta_demand,
    stub_rn_3c,
    stub_rn_baq,
)
from api.services.mpm_legacy.climate_adjuster import (
    ARTIFACTS_DIR,
    CalibratedCurveAdjuster,
    FallbackChainAdjuster,
    LinearFallbackAdjuster,
)


def test_linear_formula_matches_matlab():
    # dD = 1.099*dT - 0.01963
    assert LINEAR_SLOPE == 1.099
    assert LINEAR_INTERCEPT == -0.01963
    dt = 2.0
    assert math.isclose(linear_delta_demand(dt), 1.099 * 2.0 - 0.01963)


def test_apply_linear_scales_to_wh():
    base = 30_000_000.0
    dt = 1.0
    expected = base + (1.099 * 1.0 - 0.01963) * 1e6
    assert math.isclose(apply_linear_baq(base, dt), expected)


def test_zero_delta_is_near_base():
    base = 25_000_000.0
    # dT=0 → dD = -0.01963 GW → base casi sin cambios (pequeño intercepto).
    assert math.isclose(apply_linear_baq(base, 0.0), base + LINEAR_INTERCEPT * 1e6)


def test_stub_rn_baq_equals_linear(monkeypatch):
    monkeypatch.setenv("MPM_CLIMATE_ADJUSTER", "linear")
    base = 28_000_000.0
    assert stub_rn_baq(base, 1.5) == apply_linear_baq(base, 1.5)


def test_stub_rn_3c_uses_average_delta(monkeypatch):
    monkeypatch.setenv("MPM_CLIMATE_ADJUSTER", "linear")
    base = 28_000_000.0
    deltas = [1.0, 2.0, 3.0]  # promedio 2.0
    assert math.isclose(stub_rn_3c(base, deltas), apply_linear_baq(base, 2.0))


def test_stub_rn_3c_empty_returns_base():
    assert stub_rn_3c(28_000_000.0, []) == 28_000_000.0


def _calibrated():
    return CalibratedCurveAdjuster(
        ARTIFACTS_DIR / "rn_baq_v1.json",
        ARTIFACTS_DIR / "rn_3c_v1.json",
    )


def test_calibrated_baq_is_monotonic_on_grid():
    adjuster = _calibrated()
    values = [adjuster.adjust_baq(d_t / 10.0) for d_t in range(-70, 71)]
    assert all(left <= right for left, right in zip(values, values[1:]))


def test_calibrated_3c_is_monotonic_per_input():
    adjuster = _calibrated()
    grid = [value / 2.0 for value in range(-10, 11)]
    for position in range(3):
        values = []
        for d_t in grid:
            deltas = [0.0, 0.0, 0.0]
            deltas[position] = d_t
            values.append(adjuster.adjust_3c(*deltas))
        assert all(left <= right for left, right in zip(values, values[1:]))


def test_calibrated_zero_delta_is_small():
    adjuster = _calibrated()
    assert abs(adjuster.adjust_baq(0.0)) < 0.1
    assert abs(adjuster.adjust_3c(0.0, 0.0, 0.0)) < 0.1


def test_calibrated_clips_inputs_and_outputs():
    adjuster = _calibrated()
    assert adjuster.adjust_baq(-100.0) == adjuster.adjust_baq(-5.0)
    assert adjuster.adjust_baq(100.0) == adjuster.adjust_baq(5.0)
    assert -3.0 <= adjuster.adjust_baq(100.0) <= 3.0
    assert adjuster.adjust_3c(100.0, 100.0, 100.0) <= 3.0


def test_fallback_nan_uses_linear():
    class NanAdjuster:
        def adjust_baq(self, dT):
            return float("nan")

        def adjust_3c(self, dT_baq, dT_car, dT_sm):
            return float("nan")

    adjuster = FallbackChainAdjuster(NanAdjuster())
    linear = LinearFallbackAdjuster()
    assert adjuster.adjust_baq(1.0) == linear.adjust_baq(1.0)
    assert adjuster.adjust_3c(1.0, 2.0, 3.0) == linear.adjust_3c(1.0, 2.0, 3.0)


def test_linear_artifact_reproduces_legacy_formula(tmp_path: Path):
    knots = [-5.0, 0.0, 5.0]
    values = [LINEAR_SLOPE * value + LINEAR_INTERCEPT for value in knots]
    baq = {
        "version": "v1",
        "type": "piecewise_linear",
        "dT_knots": knots,
        "dD_values": values,
    }
    three_c = {
        "version": "v1",
        "type": "linear_3c",
        "intercept": LINEAR_INTERCEPT,
        "coef_baq": LINEAR_SLOPE / 3.0,
        "coef_car": LINEAR_SLOPE / 3.0,
        "coef_sm": LINEAR_SLOPE / 3.0,
    }
    baq_path = tmp_path / "baq.json"
    three_c_path = tmp_path / "3c.json"
    baq_path.write_text(json.dumps(baq), encoding="utf-8")
    three_c_path.write_text(json.dumps(three_c), encoding="utf-8")
    adjuster = CalibratedCurveAdjuster(baq_path, three_c_path)
    assert adjuster.adjust_baq(1.5) == pytest.approx(linear_delta_demand(1.5))
    assert adjuster.adjust_3c(1.0, 2.0, 3.0) == pytest.approx(
        linear_delta_demand(2.0)
    )


# --------------------------------------------------------------------------
# find_climate_reference: ismember(RefData(:,[2 7]), ProData(i,[2 7])) y el
# fallback ordinario con pos = f - ProData(i,1) (f es 1-based en MATLAB).
# --------------------------------------------------------------------------

# RefData de 4 filas: Mar(3), Mié(4), Jue(5), Vie(6), todas ordinarias.
REF_WEEKDAYS = [3, 4, 5, 6]
REF_TIPOS = [0, 0, 0, 0]
REF_TEMPS = [30.0, 30.5, 31.0, 31.5]


def _find(weekday, tipo, temp):
    return find_climate_reference(
        target_weekday=weekday,
        target_tipo=tipo,
        target_temp_baq=temp,
        ref_weekdays=REF_WEEKDAYS,
        ref_tipos=REF_TIPOS,
        ref_temps_baq=REF_TEMPS,
    )


def test_exact_tipo_and_temp_match_skips_adjustment():
    assert _find(5, 0, 31.0) == (2, True)


def test_same_weekday_fallback_takes_last_row():
    assert _find(4, 0, 28.0) == (1, False)


def test_non_ordinary_without_exact_match_has_no_reference():
    assert _find(7, 1, 28.0) is None


def test_weekday_distance_fallback_is_one_based():
    # target weekday 2 (lunes) no está en RefData.
    # MATLAB: pos = f - ProData(i,1) con f = 1..4 → [-1, 0, 1, 2].
    # La fila con pos == -1 es f = 1, es decir el índice 0 en Python.
    # Con índices 0-based el cálculo daría [-2, -1, 0, 1] → índice 1 (mal).
    assert _find(2, 0, 28.0) == (0, False)


def test_weekday_distance_fallback_prefers_plus_one():
    # target weekday 1: pos = [0, 1, 2, 3]; no hay -1, se toma pos == 1 → f = 2.
    assert _find(1, 0, 28.0) == (1, False)


def test_weekday_distance_fallback_uses_min_abs_when_no_neighbour():
    # target weekday 7: pos = [-6, -5, -4, -3]; sin -1 ni 1 → min|pos| → f = 4.
    assert _find(7, 0, 28.0) == (3, False)
