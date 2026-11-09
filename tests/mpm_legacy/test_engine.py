"""Tests end-to-end del motor MPM legacy con fixture sintético."""

import json
from datetime import date
from pathlib import Path

import pytest

from api.services.mpm_legacy import LINEAR_INTERCEPT, LINEAR_SLOPE, predict_month
from api.services.mpm_legacy.climate_adjuster import (
    CalibratedCurveAdjuster,
    LinearFallbackAdjuster,
    ARTIFACTS_DIR,
)
from api.services.mpm_legacy import engine as engine_module
from api.services.mpm_legacy.engine import PredictMonthResult

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "mpm" / "synthetic_month.json"
)


def _load_fixture():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    dates = [date.fromisoformat(d) for d in data["dates"]]
    demand_by_date = {
        date.fromisoformat(k): float(v) for k, v in data["demand_by_date"].items()
    }
    holidays = {date.fromisoformat(d) for d in data["holidays"]}
    temps = {k: list(v) for k, v in data["temps"].items()}
    return data, dates, demand_by_date, holidays, temps


def _run(method_id: int) -> PredictMonthResult:
    data, dates, demand_by_date, holidays, temps = _load_fixture()
    return predict_month(
        dates=dates,
        demand_by_date=demand_by_date,
        previous_days_demand=None,
        holidays=holidays,
        temps=temps,
        scenario=data["scenario"],
        method_id=method_id,
    )


@pytest.mark.parametrize("method_id", [1, 2, 3, 4])
def test_predict_shapes(method_id):
    result = _run(method_id)
    n_total = 10
    n_observed = 5
    assert len(result.all_dates) == n_total
    assert len(result.day_types) == n_total
    assert len(result.series_for_method) == n_total
    assert len(result.historic_demands) == n_observed
    assert len(result.daily_predictions) == n_total - n_observed


def test_historic_matches_observed():
    result = _run(1)
    assert result.historic_demands == [
        30000000.0, 26000000.0, 24000000.0, 31000000.0, 31500000.0
    ]


def test_series_uses_real_for_observed_days():
    result = _run(1)
    # Los primeros 5 días de la serie deben ser la demanda real observada.
    assert result.series_for_method[:5] == result.historic_demands


def test_totals_are_positive_and_in_gw_range():
    result = _run(1)
    # 4 totales siempre presentes.
    assert len(result.assembled_totals_gw) == 4
    for total in result.assembled_totals_gw:
        assert total > 0.0
    # Suma de ~10 días de ~30M Wh → ~300 GW al dividir por 1e6.
    assert 100.0 < result.assembled_totals_gw[0] < 600.0


def test_method_id_selects_correct_total():
    r1 = _run(1)
    r4 = _run(4)
    assert r1.total_gw == r1.assembled_totals_gw[0]
    assert r4.total_gw == r4.assembled_totals_gw[3]


def test_method_4_uses_median_by_type():
    # id=4 debe basarse en Prom_Dia (medianas), no en el matching de días.
    result = _run(4)
    assert result.total_gw > 0.0
    # La serie en días faltantes debe ser una de las medianas por tipo.
    assert all(v > 0.0 for v in result.series_for_method)


def test_invalid_method_raises():
    _, dates, demand_by_date, holidays, temps = _load_fixture()
    with pytest.raises(ValueError):
        predict_month(
            dates=dates,
            demand_by_date=demand_by_date,
            holidays=holidays,
            temps=temps,
            scenario=1,
            method_id=99,
        )


def test_empty_dates_raises():
    with pytest.raises(ValueError):
        predict_month(
            dates=[],
            demand_by_date={},
            holidays=set(),
            temps={"bolivar": [], "cordoba_sucre": [], "cesar": []},
            scenario=1,
            method_id=1,
        )


# --------------------------------------------------------------------------
# GOLDEN: mes chico de 8 días calculado a mano.
#
#   Jun 01 Lun(2) 30 M | Jun 02 Mar(3) 31 M | Jun 03 Mié(4) 32 M
#   Jun 04 Jue(5) 33 M | Jun 05 Vie(6) 40 M | observados (prefijo N = 5)
#   Jun 06 Sáb(7)      | Jun 07 Dom(1)      | Jun 08 Lun(2)  a pronosticar
#
# Días previos (sin temperatura): Sáb 30/05 = 24 M, Dom 31/05 = 20 M.
# Temperatura Baq/Car = 30° salvo Jun 08 = 31°; Mon/SM = 30° siempre.
# --------------------------------------------------------------------------

GOLDEN_DATES = [date(2026, 6, day) for day in range(1, 9)]
GOLDEN_DEMAND = {
    date(2026, 6, 1): 30_000_000.0,
    date(2026, 6, 2): 31_000_000.0,
    date(2026, 6, 3): 32_000_000.0,
    date(2026, 6, 4): 33_000_000.0,
    date(2026, 6, 5): 40_000_000.0,
}
GOLDEN_PREVIOUS = [
    (date(2026, 5, 30), 24_000_000.0),
    (date(2026, 5, 31), 20_000_000.0),
]
GOLDEN_TEMPS = {
    "bolivar": [30.0] * 7 + [31.0],
    "cordoba_sucre": [30.0] * 8,
    "cesar": [30.0] * 8,
}

# case 1, lunes: candidato Jun 01 = 30 M y mean(DemandaActual) = 33.2 M,
# Desv = 3.2e6 > 7e5 → se escribe la media del mes actual en Jun 01 y Jun 08.
GOLDEN_MONDAY_FILL = 33_200_000.0
GOLDEN_F3 = [
    30_000_000.0,
    31_000_000.0,
    32_000_000.0,
    33_000_000.0,
    40_000_000.0,
    24_000_000.0,  # case 2: sin sábados del mes, DemandaActual = pool
    20_000_000.0,  # case 3: media del pool domingo/festivo
    GOLDEN_MONDAY_FILL,
]


def _run_golden(method_id: int) -> PredictMonthResult:
    return predict_month(
        dates=GOLDEN_DATES,
        demand_by_date=GOLDEN_DEMAND,
        previous_days_demand=GOLDEN_PREVIOUS,
        holidays=set(),
        temps=GOLDEN_TEMPS,
        scenario=1,
        method_id=method_id,
        adjuster=LinearFallbackAdjuster(),
    )


def test_previous_days_without_temperature_does_not_crash():
    # Antes reventaba con "cannot convert float NaN to integer" al hacer
    # round(nan) sobre la temperatura ausente de los días del mes anterior.
    result = _run_golden(1)
    assert len(result.daily_predictions) == 3


def test_previous_days_use_temperature_when_provided(monkeypatch):
    captured = {}
    original = engine_module.fill_base_demands

    def capture_historic(**kwargs):
        captured["historic"] = kwargs["historic"]
        return original(**kwargs)

    monkeypatch.setattr(engine_module, "fill_base_demands", capture_historic)
    predict_month(
        dates=GOLDEN_DATES,
        demand_by_date=GOLDEN_DEMAND,
        previous_days_demand=[
            (date(2026, 5, 30), 24_000_000.0, 28.5),
            (date(2026, 5, 31), 20_000_000.0, 29.5),
        ],
        holidays=set(),
        temps=GOLDEN_TEMPS,
        scenario=1,
        method_id=1,
    )

    assert [day.temp_baq for day in captured["historic"][:2]] == [28.5, 29.5]


def test_golden_f3_series_matches_hand_computation():
    result = _run_golden(1)
    assert result.series_for_method == GOLDEN_F3


def test_golden_pronostico_total_f3():
    # sum(f3) = 243.2 M Wh → 243.2 GW
    result = _run_golden(1)
    assert result.assembled_totals_gw[0] == pytest.approx(243.2)
    assert result.total_gw == pytest.approx(243.2)


def test_golden_pronostico_total_prom_dia():
    # medianas: ordinario 32 M, sábado 24 M, domingo 20 M
    # conteos: 6 ordinarios, 1 sábado, 1 domingo
    # (32*6 + 24 + 20) M / 1e6 = 236.0 GW
    result = _run_golden(4)
    assert result.assembled_totals_gw[3] == pytest.approx(236.0)
    assert result.total_gw == pytest.approx(236.0)


def test_golden_pronostico_total_climate_adjusted():
    # Solo Jun 08 recibe ajuste: es ordinario, no hay referencia con 31.0°
    # y sí un lunes de referencia (Jun 01) a 30.0° → dT_baq = 1.0.
    dd_baq = LINEAR_SLOPE * 1.0 + LINEAR_INTERCEPT
    # 3C: dT = (Baq 1.0, Car 1.0, SM 0.0) → promedio 2/3 en el stub lineal.
    dd_3c = LINEAR_SLOPE * (2.0 / 3.0) + LINEAR_INTERCEPT
    observed_and_weekend = sum(GOLDEN_F3[:7])

    result = _run_golden(1)
    expected_f5 = (observed_and_weekend + GOLDEN_MONDAY_FILL + dd_baq * 1e6) / 1e6
    expected_f6 = (observed_and_weekend + GOLDEN_MONDAY_FILL + dd_3c * 1e6) / 1e6
    assert result.assembled_totals_gw[1] == pytest.approx(expected_f5)
    assert result.assembled_totals_gw[2] == pytest.approx(expected_f6)
    assert result.assembled_totals_gw[1] == pytest.approx(244.27937)
    assert result.assembled_totals_gw[2] == pytest.approx(243.91303666666667)


def _run_golden_calibrated(last_temp: float) -> PredictMonthResult:
    temps = {key: list(values) for key, values in GOLDEN_TEMPS.items()}
    temps["bolivar"][-1] = last_temp
    adjuster = CalibratedCurveAdjuster(
        ARTIFACTS_DIR / "rn_baq_v1.json",
        ARTIFACTS_DIR / "rn_3c_v1.json",
    )
    return predict_month(
        dates=GOLDEN_DATES,
        demand_by_date=GOLDEN_DEMAND,
        previous_days_demand=GOLDEN_PREVIOUS,
        holidays=set(),
        temps=temps,
        scenario=1,
        method_id=2,
        adjuster=adjuster,
    )


def test_sensitivity_plus_one_degree_increases_f5():
    warm = _run_golden_calibrated(31.0)
    warmer = _run_golden_calibrated(32.0)
    assert warmer.total_gw > warm.total_gw


def test_calibrated_id2_can_differ_from_linear_f4():
    calibrated = _run_golden_calibrated(31.0)
    linear = _run_golden(2)
    assert calibrated.total_gw != pytest.approx(linear.total_gw)
