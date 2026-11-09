"""Backtest walk-forward rápido del motor MPM legacy con datos reales."""

from __future__ import annotations

import calendar
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest
from holidays_co import get_colombia_holidays_by_year

from api.services.mpm_legacy import predict_month


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[4]
    / "mpm_gmr"
    / "Combined_Demanda_Clima.csv"
)
CUT_DAY = 15
MONTHS_IN_CI = 3
# Señal climática mínima (°C) para exigir que el adjuster mueva el total.
SIGNIFICANT_DT = 0.3
# Separación relativa mínima entre el total base (id=1) y el corregido.
RELATIVE_EPSILON = 1e-4


@dataclass(frozen=True)
class BacktestMetrics:
    mae_by_method: dict[int, float]
    monthly_relative_errors: dict[int, list[float]]
    totals_by_method: dict[int, list[float]]
    mean_abs_dt: list[float]
    months: list[str]


def _dataset_path() -> Path:
    configured = os.getenv("MPM_BACKTEST_CSV")
    return Path(configured) if configured else DEFAULT_DATASET


def _load_rows(path: Path) -> dict[date, tuple[float, float, float, float]]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source, delimiter=";"):
            day = date.fromisoformat(row["Fecha"])
            rows[day] = (
                float(row["P"]),
                float(row["t_CTG"]),
                float(row["t_MON"]),
                float(row["t_VDP"]),
            )
    return rows


def _complete_months(
    rows: dict[date, tuple[float, float, float, float]], count: int
) -> list[tuple[int, int]]:
    """Últimos `count` meses completos con temperatura diaria variable.

    La cola del dataset trae climatología mensual (un único valor de t_CTG
    repetido todo el mes). Con dT idénticamente cero el corrector climático es
    un no-op y el backtest no probaría nada, así que esos meses se descartan.
    """

    grouped: dict[tuple[int, int], set[int]] = defaultdict(set)
    temperatures: dict[tuple[int, int], set[float]] = defaultdict(set)
    for day, values in rows.items():
        grouped[(day.year, day.month)].add(day.day)
        temperatures[(day.year, day.month)].add(values[1])
    complete = [
        key
        for key, observed_days in grouped.items()
        if len(observed_days) == calendar.monthrange(*key)[1]
        and len(temperatures[key]) > 1
    ]
    return sorted(complete)[-count:]


def _holiday_dates(years: set[int]) -> set[date]:
    result = set()
    for year in years:
        for holiday in get_colombia_holidays_by_year(year):
            value = holiday.date
            result.add(value.date() if hasattr(value, "date") else value)
    return result


def run_backtest(path: Path, month_count: int = MONTHS_IN_CI) -> BacktestMetrics:
    rows = _load_rows(path)
    selected_months = _complete_months(rows, month_count)
    if len(selected_months) < month_count:
        raise ValueError("No hay suficientes meses completos para el backtest")

    absolute_errors: dict[int, list[float]] = defaultdict(list)
    relative_errors: dict[int, list[float]] = defaultdict(list)
    totals: dict[int, list[float]] = defaultdict(list)
    mean_abs_dt: list[float] = []

    for year, month in selected_months:
        month_days = calendar.monthrange(year, month)[1]
        dates = [date(year, month, day) for day in range(1, month_days + 1)]
        observed = {day: rows[day][0] for day in dates[:CUT_DAY]}
        first_day = dates[0]
        previous = [
            (day, values[0], values[1])
            for day, values in sorted(rows.items())
            if day < first_day
        ][-30:]
        temps = {
            "bolivar": [rows[day][1] for day in dates],
            "cordoba_sucre": [rows[day][2] for day in dates],
            "cesar": [rows[day][3] for day in dates],
        }
        holidays = _holiday_dates({year, first_day.year})
        actual_future = [rows[day][0] for day in dates[CUT_DAY:]]
        actual_total = sum(rows[day][0] for day in dates)

        # Proxy de la señal climática del mes: cuánto se apartan las
        # temperaturas Baq del futuro respecto al tramo ya observado.
        observed_mean_temp = sum(rows[day][1] for day in dates[:CUT_DAY]) / CUT_DAY
        future_days = dates[CUT_DAY:]
        mean_abs_dt.append(
            sum(abs(rows[day][1] - observed_mean_temp) for day in future_days)
            / len(future_days)
        )

        for method_id in (1, 2, 3, 4):
            result = predict_month(
                dates=dates,
                demand_by_date=observed,
                previous_days_demand=previous,
                holidays=holidays,
                temps=temps,
                scenario=1,
                method_id=method_id,
            )
            predicted_future = result.daily_predictions
            absolute_errors[method_id].extend(
                abs(predicted - actual)
                for predicted, actual in zip(predicted_future, actual_future)
            )
            predicted_total = sum(observed.values()) + sum(predicted_future)
            totals[method_id].append(predicted_total)
            relative_errors[method_id].append(
                abs(predicted_total - actual_total) / actual_total
            )

    return BacktestMetrics(
        mae_by_method={
            method_id: sum(errors) / len(errors)
            for method_id, errors in absolute_errors.items()
        },
        monthly_relative_errors=dict(relative_errors),
        totals_by_method=dict(totals),
        mean_abs_dt=mean_abs_dt,
        months=[f"{year:04d}-{month:02d}" for year, month in selected_months],
    )


@pytest.fixture(scope="module")
def metrics() -> BacktestMetrics:
    path = _dataset_path()
    if not path.exists():
        pytest.skip(
            "Dataset no disponible; configure MPM_BACKTEST_CSV para ejecutarlo"
        )
    return run_backtest(path)


def test_walk_forward_climate_methods_do_not_degrade(metrics: BacktestMetrics):
    assert math.isfinite(metrics.mae_by_method[1])
    assert metrics.mae_by_method[2] <= metrics.mae_by_method[1] * 1.10
    assert metrics.mae_by_method[3] <= metrics.mae_by_method[1] * 1.20
    assert all(
        total > 0 and math.isfinite(total)
        for method_totals in metrics.totals_by_method.values()
        for total in method_totals
    )
    assert all(
        math.isfinite(error)
        for method_errors in metrics.monthly_relative_errors.values()
        for error in method_errors
    )


def test_climate_adjuster_moves_totals_when_temperature_differs(
    metrics: BacktestMetrics,
):
    """Un corrector que devolviera siempre 0 pasaría los límites de MAE."""

    baseline = metrics.totals_by_method[1]
    signal_months = [
        idx
        for idx, mean_dt in enumerate(metrics.mean_abs_dt)
        if mean_dt >= SIGNIFICANT_DT
    ]
    assert signal_months, (
        "Ningún mes del backtest tiene señal climática; el dataset no permite "
        f"validar el corrector (mean|dT| = {metrics.mean_abs_dt})"
    )

    for method_id in (2, 3):
        adjusted = metrics.totals_by_method[method_id]
        divergences = [
            abs(adjusted[idx] - baseline[idx]) / baseline[idx]
            for idx in signal_months
        ]
        assert max(divergences) > RELATIVE_EPSILON, (
            f"id={method_id} no se separa de id=1 en ningún mes con clima: "
            f"{divergences}"
        )
