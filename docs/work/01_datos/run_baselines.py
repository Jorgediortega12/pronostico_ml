"""Baselines naive walk-forward sobre la BD real: el piso que los modelos deben superar.

Uso (desde la raíz del repo): ``python docs/work/01_datos/run_baselines.py``
Salidas en docs/work/01_datos/: baselines_walkforward.{csv,md}, baselines_walkforward_detalle.csv.gz,
demanda_mensual_limpia.csv, dedup_log.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from training.common.data import load_monthly_demand, target_series  # noqa: E402
from training.common.walkforward import (  # noqa: E402
    default_exclusions,
    evaluate_with_baselines,
    make_origins,
)

OUT = Path(__file__).resolve().parent
START, END, HORIZON = "2017-12", "2025-12", 24


def _fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".2f")


def main() -> None:
    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    log = monthly.attrs["dedup_log"]
    train = target_series(monthly)  # incluye imputados (válidos para train)
    actuals = target_series(monthly, include_imputed=False)  # solo meses observados completos
    exc = default_exclusions(mb)

    monthly.to_csv(OUT / "demanda_mensual_limpia.csv")
    log.to_csv(OUT / "dedup_log.csv", index=False)

    escenarios = {
        "origenes_mensuales": make_origins(START, END, 1),
        "origenes_diciembre": make_origins(START, END, 12),
    }
    resumenes, detalles = [], []
    for nombre, origins in escenarios.items():
        res, summ = evaluate_with_baselines(None, train, origins, HORIZON, actuals=actuals, exclusions=exc)
        summ.insert(0, "escenario", nombre)
        res.insert(0, "escenario", nombre)
        resumenes.append(summ)
        detalles.append(res)
    resumen = pd.concat(resumenes, ignore_index=True)
    detalle = pd.concat(detalles, ignore_index=True)
    resumen.to_csv(OUT / "baselines_walkforward.csv", index=False)
    detalle.to_csv(OUT / "baselines_walkforward_detalle.csv.gz", index=False)

    # Error anual por año objetivo (orígenes diciembre, h 1-12): piso año a año.
    from training.common.walkforward import annual_totals

    dic = detalle[detalle["escenario"] == "origenes_diciembre"].drop(columns="escenario")
    anual = annual_totals(dic, exc)
    anual = anual[anual["h_max"] <= 12].pivot_table(index="anio", columns="modelo", values="err_pct").reset_index()
    anual["anio"] = anual["anio"].astype(str)

    # Diagnóstico del quiebre (solo informativo: NO se usa para escalar).
    diag = ""
    if mb is not None:
        post = monthly.loc[monthly.index >= mb.first_post_month].dropna(subset=["demanda_kwh"])
        filas = []
        for m, v in post["demanda_kwh"].items():
            prev = monthly["demanda_kwh"].get(m - pd.DateOffset(years=1))
            filas.append({"mes": f"{m:%Y-%m}", "GWh": v / 1e6, "GWh_mismo_mes_anio_previo": prev / 1e6,
                          "ratio_yoy": v / prev})
        diag = _fmt(pd.DataFrame(filas)) if filas else "(sin meses post-quiebre completos)"

    cols = ["escenario", "modelo", "n_origenes", "n", "mape", "sesgo", "max_ape", "mape_h1_12", "sesgo_h1_12",
            "mape_h13_24", "sesgo_h13_24", "n_anual", "mape_anual", "sesgo_anual", "mape_anual_h1_12",
            "mape_anual_h13_24"]
    md = [
        "# Baselines walk-forward (piso a superar)",
        "",
        f"Generado por `docs/work/01_datos/run_baselines.py`. Orígenes {START}..{END}, horizonte {HORIZON} meses,",
        "expanding window. Train = serie limpia (dedupe 2020-Q4, 2021-09 imputado); real = solo meses observados",
        "completos (2021-09 NO cuenta en métricas). Excluidos de métricas: COVID 2020-03..2021-02 y meses",
        f"≥ {mb.first_post_month:%Y-%m} (quiebre de mercado, `demand_split_config` = {mb.fecha_limite} / "
        f"{mb.porcentaje:g} %)." if mb else "Sin configuración de quiebre de mercado.",
        "",
        "Errores en %: MAPE mensual, sesgo = media (pred−real)/real (negativo = subestima), `mape_anual` = error del",
        "total de cada año calendario completo dentro del horizonte (`_h1_12` = año completo en h≤12).",
        "",
        "## Resumen",
        "",
        _fmt(resumen[cols]),
        "",
        "## Error del total anual por año objetivo (orígenes diciembre, h 1–12, %)",
        "",
        _fmt(anual),
        "",
        "## Diagnóstico del quiebre de mercado (informativo, no se usa para escalar)",
        "",
        diag,
        "",
        f"Dedupe: {len(log)} fechas duplicadas resueltas (ver `dedup_log.csv`); "
        f"razón max/min entre candidatos {log['razon_max_min'].min():.1f}–{log['razon_max_min'].max():.1f}×."
        if len(log) else "Dedupe: sin fechas duplicadas.",
        "",
    ]
    (OUT / "baselines_walkforward.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
