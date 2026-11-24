"""Evaluación final (se corre UNA vez, con la configuración ya congelada en ``selected_config.json``).

1. Walk-forward comparable con el piso de 01_datos: orígenes mensuales 2017-12..2025-12, h = 24, mismas
   exclusiones (COVID y >= quiebre). Incluye objetivos ene-ago 2026 (por eso no se usa para elegir nada).
2. Test final intocable: origen 2025-12 (modelo entrenado solo con datos <= 2025-12), ene-ago 2026 crudo.
3. Hindcast 2025 desde origen 2024-12 (comparable con DIAGNOSTICO §3).

Uso::

    python -m training.forecast_rfxgb.final_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from api.services import forecast_rfxgb as F
from training.common.data import load_monthly_demand, target_series
from training.common.walkforward import (
    BASELINES,
    default_exclusions,
    make_origins,
    seasonal_naive_drift,
    summarize,
    walk_forward,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "work" / "02_rfxgb"


def fit_predict_factory(cfg: dict, model_id: int):
    def fn(train: pd.Series, horizon: int) -> pd.Series:
        art = F.fit_model(train, model_id, cfg)
        return F.predict_from_artifact(art, train, horizon)

    return fn


def ensemble_factory(cfg_rf: dict, cfg_xgb: dict):
    f1, f2 = fit_predict_factory(cfg_rf, 1), fit_predict_factory(cfg_xgb, 2)

    def fn(train: pd.Series, horizon: int) -> pd.Series:
        return (f1(train, horizon) + f2(train, horizon)) / 2

    return fn


def main() -> None:
    sel = json.loads((HERE / "selected_config.json").read_text(encoding="utf-8"))
    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    train = target_series(monthly, market_break=mb)
    actuals = target_series(monthly, include_imputed=False)
    exc = default_exclusions(mb)
    models = {
        "rf_v2": fit_predict_factory(sel["rf"], 1),
        "xgb_v2": fit_predict_factory(sel["xgb"], 2),
        "media_rf_xgb_v2": ensemble_factory(sel["rf"], sel["xgb"]),
        **BASELINES,
    }

    # 1) walk-forward completo
    origins = make_origins("2017-12", "2025-12", 1)
    res = pd.concat([walk_forward(fn, train, origins, 24, actuals, n) for n, fn in models.items()], ignore_index=True)
    summ = summarize(res, exc)
    summ.to_csv(OUT / "walkforward_final_resumen.csv", index=False)
    res.to_csv(OUT / "walkforward_final_detalle.csv.gz", index=False)
    # Subconjunto de selección (orígenes <= 2023-12) y "validación" (objetivos 2024-2025)
    sub_sel = summarize(res[res["origen"] <= "2023-12-01"], exc)
    sub_val = summarize(res[(res["objetivo"] >= "2024-01-01") & (res["objetivo"] <= "2025-12-01")], exc)
    sub_sel.to_csv(OUT / "walkforward_final_resumen_origenes_hasta_2023.csv", index=False)
    sub_val.to_csv(OUT / "walkforward_final_resumen_objetivos_2024_2025.csv", index=False)

    # 2) test 2026 (origen 2025-12) y 3) hindcast 2025 (origen 2024-12)
    tests = []
    for origin, h, tag in (("2025-12-01", 8, "test_2026_ene_ago"), ("2024-12-01", 12, "hindcast_2025")):
        o = pd.Timestamp(origin)
        tr = train[train.index <= o].copy()
        tr.attrs = {"origen": o}
        targets = pd.date_range(o + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        real = actuals.reindex(targets)
        for name, fn in models.items():
            p = fn(tr, h)
            p = pd.Series(np.asarray(p, dtype=float), index=targets) if not isinstance(p, pd.Series) else p.reindex(targets)
            e = (p - real) / real * 100
            tests.append({"caso": tag, "modelo": name, "mape": e.abs().mean(), "sesgo": e.mean(),
                          "max_ape": e.abs().max(), "err_total": (p.sum() - real.sum()) / real.sum() * 100,
                          **{f"err_{t:%Y-%m}": v for t, v in e.items()}})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(OUT / "test_final.csv", index=False)
    cols = ["modelo", "n_origenes", "mape", "sesgo", "max_ape", "mape_h1_12", "mape_h13_24", "mape_anual"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("== walk-forward 2017-12..2025-12 ==")
        print(summ[cols].round(3).to_string(index=False))
        print("== orígenes <= 2023-12 ==")
        print(sub_sel[cols].round(3).to_string(index=False))
        print("== objetivos 2024-2025 ==")
        print(sub_val[cols].round(3).to_string(index=False))
        print("== test final / hindcast ==")
        print(tdf.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
