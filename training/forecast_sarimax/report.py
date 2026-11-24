"""Reporte antes/después del id=3.

1. Dev walk-forward (orígenes 2017-12..2025-12 mensuales y diciembres, h=24, reales <= 2025-12): elegido vs
   original reevaluado (mismo orden+macro constante, reajustado por origen) vs baselines.
2. ``--final``: test final 2026 (se corre UNA vez, con el modelo ya congelado): origen 2025-12, h=1..8
   (ene-ago 2026, pre-quiebre), vía ``predict_monthly`` (re-estima con datos <= 2025-12: sin fuga) y orígenes
   móviles 2025-12..2026-07 (h hasta ago-2026).

Uso: ``python -m training.forecast_sarimax.report [--final]`` → docs/work/03_sarimax/report_*.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from api.services.forecast_sarimax import ModelSpec, load_artifact, predict_monthly
from training.common.walkforward import BASELINES, walk_forward
from training.forecast_sarimax.train_final import SELECTION_FILE
from training.forecast_sarimax.wf import (
    DEV_ORIGINS,
    HORIZON,
    OUT,
    EnsembleFn,
    OriginalFn,
    SpecFn,
    baseline_runs,
    default_exclusions,
    load_data,
    make_origins,
    parallel_walk_forward,
    summarize,
)

COLS = ["modelo", "n", "mape", "sesgo", "max_ape", "mape_h1_12", "mape_h13_24", "n_anual", "mape_anual", "sesgo_anual"]


def chosen_fn():
    sel = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    specs = [ModelSpec.from_dict(d) for d in sel["components"]]
    return sel["nombre"], (SpecFn(specs[0]) if len(specs) == 1 else EnsembleFn(specs))


def dev():
    train, act_dev, _, mb = load_data()
    name, fn = chosen_fn()
    out = []
    for tag, step in (("mensual", 1), ("diciembre", 12)):
        origins = make_origins(*DEV_ORIGINS, step_months=step)
        runs = baseline_runs(train, origins, HORIZON, act_dev)
        runs.append(parallel_walk_forward(OriginalFn(True), train, origins, HORIZON, act_dev, "original_516_macro"))
        runs.append(parallel_walk_forward(fn, train, origins, HORIZON, act_dev, name))
        res = pd.concat(runs, ignore_index=True)
        s = summarize(res, default_exclusions(mb)).sort_values("mape")
        s.insert(0, "origenes", tag)
        res.to_csv(OUT / f"report_dev_{tag}_detalle.csv.gz", index=False)
        out.append(s)
    summ = pd.concat(out, ignore_index=True)
    summ.to_csv(OUT / "report_dev_resumen.csv", index=False)
    print(summ[["origenes"] + COLS].round(2).to_string(index=False))


def _metrics(pred: pd.Series, real: pd.Series) -> dict:
    e = (pred - real) / real * 100
    return {"n": int(e.size), "mape": e.abs().mean(), "sesgo": e.mean(), "max_ape": e.abs().max(),
            "err_total": (pred.sum() - real.sum()) / real.sum() * 100}


def final():
    train, _, act_all, mb = load_data()
    name, fn = chosen_fn()
    real = act_all.loc["2026-01":"2026-08"]
    art = load_artifact()
    rows, detail = [], []
    # (a) módulo de inferencia tal como lo servirá la API (start=2026-01, re-estima con <= 2025-12)
    p_mod = predict_monthly(train, "2026-01-01", "2026-08-31", artifact=art).set_index("fecha")["prediccion"]
    # (b) original reevaluado (reajustado en origen 2025-12) y baselines
    hist = train.loc[:"2025-12"].copy()
    hist.attrs = {"origen": pd.Timestamp("2025-12-01")}
    p_orig = pd.Series(np.asarray(OriginalFn(True)(hist, 8)), index=real.index)
    preds = {name + " (predict_monthly)": p_mod, "original_516_macro (reajustado 2025-12)": p_orig}
    for b, bfn in BASELINES.items():
        preds[b] = pd.Series(np.asarray(bfn(hist, 8), dtype=float), index=real.index)
    # (c) artefacto original tal cual lo servía la API (crudo, sin tope/override) — del diagnóstico
    base = OUT.parent / "00_entorno" / "baseline_2026_2027.csv"
    if base.exists():
        b = pd.read_csv(base)
        if {"mes", "SARIMAX_crudo"} <= set(b.columns):
            s = b.set_index(pd.to_datetime(b["mes"]))["SARIMAX_crudo"]
            preds["original_artefacto_api (crudo)"] = s.reindex(real.index).astype(float)
    for k, p in preds.items():
        if p.isna().any():
            continue
        rows.append({"modelo": k, **_metrics(p, real)})
        detail.append(pd.DataFrame({"modelo": k, "objetivo": real.index, "real": real.values, "pred": p.values,
                                    "err_pct": (p.values - real.values) / real.values * 100}))
    t = pd.DataFrame(rows).sort_values("mape")
    pd.concat(detail).to_csv(OUT / "report_test2026_detalle.csv", index=False)
    t.to_csv(OUT / "report_test2026_origen_2025_12.csv", index=False)
    print("Test final 2026 (origen 2025-12, ene-ago 2026):")
    print(t.round(2).to_string(index=False))
    # (d) orígenes móviles 2025-12..2026-07 (targets <= 2026-08)
    origins = make_origins("2025-12", "2026-07")
    act = act_all.loc[:"2026-08"]
    runs = [walk_forward(fn, train, origins, 8, act, name),
            walk_forward(OriginalFn(True), train, origins, 8, act, "original_516_macro")]
    runs += [walk_forward(bfn, train, origins, 8, act, b) for b, bfn in BASELINES.items()]
    res = pd.concat(runs, ignore_index=True)
    s = summarize(res, default_exclusions(mb), buckets=((1, 3), (4, 8)))
    s.to_csv(OUT / "report_test2026_movil.csv", index=False)
    print("Orígenes móviles 2025-12..2026-07:")
    print(s[["modelo", "n", "mape", "sesgo", "max_ape", "mape_h1_3", "mape_h4_8"]].round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    final() if a.final else dev()
