"""Selección de variantes/hiperparámetros de RF (id 1) y XGB (id 2) por walk-forward.

Protocolo (ver docs/work/02_rfxgb/PLAN.md):
* Solo orígenes ``<= 2023-12`` (por defecto 2017-12..2023-12 cada 3 meses) y h = 1..24 → todos los
  objetivos son ``<= 2025-12``. **2026 no se mira aquí.**
* En cada origen el modelo se entrena SOLO con la serie truncada al origen (ventana expansiva), con el
  mismo código de features que la inferencia (``api.services.forecast_rfxgb``).
* Métricas con el harness común (``training.common.walkforward.summarize``), excluyendo COVID.

Uso::

    python -m training.forecast_rfxgb.experiments [--step 3] [--out docs/work/02_rfxgb]
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from api.services import forecast_rfxgb as F
from training.common import config
from training.common.data import load_monthly_demand, target_series
from training.common.walkforward import (
    BASELINES,
    default_exclusions,
    make_origins,
    summarize,
    walk_forward,
)

REPO = Path(__file__).resolve().parents[2]
SELECTION_END = "2023-12"
HORIZON = 24

RF_GRID = [dict(min_samples_leaf=l, max_features=f) for l, f in itertools.product([5, 20], [0.5, 1.0])]
XGB_GRID = [dict(max_depth=d, min_child_weight=w) for d, w in itertools.product([2, 3], [5, 20])]


def candidate_configs() -> list[dict]:
    cfgs = []
    for offset in F.OFFSETS:
        for excl in (True, False):
            ex = (config.COVID_EXCLUDED,) if excl else ()
            for p in RF_GRID:
                cfgs.append(dict(model_id=1, offset=offset, params=p, exclude_targets=ex, h_train=36))
            for p in XGB_GRID:
                cfgs.append(dict(model_id=2, offset=offset, params=p, exclude_targets=ex, h_train=36))
    return cfgs


def cfg_name(c: dict) -> str:
    p = "_".join(f"{k}={v}" for k, v in c["params"].items())
    return f"{F.MODEL_NAMES[c['model_id']]}|{c['offset']}|covid_excl={bool(c['exclude_targets'])}|{p}"


def run_grid(train_series: pd.Series, actuals: pd.Series, origins, cfgs) -> pd.DataFrame:
    """Corre todas las configs en cada origen (diseño construido una vez por origen, solo con datos <= origen)."""
    rows = []
    for o in origins:
        o = pd.Timestamp(o)
        tr = train_series[train_series.index <= o]
        design = F.build_design(tr, 36)
        pred_d = F.build_design(tr, HORIZON, origins=[o])
        targets = pd.DatetimeIndex(pred_d["objetivo"])
        real = actuals.reindex(targets).to_numpy()
        X = pred_d[F.FEATURES].to_numpy(dtype=float)
        for off in F.OFFSETS:  # offset puro (sin ML) como referencia
            rows.append(pd.DataFrame({"modelo": f"offset_solo|{off}", "origen": o, "objetivo": targets,
                                      "h": pred_d["h"].to_numpy(), "real": real,
                                      "pred": np.exp(pred_d[f"off_{off}"].to_numpy())}))
        for c in cfgs:
            trr = F.training_rows(design, tuple(c["exclude_targets"]))
            est = F.make_estimator(c["model_id"], c["params"])
            est.fit(trr[F.FEATURES].to_numpy(dtype=float), F.residual_target(trr, c["offset"]).to_numpy(dtype=float))
            pred = np.exp(pred_d[f"off_{c['offset']}"].to_numpy() + est.predict(X))
            rows.append(pd.DataFrame({"modelo": cfg_name(c), "origen": o, "objetivo": targets,
                                      "h": pred_d["h"].to_numpy(), "real": real, "pred": pred}))
        print(f"origen {o:%Y-%m} listo", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out["err_pct"] = (out["pred"] - out["real"]) / out["real"] * 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--start", default="2017-12")
    ap.add_argument("--out", default=str(REPO / "docs" / "work" / "02_rfxgb"))
    args = ap.parse_args()

    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    train = target_series(monthly, market_break=mb)
    actuals = target_series(monthly, include_imputed=False)
    origins = make_origins(args.start, SELECTION_END, args.step)
    assert max(origins) + pd.DateOffset(months=HORIZON) <= pd.Timestamp("2025-12-01"), "la selección no debe ver 2026"
    exc = default_exclusions(mb)

    t0 = time.time()
    res = run_grid(train, actuals, origins, candidate_configs())
    base = pd.concat([walk_forward(fn, train, origins, HORIZON, actuals, n) for n, fn in BASELINES.items()])
    res = pd.concat([res, base], ignore_index=True)
    summ = summarize(res, exc).sort_values("mape")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "seleccion_resumen.csv", index=False)
    res.to_csv(out / "seleccion_detalle.csv.gz", index=False)
    cols = ["modelo", "n_origenes", "mape", "sesgo", "mape_h1_12", "mape_h13_24", "mape_anual"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 100):
        print(summ[cols].round(3).to_string(index=False))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
