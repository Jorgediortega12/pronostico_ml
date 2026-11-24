"""Cache de predicciones fuera de muestra por origen mensual para ids 1 (RF), 2 (XGB), 3 (SARIMAX), 4 (Clima v3)
y naive+drift: ``h = 1..24`` desde cada origen ``ORIGINS`` con el modelo REAJUSTADO solo con datos ``<= origen``
(misma funcion de inferencia que la API). Permite evaluar la correccion de nivel de forma vectorizada.

Uso::

    python -m training.level_adapt.build_cache [--n-jobs 12]

Salida: ``docs/work/11_nivel_adaptativo/resultados/pred_cache.csv.gz`` (modelo, origen, objetivo, h, pred).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from api.services import forecast_rfxgb as F
from api.services import forecast_sarimax as S
from training.common.data import load_monthly_demand, target_series
from training.common.walkforward import BASELINES, make_origins
from training.forecast_climate import clima_v3 as v3
from training.forecast_climate import walkforward_clima_v3 as wf3

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "docs" / "work" / "11_nivel_adaptativo" / "resultados"
HORIZON = 24
FIRST_ORIGIN, LAST_ORIGIN = "2016-01", "2026-07"


def _rows(name: str, origin: pd.Timestamp, pred: pd.Series) -> pd.DataFrame:
    p = pd.Series(np.asarray(pred, dtype=float), index=pd.DatetimeIndex(pred.index))
    return pd.DataFrame({"modelo": name, "origen": origin, "objetivo": p.index, "h": np.arange(1, len(p) + 1), "pred": p.to_numpy()})


def one_origin(origin: pd.Timestamp, train_full: pd.Series, sel: dict, sar_art, clima_full: pd.DataFrame) -> pd.DataFrame:
    tr = train_full[train_full.index <= origin].copy()
    tr.attrs = {"origen": origin}
    out = []
    for mid, key, name in ((1, "rf", "rf"), (2, "xgb", "xgb")):
        cfg = {**sel[key], "params": {**sel[key]["params"], "n_jobs": 1}, "exclude_targets": [tuple(x) for x in sel[key]["exclude_targets"]]}
        art = F.fit_model(tr, mid, cfg)
        out.append(_rows(name, origin, F.predict_from_artifact(art, tr, HORIZON)))
    spec = sar_art.specs()[0]
    out.append(_rows("sarimax", origin, S.fit_predict(tr, HORIZON, spec)))
    out.append(_rows("naive_drift", origin, pd.Series(BASELINES["naive_estacional_drift"](tr, HORIZON),
                                                      index=pd.date_range(origin + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS"))))
    h4 = wf3.hist_upto(clima_full, origin)
    out.append(_rows("clima", origin, v3.predict_v3(h4, HORIZON, wf3.CHOSEN)))
    return pd.concat(out, ignore_index=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args(argv)
    from joblib import Parallel, delayed

    sel = json.loads((REPO / "training" / "forecast_rfxgb" / "selected_config.json").read_text(encoding="utf-8"))
    monthly = load_monthly_demand(impute_incomplete=True)
    train = target_series(monthly, market_break=monthly.attrs["market_break"])
    sar_art = S.load_artifact()
    clima_full = wf3.load_hist()
    origins = make_origins(FIRST_ORIGIN, LAST_ORIGIN, 1)
    t0 = time.time()
    parts = Parallel(n_jobs=args.n_jobs)(delayed(one_origin)(o, train, sel, sar_art, clima_full) for o in origins)
    df = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pred_cache.csv.gz", index=False)
    print(f"{len(origins)} origenes, {len(df)} filas, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
