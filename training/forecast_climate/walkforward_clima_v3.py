"""Walk-forward del modelo id=4 v3 (ver ``clima_v3.py``) con origenes MENSUALES, h=24.

Protocolo (``docs/work/09_clima_v3/PLAN.md``):

* TUNE: origenes 2017-12..2023-12, objetivos <= 2025-12 (unico uso permitido para elegir hiperparametros).
* VAL: origenes 2024-01..2025-11, objetivos <= 2025-12 (validacion; no se ajusta nada con ella).
* TEST: objetivos >= 2026-01 (2026-01..08, pre-quiebre de mercado); se mira UNA vez, al final.
* Comparadores sobre las mismas filas: naive estacional + drift (``training.common.walkforward``) y v2
  (``docs/work/04_clima/resultados/walkforward_mens.csv.gz``; v2 usa ``cutoff = anio_origen - 1`` como la API).

Uso::

    python -m training.forecast_climate.walkforward_clima_v3 --tune      # grilla en TUNE (+VAL informativo)
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from training.common.walkforward import (
    BASELINES,
    evaluable,
    excluded_mask,
    make_origins,
    walk_forward,
)
from training.forecast_climate import clima_v3 as v3

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve().parent
REPO_ROOT = _THIS.parents[1]
OUT_DIR = REPO_ROOT / "docs" / "work" / "09_clima_v3" / "resultados"
DATASET = _THIS / "artifacts" / "dataset_mensual.csv"
V2_WF = REPO_ROOT / "docs" / "work" / "04_clima" / "resultados" / "walkforward_mens.csv.gz"
V2_NAME = "despues_v2 (nys=2, sin parche)"

HORIZON = 24
TUNE_LAST_ORIGIN = pd.Timestamp("2023-12-01")
TUNE_LAST_TARGET = pd.Timestamp("2025-12-01")
TEST_FIRST_TARGET = pd.Timestamp("2026-01-01")


def load_hist() -> pd.DataFrame:
    return pd.read_csv(DATASET)


def hist_upto(full: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    key = full["year"] * 12 + full["month"]
    return full[key <= origin.year * 12 + origin.month]


def make_fit_predict(full: pd.DataFrame, spec: v3.V3Spec):
    def fit_predict(train: pd.Series, horizon: int) -> pd.Series:
        origin = pd.Timestamp(train.attrs["origen"])
        h = hist_upto(full, origin)
        # cero fuga: el historico del modelo == train del harness (misma serie hasta el origen)
        own = pd.Series(h["demand_value"].to_numpy(float),
                        index=pd.DatetimeIndex(pd.to_datetime(dict(year=h["year"], month=h["month"], day=1))))
        assert own.index.max() == origin and np.allclose(own.to_numpy(), train.reindex(own.index).to_numpy(), equal_nan=True)
        return v3.predict_v3(h, horizon, spec)

    return fit_predict


def series_of(full: pd.DataFrame) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(dict(year=full["year"], month=full["month"], day=1)))
    return pd.Series(full["demand_value"].to_numpy(float), index=idx, name="demanda_kwh")


def actuals_and_exclusions():
    from training.common.data import load_monthly_demand, target_series
    from training.common.walkforward import default_exclusions

    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    return target_series(monthly, include_imputed=False), default_exclusions(mb)


def v2_rows() -> pd.DataFrame:
    r = pd.read_csv(V2_WF, parse_dates=["origen", "objetivo"])
    return r[r["modelo"] == V2_NAME]


def v2_for_origin_2026(v2: pd.DataFrame, origin: pd.Timestamp, actuals: pd.Series) -> pd.DataFrame:
    """v2 sirve cutoff = anio_origen - 1: para origenes 2026 la prediccion es la del origen 2025-12."""
    base = v2[v2["origen"] == pd.Timestamp("2025-12-01")]
    sub = base[base["objetivo"] > origin].copy()
    sub["origen"] = origin
    sub["h"] = ((sub["objetivo"].dt.year - origin.year) * 12 + sub["objetivo"].dt.month - origin.month).astype(int)
    sub = sub[sub["h"] <= HORIZON]
    sub["real"] = actuals.reindex(sub["objetivo"]).to_numpy()
    sub["err_pct"] = (sub["pred"] - sub["real"]) / sub["real"] * 100
    return sub


def run_spec(full, spec, origins, actuals, name=None) -> pd.DataFrame:
    return walk_forward(make_fit_predict(full, spec), series_of(full), origins, HORIZON, actuals, name or spec.label())


def periods(res: pd.DataFrame) -> dict:
    """Particiones TUNE / VAL / TEST de las filas de resultados."""
    o, t = res["origen"], res["objetivo"]
    return {
        "TUNE": res[(o <= TUNE_LAST_ORIGIN) & (t <= TUNE_LAST_TARGET)],
        "VAL": res[(o > TUNE_LAST_ORIGIN) & (t <= TUNE_LAST_TARGET)],
        "TEST26": res[t >= TEST_FIRST_TARGET],
    }


def metric_row(df: pd.DataFrame, exclusions) -> dict:
    ev = evaluable(df, exclusions, common_support=False)
    e = ev["err_pct"]
    return {"n": len(e), "mape": e.abs().mean(), "sesgo": e.mean(),
            "h1_12": ev.loc[ev["h"] <= 12, "err_pct"].abs().mean(),
            "h13_24": ev.loc[ev["h"] > 12, "err_pct"].abs().mean()}


def grid() -> list[v3.V3Spec]:
    out = []
    for k, w, bs, lag in itertools.product((6, 12, 24), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0, 3)):
        if bs == 0.0 and lag == 3:
            continue
        out.append(v3.V3Spec(family="A", k_recent=k, w_recent=w, beta_shrink=bs, oni_lag=lag))
    for win, sl, k, w, bs, lag in itertools.product((36, 60), ("fixed", "ols"), (12, 24), (0.5, 1.0), (0.0, 1.0), (0, 3)):
        if sl == "ols" and (k != 12 or w != 1.0):
            continue
        if bs == 0.0 and lag == 3:
            continue
        out.append(v3.V3Spec(family="B", window=win, slope=sl, k_recent=k, w_recent=w, beta_shrink=bs, oni_lag=lag))
    return out


def grid2() -> list[v3.V3Spec]:
    """Segunda etapa (refinamiento alrededor de la familia B ganadora de la etapa 1; solo TUNE decide)."""
    out = []
    for win, k, w, lag in itertools.product((24, 36, 48, 60, 72), (24, 36, 48), (0.25, 0.5, 0.75), (0, 2, 3, 4)):
        out.append(v3.V3Spec(family="B", window=win, slope="fixed", k_recent=k, w_recent=w, beta_shrink=1.0, oni_lag=lag))
    return out


def grid3() -> list[v3.V3Spec]:
    """Tercera etapa: el optimo de la etapa 2 estaba en el borde (k=48, w bajo); se extiende hacia deriva larga. oni_lag=0 FIJO por
    semantica de producto: el escenario ENSO de un mes debe mover la demanda de ESE mes (selector mes a mes)."""
    out = []
    for win, k, w in itertools.product((48, 60, 72), (48, 72, 96), (0.0, 0.25, 0.5)):
        if w == 0.0 and k != 48:
            continue
        out.append(v3.V3Spec(family="B", window=win, slope="fixed", k_recent=k, w_recent=w, beta_shrink=1.0, oni_lag=0))
    return out


#: Spec elegido por la regla "minimo MAPE en TUNE (origenes <= 2023-12, objetivos <= 2025-12)" con oni_lag=0 fijo
#: (semantica del selector ENSO mes a mes) sobre las grillas 1-3 (``tuning_grid*_v3.csv``). Congelado ANTES de mirar 2026.
CHOSEN = v3.V3Spec(family="B", window=60, slope="fixed", k_recent=72, w_recent=0.5, beta_shrink=1.0, oni_lag=0)


def final_report(spec: v3.V3Spec = CHOSEN) -> str:
    """Tabla final (se corre tras congelar ``spec``): TUNE / VAL / TEST26, por horizonte, vs v2 y naive+drift."""
    full = load_hist()
    actuals, exclusions = actuals_and_exclusions()
    series = series_of(full)
    origins = make_origins("2017-12", "2026-07", 1)
    models = {
        "v3": make_fit_predict(full, spec),
        "v3_sin_ONI": make_fit_predict(full, v3.V3Spec(**{**spec.__dict__, "beta_mode": "none"})),
        "naive_drift": BASELINES["naive_estacional_drift"],
        "naive": BASELINES["naive_estacional"],
    }
    runs = [walk_forward(fn, series, origins, HORIZON, actuals, n) for n, fn in models.items()]
    v2 = v2_rows().copy()
    v2["modelo"] = "v2"
    extra = [v2_for_origin_2026(v2_rows(), o, actuals).assign(modelo="v2") for o in origins if o >= pd.Timestamp("2026-01-01")]
    res = pd.concat(runs + [v2] + extra, ignore_index=True)
    res.to_csv(OUT_DIR / "walkforward_v3_final.csv.gz", index=False)
    lines = [f"# Resultados finales v3 (spec: {spec.label()})", ""]
    for scope, df in periods(res).items():
        ev = evaluable(df, exclusions, common_support=True)
        rows = []
        for m, g in ev.groupby("modelo"):
            e = g["err_pct"]
            rows.append({"modelo": m, "n": len(e), "mape": e.abs().mean(), "sesgo": e.mean(),
                         "mape_h1_12": g.loc[g["h"] <= 12, "err_pct"].abs().mean(),
                         "mape_h13_24": g.loc[g["h"] > 12, "err_pct"].abs().mean()})
        lines += ["", f"## {scope} (soporte comun)", "", pd.DataFrame(rows).round(2).to_markdown(index=False)]
    # Origen dic-2025: ene-ago 2026 mes a mes
    d = res[(res["origen"] == pd.Timestamp("2025-12-01")) & (res["objetivo"].dt.year == 2026) & res["real"].notna()]
    d = d[~excluded_mask(d["objetivo"], exclusions)]
    piv = d.pivot_table(index="objetivo", columns="modelo", values="err_pct").round(2)
    lines += ["", "## Origen 2025-12: error % por mes 2026", "", piv.to_markdown()]
    tot = d.groupby("modelo")[["pred", "real"]].sum()
    lines += ["", "Error del total ene-ago 2026 (origen 2025-12):", "",
              ((tot["pred"] / tot["real"] - 1) * 100).round(2).to_frame("err_total_%").to_markdown()]
    # Origenes diciembre: total anual con h1-12
    from training.common.walkforward import annual_totals

    at = annual_totals(res[res["origen"].dt.month == 12], exclusions)
    at = at[(at["h_max"] <= 12) & (at["anio"] <= 2025)]
    lines += ["", "## Total anual con origen dic previo (h1-12), error %", "",
              at.pivot_table(index="anio", columns="modelo", values="err_pct").round(2).to_markdown()]
    txt = "\n".join(lines)
    (OUT_DIR / "resultados_final_v3.md").write_text(txt, encoding="utf-8")
    print(txt)
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune2", action="store_true")
    ap.add_argument("--tune3", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.final:
        final_report()
        return
    full = load_hist()
    actuals, exclusions = actuals_and_exclusions()
    origins = make_origins("2017-12", "2025-12", 1)  # TUNE usa <= 2023-12; VAL el resto (informativo)
    if args.tune or args.tune2 or args.tune3:
        from joblib import Parallel, delayed

        specs = grid() if args.tune else (grid2() if args.tune2 else grid3())

        def one(spec):
            res = run_spec(full, spec, origins, actuals)
            p = periods(res)
            row = {"label": spec.label(), **{f"spec_{k}": v for k, v in spec.__dict__.items()}}
            for k in ("TUNE", "VAL"):
                row.update({f"{k}_{a}": b for a, b in metric_row(p[k], exclusions).items()})
            return row

        rows = Parallel(n_jobs=args.n_jobs)(delayed(one)(s) for s in specs)
        df = pd.DataFrame(rows).sort_values("TUNE_mape")
        df.to_csv(OUT_DIR / ("tuning_grid_v3.csv" if args.tune else ("tuning_grid2_v3.csv" if args.tune2 else "tuning_grid3_v3.csv")), index=False)
        # comparadores en TUNE/VAL
        for name, fn in BASELINES.items():
            res = walk_forward(fn, series_of(full), origins, HORIZON, actuals, name)
            p = periods(res)
            print(name, {k: {a: round(b, 2) for a, b in metric_row(p[k], exclusions).items()} for k in ("TUNE", "VAL")})
        v2 = v2_rows()
        p = periods(v2)
        print("v2", {k: {a: round(b, 2) for a, b in metric_row(p[k], exclusions).items()} for k in ("TUNE", "VAL")})
        pd.set_option("display.width", 250)
        print(df[["label", "TUNE_mape", "TUNE_sesgo", "TUNE_h1_12", "VAL_mape", "VAL_sesgo"]].head(25).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
