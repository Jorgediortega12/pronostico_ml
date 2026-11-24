"""Evaluacion vectorizada de la correccion de nivel sobre el cache de predicciones (``build_cache``).

Semantica identica a ``training.common.level_adapt.adjust`` (hay un test de paridad): para el origen ``o`` y
``k`` meses de residuos, el origen de referencia es ``o - k`` y los residuos son
``log(real_m / pred(o-k, m))`` para ``m = o-k+1 .. o``.

Particiones (regla de ``docs/work/11_nivel_adaptativo/PLAN.md``):

* TUNE : origenes 2017-12..2023-12, objetivos <= 2025-12 (unico uso permitido para elegir hiperparametros).
* VAL  : origenes 2024-01..2025-12, objetivos <= 2025-12.
* TEST26: objetivos >= 2026-01 (origenes 2024-01..2026-07). Se mira UNA vez, al final.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from training.common.data import load_monthly_demand, target_series
from training.common.level_adapt import LevelSpec, covid_exclusions, horizon_decay
from training.forecast_climate import walkforward_clima_v3 as wf3

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "work" / "11_nivel_adaptativo" / "resultados"
HORIZON = 24
EVAL_FIRST, EVAL_LAST = pd.Timestamp("2017-12-01"), pd.Timestamp("2026-07-01")
TUNE_LAST_ORIGIN, TUNE_LAST_TARGET = pd.Timestamp("2023-12-01"), pd.Timestamp("2025-12-01")
TEST_FIRST_TARGET = pd.Timestamp("2026-01-01")
MODELS = ("rf", "xgb", "sarimax", "clima", "naive_drift")
LABEL = {"rf": "RF (id 1)", "xgb": "XGB (id 2)", "sarimax": "SARIMAX (id 3)", "clima": "Clima v3 (id 4)", "naive_drift": "naive+drift"}


def _midx(o: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(o, periods=n, freq="MS")


@dataclass
class Data:
    origins: pd.DatetimeIndex  # origenes del cache (contiguos)
    P: dict  # modelo -> array (n_origenes_cache, HORIZON) pred
    hist: dict  # modelo -> Serie historia para residuos (kWh)
    real: pd.Series  # reales observados (no imputados)
    excl_resid: pd.DatetimeIndex  # meses COVID (no entran a residuos ni metricas)
    last_month: pd.Timestamp  # ultimo mes con real

    def oi(self, o: pd.Timestamp) -> int:
        return int((o.year - self.origins[0].year) * 12 + o.month - self.origins[0].month)


def load_data() -> Data:
    cache = pd.read_csv(OUT / "pred_cache.csv.gz", parse_dates=["origen", "objetivo"])
    origins = pd.DatetimeIndex(sorted(cache["origen"].unique()))
    assert len(origins) == (origins[-1].year - origins[0].year) * 12 + origins[-1].month - origins[0].month + 1
    P = {}
    for m, g in cache.groupby("modelo"):
        arr = np.full((len(origins), HORIZON), np.nan)
        i = ((g["origen"].dt.year - origins[0].year) * 12 + g["origen"].dt.month - origins[0].month).to_numpy()
        arr[i, g["h"].to_numpy() - 1] = g["pred"].to_numpy()
        P[m] = arr
    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    train = target_series(monthly, market_break=mb)
    real = target_series(monthly, include_imputed=False)
    real = real[real.index < mb.first_post_month] if mb is not None else real
    clima = wf3.series_of(wf3.load_hist())
    hist = {"rf": train, "xgb": train, "sarimax": train, "naive_drift": train, "clima": clima}
    lo, hi = covid_exclusions()[0]
    excl = pd.date_range(lo, hi, freq="MS")
    return Data(origins, P, hist, real, excl, real.index.max())


def _eval_origins(d: Data) -> pd.DatetimeIndex:
    last_o = min(EVAL_LAST, d.last_month - pd.offsets.MonthBegin(1))
    return pd.date_range(EVAL_FIRST, last_o, freq="MS")


def base_arrays(d: Data, model: str):
    """``logerr0[i, h-1] = log(pred/real)`` (NaN si no evaluable) para los origenes de evaluacion."""
    E = _eval_origins(d)
    le = np.full((len(E), HORIZON), np.nan)
    real = d.real
    for a, o in enumerate(E):
        t = _midx(o + pd.offsets.MonthBegin(1), HORIZON)
        y = real.reindex(t).to_numpy()
        p = d.P[model][d.oi(o)]
        ok = np.isfinite(y) & np.isfinite(p) & ~np.isin(t, d.excl_resid)
        le[a, ok] = np.log(p[ok] / y[ok])
    return E, le


def resid_matrix(d: Data, model: str, E: pd.DatetimeIndex, k: int) -> np.ndarray:
    """``R[i, j]`` = residuo log del mes ``E[i] - j`` (j=0 el mas reciente) con origen de referencia ``E[i]-k``."""
    h = d.hist[model]
    R = np.full((len(E), k), np.nan)
    for a, o in enumerate(E):
        r = o - pd.DateOffset(months=k)
        pr = d.P[model][d.oi(r)]
        for j in range(k):
            m = o - pd.DateOffset(months=j)
            y = h.get(m, np.nan)
            p = pr[k - j - 1]
            if m in d.excl_resid or not (np.isfinite(y) and np.isfinite(p) and y > 0 and p > 0):
                continue
            R[a, j] = np.log(y / p)
    return R


def level_from_R(R: np.ndarray, spec: LevelSpec) -> np.ndarray:
    k = R.shape[1]
    age = np.arange(k, dtype=float)
    w = np.ones(k) if spec.hl_w is None else 0.5 ** (age / spec.hl_w)
    ok = np.isfinite(R)
    n = ok.sum(1)
    if spec.est == "mean":
        num = np.where(ok, R * w, 0.0).sum(1)
        den = np.where(ok, w, 0.0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            lv = num / den
    else:
        lv = np.zeros(len(R))
        for a in range(len(R)):
            v = R[a, ok[a]]
            if len(v):
                ww = w[ok[a]]
                o = np.argsort(v)
                cw = np.cumsum(ww[o])
                lv[a] = v[o][np.searchsorted(cw, 0.5 * cw[-1])]
    lv = np.nan_to_num(lv)
    if spec.agree > 0:
        frac = np.where(ok, np.sign(R) == np.sign(lv)[:, None], False).sum(1) / np.maximum(n, 1)
        lv = np.where(frac >= spec.agree - 1e-12, lv, 0.0)
    lv = np.sign(lv) * np.maximum(np.abs(lv) - spec.tau, 0.0)
    return np.where(n >= spec.min_obs, lv, 0.0)


def correction_log(level: np.ndarray, spec: LevelSpec) -> np.ndarray:
    """``c[i, h-1]`` = ajuste en log para cada origen y horizonte."""
    c = np.clip(spec.phi * level, -spec.cap, spec.cap)
    return c[:, None] * horizon_decay(np.arange(1, HORIZON + 1), spec)[None, :]


def masks(E: pd.DatetimeIndex) -> dict:
    t = E.values[:, None].astype("datetime64[M]") + np.arange(1, HORIZON + 1)[None, :].astype("timedelta64[M]")
    t = pd.DatetimeIndex(t.ravel()).values.reshape(len(E), HORIZON)
    o = np.broadcast_to(E.values[:, None], (len(E), HORIZON))
    tune = (o <= np.datetime64(TUNE_LAST_ORIGIN)) & (t <= np.datetime64(TUNE_LAST_TARGET))
    val = (o > np.datetime64(TUNE_LAST_ORIGIN)) & (t <= np.datetime64(TUNE_LAST_TARGET))
    test = t >= np.datetime64(TEST_FIRST_TARGET)
    return {"TUNE": tune, "VAL": val, "TEST26": test, "TARGETS": t}


def stats(le: np.ndarray, mask: np.ndarray, hmask: np.ndarray | None = None) -> dict:
    m = mask & np.isfinite(le)
    if hmask is not None:
        m = m & hmask
    e = (np.exp(le[m]) - 1.0) * 100.0
    return {"n": int(m.sum()), "mape": float(np.abs(e).mean()) if len(e) else np.nan, "sesgo": float(e.mean()) if len(e) else np.nan}


def hmask(lo: int, hi: int) -> np.ndarray:
    h = np.arange(1, HORIZON + 1)
    return ((h >= lo) & (h <= hi))[None, :]


def grid() -> list[LevelSpec]:
    out = []
    for k, hw, est, phi, hh, tau, ag in itertools.product(
            (3, 4, 6, 9), (None, 2.0, 4.0), ("mean", "median"), (0.5, 0.75, 1.0), (3.0, 6.0, 12.0, 24.0, None),
            (0.0, 0.01, 0.02, 0.03), (0.0, 0.7, 0.85, 1.0)):
        out.append(LevelSpec(k=k, hl_w=hw, est=est, phi=phi, hl_h=hh, cap=0.20, min_obs=min(3, k), tau=tau, agree=ag))
    return out


def run_grid(d: Data, models=("rf", "xgb", "clima"), specs=None) -> pd.DataFrame:
    specs = specs or grid()
    ms = None
    cache = {}
    rows = []
    base = {}
    for m in models:
        E, le0 = base_arrays(d, m)
        ms = masks(E)
        base[m] = (le0, stats(le0, ms["TUNE"])["mape"])
    for s in specs:
        row = {**{f"s_{k}": v for k, v in s.to_dict().items()}}
        rel = []
        for m in models:
            key = (m, s.k)
            if key not in cache:
                cache[key] = resid_matrix(d, m, E, s.k)
            lv = level_from_R(cache[key], s)
            le = base[m][0] + correction_log(lv, s)
            st = stats(le, ms["TUNE"])
            row[f"TUNE_{m}"] = st["mape"]
            row[f"TUNE_{m}_sesgo"] = st["sesgo"]
            rel.append(st["mape"] / base[m][1])
        row["TUNE_rel_media"] = float(np.mean(rel))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("TUNE_rel_media")


#: Spec elegida con SOLO TUNE (origenes <= 2023-12, objetivos <= 2025-12): regla ex-ante = entre las specs con
#: ``TUNE_rel_media`` a menos de 0.01 de la mejor, la de ventana ``k`` mas larga (menos ruido de un solo mes) y,
#: a igualdad, la de menor puntaje (grillas ``tuning_grid*.csv``). ``cap`` = percentil 99.5 de |phi*L| en TUNE
#: (~0.05-0.06) redondeado. Congelada ANTES de mirar 2026.
CHOSEN = LevelSpec(k=6, hl_w=2.0, est="median", phi=0.5, hl_h=3.0, cap=0.06, min_obs=3, tau=0.0, agree=0.0)
START_MONTHS = {"inicio ene (origen dic)": 12, "inicio may (origen abr)": 4, "inicio jul (origen jun)": 6}


def corrected(d: Data, model: str, spec: LevelSpec):
    E, le0 = base_arrays(d, model)
    c = correction_log(level_from_R(resid_matrix(d, model, E, spec.k), spec), spec)
    return E, le0, le0 + c, c


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def final_report(spec: LevelSpec = CHOSEN, test: bool = False) -> str:
    """Tablas finales. ``test=True`` incluye TEST26 (objetivos >= 2026-01): correrlo UNA vez con la spec congelada."""
    d = load_data()
    periods = ["TUNE", "VAL"] + (["TEST26"] if test else [])
    arrs = {m: corrected(d, m, spec) for m in ("rf", "xgb", "sarimax", "clima")}
    E, le_nd = base_arrays(d, "naive_drift")
    ms = masks(E)
    L = [f"# Resultados nivel adaptativo (spec: {spec.to_dict()})", ""]
    # 1) por periodo
    for per in periods:
        rows = []
        for m in ("rf", "xgb", "sarimax", "clima"):
            _, le0, le1, _ = arrs[m]
            a, b = stats(le0, ms[per]), stats(le1, ms[per])
            rows.append({"modelo": LABEL[m], "n": a["n"], "MAPE sin": a["mape"], "MAPE con": b["mape"], "delta": b["mape"] - a["mape"],
                         "sesgo sin": a["sesgo"], "sesgo con": b["sesgo"]})
        nd = stats(le_nd, ms[per])
        rows.append({"modelo": "naive+drift", "n": nd["n"], "MAPE sin": nd["mape"], "MAPE con": np.nan, "delta": np.nan, "sesgo sin": nd["sesgo"], "sesgo con": np.nan})
        L += [f"## {per} (todos los h 1-24)", "", _md(pd.DataFrame(rows).round(2)), ""]
    # 2) por horizonte
    for per in periods[1:]:
        rows = []
        for m in ("rf", "xgb", "sarimax", "clima"):
            _, le0, le1, _ = arrs[m]
            r = {"modelo": LABEL[m]}
            for lo, hi in ((1, 3), (4, 6), (7, 12), (13, 24)):
                a, b = stats(le0, ms[per], hmask(lo, hi)), stats(le1, ms[per], hmask(lo, hi))
                r[f"h{lo}-{hi} sin"], r[f"h{lo}-{hi} con"] = a["mape"], b["mape"]
            rows.append(r)
        L += [f"## {per}: MAPE por horizonte (sin -> con)", "", _md(pd.DataFrame(rows).round(2)), ""]
    # 3) por inicio
    for per in periods[1:]:
        rows = []
        for lab, mo in START_MONTHS.items():
            sel = ms[per] & (E.month == mo)[:, None]
            for m in ("rf", "xgb", "sarimax", "clima"):
                _, le0, le1, _ = arrs[m]
                a, b = stats(le0, sel), stats(le1, sel)
                rows.append({"inicio": lab, "modelo": LABEL[m], "n": a["n"], "MAPE sin": a["mape"], "MAPE con": b["mape"], "sesgo sin": a["sesgo"], "sesgo con": b["sesgo"]})
            nd = stats(le_nd, sel)
            rows.append({"inicio": lab, "modelo": "naive+drift", "n": nd["n"], "MAPE sin": nd["mape"], "MAPE con": np.nan, "sesgo sin": nd["sesgo"], "sesgo con": np.nan})
        L += [f"## {per}: por mes de inicio", "", _md(pd.DataFrame(rows).round(2)), ""]
    # 4) por anio del origen (incluye anios sin shock)
    rows = []
    allm = np.ones_like(ms["TUNE"])
    for y in sorted(set(E.year)):
        if y >= 2026 and not test:
            continue
        sel = allm & (E.year == y)[:, None] & (ms["TUNE"] | ms["VAL"] | (ms["TEST26"] if test else False))
        r = {"origen anio": y}
        for m in ("rf", "xgb", "sarimax", "clima"):
            _, le0, le1, _ = arrs[m]
            a, b = stats(le0, sel), stats(le1, sel)
            r["n"] = a["n"]
            r[f"{m} sin"], r[f"{m} con"] = a["mape"], b["mape"]
        rows.append(r)
    L += ["## Por anio del origen (objetivos <= 2025-12" + (" y >= 2026" if test else "") + ")", "", _md(pd.DataFrame(rows).round(2)), ""]
    # 5) tasa de victoria por origen
    rows = []
    for per in periods:
        for m in ("rf", "xgb", "sarimax", "clima"):
            _, le0, le1, _ = arrs[m]
            wins = worse = tot = 0
            for i in range(len(E)):
                mk = ms[per][i] & np.isfinite(le0[i])
                if mk.sum() < 3:
                    continue
                a = (np.abs(np.exp(le0[i][mk]) - 1) * 100).mean()
                b = (np.abs(np.exp(le1[i][mk]) - 1) * 100).mean()
                tot += 1
                wins += b < a - 1e-9
                worse += b > a + 0.5
            rows.append({"periodo": per, "modelo": LABEL[m], "origenes": tot, "mejora %": 100 * wins / tot, "empeora > 0.5pp %": 100 * worse / tot})
    L += ["## Tasa de origenes (MAPE del origen, sin -> con)", "", _md(pd.DataFrame(rows).round(1)), ""]
    txt = chr(10).join(L)
    (OUT / ("resultados_final.md" if test else "resultados_dev.md")).write_text(txt, encoding="utf-8")
    return txt


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--final", action="store_true", help="incluye TEST26 (correr UNA vez, spec congelada)")
    a = ap.parse_args(argv)
    if a.grid:
        d = load_data()
        t = run_grid(d)
        t.to_csv(OUT / "tuning_grid.csv", index=False)
        pd.set_option("display.width", 250)
        print(t.head(25).round(3).to_string(index=False))
    if a.report or a.final:
        print(final_report(test=a.final))


if __name__ == "__main__":
    main()
