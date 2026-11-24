"""Linea base honesta de los 4 modelos de forecast mensual (ids 1 RF, 2 XGB, 3 SARIMAX, 4 Clima+ONI).

Ejecuta el codigo REAL de produccion (ForecastService.predict) contra la BD de .env (ML_DATABASE_URL)
dos veces por modelo:
  - "tope":  tal cual lo sirve la API (cap_xgb_sarimax_error para id 1/2 + KNOWN_MONTH_OVERRIDES feb-2026)
  - "crudo": monkeypatch de cap_xgb_sarimax_error y apply_known_month_overrides a identidad
y compara contra el real mensual de PRONOSTICO_demands (solo meses completos, solo txf, sin filas duplicadas).

Ademas calcula baselines ingenuos sin modelo (honestos, solo info disponible al inicio de la ventana):
  - naive_estacional: mismo mes del ano anterior
  - naive_estacional_drift: mismo mes ano anterior * (total ult. 12m / total 12m previos)

Uso:
  python docs/work/00_entorno/scripts/baseline.py [--start 2026-01-01 --end 2027-12-31] [--tag 2026]
Salida: tabla por consola + docs/work/00_entorno/baseline_<tag>.csv y baseline_<tag>_resumen.csv

NOTA division de mercado: desde 2026-09 la BD guarda ~70 % del mercado (ver DIAGNOSTICO.md). Las metricas
se reportan separadas para meses pre-corte (<= 2026-08) y post-corte.
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from sqlalchemy import text  # noqa: E402
from api.config.database import SessionLocal  # noqa: E402
from api.config.schemas import ForecastSchema  # noqa: E402
import api.services.forecast as fc  # noqa: E402

OUT_DIR = ROOT / "docs" / "work" / "00_entorno"
SPLIT_DATE = pd.Period("2026-08", "M")  # ultimo mes 100 % mercado
MODELS = {1: "RF", 2: "XGB", 3: "SARIMAX", 4: "CLIMA_ONI"}


def real_monthly(db) -> pd.Series:
    """Real mensual (kWh) desde PRONOSTICO_demands: txf, 1 fila por dia (si hay duplicados se toma la de
    mayor total, que es la escala correcta), solo meses con todos los dias."""
    q = text('''
        SELECT fecha::date AS d, MAX(total) AS total, COUNT(*) AS n
        FROM "PRONOSTICO_demands" WHERE file_type = 'txf' GROUP BY 1 ORDER BY 1''')
    df = pd.DataFrame(db.execute(q).fetchall(), columns=["d", "total", "n"])
    df["d"] = pd.to_datetime(df["d"])
    df["ym"] = df["d"].dt.to_period("M")
    g = df.groupby("ym").agg(total=("total", "sum"), ndays=("d", "nunique"))
    g = g[g["ndays"] == g.index.days_in_month]
    return g["total"].astype(float)


def run_model(db, model_id, start, end, raw: bool):
    orig_cap, orig_ovr = fc.cap_xgb_sarimax_error, fc.apply_known_month_overrides
    if raw:
        fc.cap_xgb_sarimax_error = lambda df, **k: df
        fc.apply_known_month_overrides = lambda df, **k: df.assign(override_aplicado=False)
    try:
        search = ForecastSchema(id=model_id, user_id=1, session_id=1, start_date=start, end_date=end)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resp = fc.ForecastService().predict(search=search, db=db)
    finally:
        fc.cap_xgb_sarimax_error, fc.apply_known_month_overrides = orig_cap, orig_ovr
    if "monthly_predictions" not in resp:
        raise RuntimeError(f"id={model_id} fallo: {resp}")
    mp = resp["monthly_predictions"]
    s = pd.Series(mp["valores_pronostico"], index=pd.PeriodIndex(pd.to_datetime(mp["fechas"]), freq="M"), dtype=float)
    return s, resp


def naive_baselines(real: pd.Series, start: pd.Period, periods: pd.PeriodIndex):
    hist = real[real.index < start]
    last12 = hist.iloc[-12:].sum()
    prev12 = hist.iloc[-24:-12].sum()
    drift = last12 / prev12 if prev12 > 0 else 1.0
    out = {}
    sn, snd = [], []
    for p in periods:
        k = 1
        while (p - 12 * k) not in hist.index and k < 5:
            k += 1
        base = hist.get(p - 12 * k, np.nan)
        sn.append(base)
        snd.append(base * drift ** k)
    out["NAIVE_EST"] = pd.Series(sn, index=periods)
    out["NAIVE_EST_DRIFT"] = pd.Series(snd, index=periods)
    return out


def metrics(err: pd.Series):
    e = err.dropna()
    if e.empty:
        return dict(n=0, MAPE=np.nan, sesgo=np.nan, max_abs=np.nan)
    return dict(n=len(e), MAPE=e.abs().mean(), sesgo=e.mean(), max_abs=e.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2027-12-31")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--models", default="1,2,3,4")
    a = ap.parse_args()
    tag = a.tag or f"{a.start[:4]}_{a.end[:4]}"
    db = SessionLocal()
    real = real_monthly(db)
    periods = pd.period_range(a.start, a.end, freq="M")
    table = pd.DataFrame(index=periods)
    table["real"] = real.reindex(periods)
    annual = {}
    for mid in [int(x) for x in a.models.split(",")]:
        name = MODELS[mid]
        for raw in (True, False):
            col = f"{name}_{'crudo' if raw else 'tope'}"
            try:
                s, resp = run_model(db, mid, a.start, a.end, raw)
                table[col] = s.reindex(periods)
                if raw:
                    annual[name] = resp.get("annual_predictions", [])[-(periods.year.nunique()):]
            except Exception as ex:  # noqa: BLE001
                print(f"[WARN] {col}: {ex}")
                table[col] = np.nan
    for k, v in naive_baselines(real, periods[0], periods).items():
        table[k] = v
    pred_cols = [c for c in table.columns if c != "real"]
    for c in pred_cols:
        table[f"err%_{c}"] = (table[c] - table["real"]) / table["real"] * 100

    # resumen
    rows = []
    pre = table.index <= SPLIT_DATE
    post = table.index > SPLIT_DATE
    for c in pred_cols:
        e = table[f"err%_{c}"]
        r = {"serie": c}
        for lab, mask in (("pre_corte", pre), ("post_corte", post), ("todo", np.ones(len(table), bool))):
            m = metrics(e[mask])
            r.update({f"{lab}_{k}": v for k, v in m.items()})
        # error del total anual sobre meses pre-corte con real (mismos meses)
        mm = pre & table["real"].notna() & table[c].notna()
        r["anual_pre_corte_err%"] = (table.loc[mm, c].sum() / table.loc[mm, "real"].sum() - 1) * 100 if mm.any() else np.nan
        rows.append(r)
    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    gwh = table[["real"] + pred_cols].div(1e6).round(1)
    print("\n=== Valores mensuales (GWh, 100 % escala BD) ===")
    print(gwh.to_string())
    print("\n=== Error % mensual (pred-real)/real ===")
    print(table[[f"err%_{c}" for c in pred_cols]].round(2).to_string())
    print("\n=== Resumen (MAPE/sesgo en %; pre_corte = meses <= 2026-08) ===")
    print(summary.round(2).to_string())
    print("\nTotales anuales crudos devueltos por la API (kWh):", {k: [round(x / 1e9, 3) for x in v] for k, v in annual.items()})

    out = table.copy()
    out.index = out.index.astype(str)
    out.to_csv(OUT_DIR / f"baseline_{tag}.csv", index_label="mes")
    summary.to_csv(OUT_DIR / f"baseline_{tag}_resumen.csv", index=False)
    print(f"\nGuardado: {OUT_DIR / f'baseline_{tag}.csv'} y _resumen.csv")
    db.close()


if __name__ == "__main__":
    main()
