"""Walk-forward honesto del modelo id=4 (clima+ONI): antes vs. después, selección de hiperparámetros,
test final 2026 y sensibilidad a escenarios ONI.

Protocolo (ver ``docs/work/04_clima/REPORTE.md``):

* Harness común ``training/common/walkforward.py`` (ventana expansiva). El modelo, como la API, usa solo
  **años completos** hasta el corte: origen ``o`` → ``cutoff = o.year`` si ``o`` es diciembre, si no
  ``o.year − 1`` (igual que ``ForecastService.predict``: ``cutoff_year = start_year − 1``).
* Cero fuga: todo (tabla de entrenamiento, ONI proyectado, climatología, shares) se calcula solo con datos
  ``year <= cutoff``; el ONI del año objetivo es la proyección ``project_oni_future`` al corte.
* Hiperparámetros: SOLO orígenes ``<= 2023-12`` con objetivos ``<= 2025-12``. 2026 (origen 2025-12) es el
  test final y se mira una sola vez (``--final``).

Uso::

    python -m training.forecast_climate.walkforward_clima --tune     # grid de selección (<= 2023-12)
    python -m training.forecast_climate.walkforward_clima --final    # tabla completa + test 2026 + escenarios
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import warnings
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from training.common.walkforward import (
    BASELINES,
    default_exclusions,
    evaluable,
    excluded_mask,
    make_origins,
    summarize,
    walk_forward,
)
from training.forecast_climate import clima_oni_model as com

warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
OUT_DIR = REPO_ROOT / "docs" / "work" / "04_clima" / "resultados"
BACKUP_DATASETS = REPO_ROOT / "docs" / "work" / "backup_modelos_originales" / "climate_datasets"
BACKUP_MODELS = REPO_ROOT / "docs" / "work" / "backup_modelos_originales" / "models" / "climate_model"

TUNE_LAST_ORIGIN = pd.Timestamp("2023-12-01")
TUNE_LAST_TARGET = pd.Timestamp("2025-12-01")
HORIZON = 24


def cutoff_for_origin(origin: pd.Timestamp) -> int:
    origin = pd.Timestamp(origin)
    return origin.year if origin.month == 12 else origin.year - 1


# --------------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------------
class ClimaData:
    """Datasets mensual/anual (esquema de inferencia) + serie target limpia + caché de features."""

    def __init__(self, mensual: pd.DataFrame, anual: pd.DataFrame):
        self.m = mensual.sort_values(["year", "month"]).reset_index(drop=True)
        self.a = anual.sort_values("year").reset_index(drop=True)
        idx = pd.to_datetime(dict(year=self.m["year"], month=self.m["month"], day=1))
        self.series = pd.Series(self.m["demand_value"].to_numpy(float), index=pd.DatetimeIndex(idx), name="demanda_kwh")
        self._rows: dict = {}
        self._tables: dict = {}

    def upto(self, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.m[self.m["year"] <= cutoff].copy(), self.a[self.a["year"] <= cutoff].copy()

    def inference_rows(self, cutoff: int, years: int) -> pd.DataFrame:
        key = (cutoff, years)
        if key not in self._rows:
            hm, ha = self.upto(cutoff)
            self._rows[key] = com.capture_inference_rows(hm, ha, cutoff, years)
        return self._rows[key]

    def training_table(self, cutoff: int, oni_source: str) -> pd.DataFrame:
        """Tabla de entrenamiento con años objetivo ``<= cutoff`` (features capturadas con corte ``t−1``)."""
        key = oni_source
        if key not in self._tables:
            years = range(int(self.a["year"].min()) + 1, int(self.a["year"].max()) + 1)
            full = self.a[self.a.get("n_meses", pd.Series(12, index=self.a.index)).fillna(12) >= 12]
            rows = []
            for t in years:
                if t not in set(full["year"]):
                    continue
                hm, ha = self.upto(t)  # filas de t solo para el target/observado; las features cortan en t-1
                tab = com.build_training_table(hm, ha, [t], oni_source)
                rows.append(tab)
            self._tables[key] = pd.concat(rows)
        tab = self._tables[key]
        return tab[tab.index <= cutoff]


@lru_cache(maxsize=1)
def load_clean_data() -> ClimaData:
    from training.forecast_climate.build_clean_dataset import build_from_db

    m, a = build_from_db()
    return ClimaData(m, a)


def load_original_data() -> ClimaData:
    m = pd.read_csv(BACKUP_DATASETS / "dataset_mensual.csv")
    a = pd.read_csv(BACKUP_DATASETS / "dataset_anual.csv")
    a["n_meses"] = a["year"].map(m.groupby("year").size()).fillna(12)
    return ClimaData(m, a)


# --------------------------------------------------------------------------------------------------
# Trayectoria de pronóstico (réplica de predict_future_climate, con parche opcional de lag ONI)
# --------------------------------------------------------------------------------------------------
def month_shares(hist_monthly: pd.DataFrame, n_years_shape: int) -> dict[int, float]:
    """Igual que el paso 4 de ``predict_future_climate`` (sin override ENSO)."""
    h = hist_monthly.copy()
    h["pct"] = h["demand_value"] / h.groupby("year")["demand_value"].transform("sum")
    full = h[h.groupby("year")["month"].transform("count") == 12]
    if full.empty:
        full = h
    yrs = sorted(full["year"].unique())[-n_years_shape:]
    src = full[full["year"].isin(yrs)]
    return {m: float(src.loc[src["month"] == m, "pct"].mean()) for m in range(1, 13)}


def forecast_path(
    model: com.GrowthModel,
    data: ClimaData,
    cutoff: int,
    years: int,
    n_years_shape: int,
) -> pd.Series:
    """Pronóstico mensual (kWh) para ``cutoff+1 … cutoff+years``, replicando la recursión anual de la
    inferencia. Con features de parche (``oni_mean_anio_lag1``) aplica la semántica de
    ``docs/work/04_clima/forecast_clima_patch.md``."""
    rows = data.inference_rows(cutoff, years)
    hm, ha = data.upto(cutoff)
    coefs = com.expand_coefs(model.coefs)
    lag1 = float(rows.iloc[0]["lag_1"])
    grl = float(rows.iloc[0]["growth_rate_lag1"])
    oni_prev = float(ha.iloc[-1]["oni_mean_anio"])
    annual = []
    for i, (yr, r) in enumerate(rows.iterrows()):
        x = {k: float(r[k]) for k in r.index if k in com.INFERENCE_ANNUAL_FEATURES}
        x["lag_1"] = lag1
        x["growth_rate_lag1"] = grl
        x["oni_mean_anio_lag1"] = oni_prev
        log = lag1 + model.mu + sum(c * x[k] for k, c in coefs.items())
        annual.append(float(np.expm1(log)))
        grl, lag1, oni_prev = log - lag1, log, x["oni_mean_anio"]
    sh = month_shares(hm, n_years_shape)
    tot = sum(sh.values())
    vals, idx = [], []
    for i, yr in enumerate(rows.index):
        for mth in range(1, 13):
            idx.append(pd.Timestamp(year=int(yr), month=mth, day=1))
            vals.append(annual[i] * sh[mth] / tot)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def real_inference_path(data: ClimaData, cutoff: int, years: int, artifacts, n_years_shape: int = 2,
                        enso: Optional[dict] = None) -> pd.Series:
    """Pronóstico con el código REAL de ``forecast.py`` (``predict_future_climate``) y artefactos dados."""
    sx, mdl, sy = artifacts
    hm, ha = data.upto(cutoff)
    p = com.make_predictor(hm, ha, sx, mdl, sy)
    with contextlib.redirect_stdout(io.StringIO()):
        _, res = p.predict_future_climate(forecast_years=years, cutoff_year=cutoff, n_years_shape=n_years_shape,
                                          enso_phase_by_month=enso)
    return pd.Series(res["prediccion"].to_numpy(float), index=pd.DatetimeIndex(res["fecha"]))


# --------------------------------------------------------------------------------------------------
# fit_predict para el harness
# --------------------------------------------------------------------------------------------------
def make_fit_predict(data: ClimaData, spec: com.GrowthModelSpec):
    cache: dict[int, pd.Series] = {}

    def fit_predict(train: pd.Series, horizon: int) -> pd.Series:
        origin = pd.Timestamp(train.attrs["origen"])
        c = cutoff_for_origin(origin)
        # Garantía de cero fuga: los datos que usa el modelo (<= cutoff) coinciden con el train del harness.
        own = data.series[data.series.index.year <= c]
        assert own.index.max() <= origin and np.allclose(own.to_numpy(), train.reindex(own.index).to_numpy()), \
            "datos del modelo != train del harness"
        if c not in cache:
            tab = data.training_table(c, spec.oni_source)
            model = com.fit_growth_model(tab, spec)
            cache[c] = forecast_path(model, data, c, 3, spec.n_years_shape)
        targets = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return cache[c].reindex(targets)

    return fit_predict


def make_fit_predict_original_recipe(data: ClimaData, n_years_shape: int = 2):
    """"Antes" honesto: receta original (Ridge alpha=1 sobre 12 features con clima/ONI OBSERVADOS del mismo
    año, ``train_candidates.prepare_annual_features``), reentrenada por origen, servida con el código real."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    cache: dict[int, pd.Series] = {}

    def fit_predict(train: pd.Series, horizon: int) -> pd.Series:
        origin = pd.Timestamp(train.attrs["origen"])
        c = cutoff_for_origin(origin)
        if c not in cache:
            df = data.a[data.a["year"] <= c].sort_values("year").reset_index(drop=True).copy()
            logd = np.log1p(df["demand_total"])
            df["lag_1"], df["lag_2"] = logd.shift(1), logd.shift(2)
            df["year_index"] = np.arange(len(df))
            df["year_index_norm"] = df["year_index"] / max(len(df) - 1, 1)
            df["growth_rate_lag1"] = logd.diff().shift(1)
            df = df.dropna(subset=com.INFERENCE_ANNUAL_FEATURES)
            X = df[com.INFERENCE_ANNUAL_FEATURES]
            sx = StandardScaler().fit(X)
            sy = StandardScaler().fit(np.log1p(df[["demand_total"]].to_numpy()))
            mdl = Ridge(alpha=1.0).fit(sx.transform(X), sy.transform(np.log1p(df[["demand_total"]].to_numpy())).ravel())
            cache[c] = real_inference_path(data, c, 3, (sx, mdl, sy), n_years_shape)
        targets = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return cache[c].reindex(targets)

    return fit_predict


def make_fit_predict_artifact(data: ClimaData, model_dir: Path, n_years_shape: int = 2):
    """Artefacto productivo fijo (sin reentrenar): para años <= su fin de entrenamiento es IN-SAMPLE."""
    import joblib

    arts = tuple(joblib.load(model_dir / f) for f in ("annual_scaler_x.pkl", "annual_model.pkl", "annual_scaler_y.pkl"))
    cache: dict[int, pd.Series] = {}

    def fit_predict(train: pd.Series, horizon: int) -> pd.Series:
        origin = pd.Timestamp(train.attrs["origen"])
        c = cutoff_for_origin(origin)
        if c not in cache:
            cache[c] = real_inference_path(data, c, 3, arts, n_years_shape)
        targets = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return cache[c].reindex(targets)

    return fit_predict


# --------------------------------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------------------------------
def run(models: dict, data: ClimaData, origins, actuals: pd.Series, max_target=None) -> pd.DataFrame:
    runs = [walk_forward(fn, data.series, origins, HORIZON, actuals, name) for name, fn in models.items()]
    res = pd.concat(runs, ignore_index=True)
    if max_target is not None:
        res.loc[res["objetivo"] > max_target, "real"] = np.nan
    return res


def summary_table(res: pd.DataFrame, exclusions) -> pd.DataFrame:
    s = summarize(res, exclusions)
    cols = ["modelo", "n_origenes", "n", "mape", "sesgo", "max_ape", "mape_h1_12", "mape_h13_24", "mape_anual",
            "sesgo_anual"]
    return s[cols].round(2)


def actual_series() -> tuple[pd.Series, list]:
    from training.common.data import load_monthly_demand, target_series

    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    return target_series(monthly, include_imputed=False), default_exclusions(mb)


def grid_specs() -> list[com.GrowthModelSpec]:
    forms = [(), ("oni_mean_anio",), (com.ONI_DELTA,)]
    out = []
    for feats, src, win, excl, nys, ball in itertools.product(
        forms, ("observado", "proyectado"), (None, 8, 5, 3), ((), (2020, 2021), (2020, 2021, 2022)), (2, 3, 5),
        (False, True),
    ):
        if not feats and (src == "proyectado" or ball):
            continue  # la fuente ONI / el alcance de beta no aplican sin ONI
        if ball and win is None:
            continue  # idéntico a ball=False
        out.append(com.GrowthModelSpec(extra_features=feats, oni_source=src, window_years=win, exclude_years=excl,
                                       n_years_shape=nys, beta_all_history=ball))
    return out


def tune(dec_only: bool = False) -> pd.DataFrame:
    data = load_clean_data()
    actuals, exclusions = actual_series()
    origins_dec = make_origins("2017-12", "2023-12", 12)
    origins_m = make_origins("2017-12", "2023-12", 1)
    rows = []
    for spec in grid_specs():
        fp = make_fit_predict(data, spec)
        row = {"label": spec.label(), "n_years_shape": spec.n_years_shape, **{f"spec_{k}": v for k, v in asdict(spec).items()}}
        for tag, origins in (("dic", origins_dec), ("mens", None if dec_only else origins_m)):
            if origins is None:
                continue
            res = walk_forward(fp, data.series, origins, HORIZON, actuals, "m")
            res.loc[res["objetivo"] > TUNE_LAST_TARGET, "real"] = np.nan
            ev = evaluable(res, exclusions)
            row[f"mape_{tag}"] = ev["err_pct"].abs().mean()
            row[f"sesgo_{tag}"] = ev["err_pct"].mean()
            row[f"n_{tag}"] = len(ev)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mape_dic")
    # Baselines sobre el mismo protocolo (mismos orígenes y tope de objetivos).
    for tag, origins in (("dic", origins_dec), ("mens", origins_m)):
        for name, fn in BASELINES.items():
            res = walk_forward(fn, data.series, origins, HORIZON, actuals, name)
            res.loc[res["objetivo"] > TUNE_LAST_TARGET, "real"] = np.nan
            ev = evaluable(res, exclusions)
            print(f"[baseline {tag}] {name}: MAPE {ev['err_pct'].abs().mean():.2f} sesgo {ev['err_pct'].mean():+.2f}")
    return out


SPEC_PATH = _THIS_DIR / "clima_oni_v2_spec.json"


def spec_from_row(row: pd.Series) -> com.GrowthModelSpec:
    def tup(v):
        if isinstance(v, str):
            v = json.loads(v.replace("(", "[").replace(")", "]").replace(",]", "]").replace("'", '"'))
        return tuple(v) if v is not None else ()

    win = row["spec_window_years"]
    return com.GrowthModelSpec(
        extra_features=tup(row["spec_extra_features"]),
        oni_source=row["spec_oni_source"],
        window_years=None if pd.isna(win) else int(win),
        exclude_years=tup(row["spec_exclude_years"]),
        alpha=float(row["spec_alpha"]),
        nonneg_oni=bool(row["spec_nonneg_oni"]),
        n_years_shape=int(row["spec_n_years_shape"]),
        beta_all_history=bool(row["spec_beta_all_history"]),
    )


def select_spec(grid: pd.DataFrame) -> tuple[com.GrowthModelSpec, pd.DataFrame]:
    """Regla de selección (declarada antes de ver 2026): entre los specs **compatibles con la inferencia
    actual** (sin features de parche, ``n_years_shape=2`` que es el default de ``forecast.py``) y que
    **conservan la identidad** (``oni_mean_anio`` en las features), mínimo de la media de MAPE mensual
    (orígenes dic + orígenes mensuales <= 2023-12, objetivos <= 2025-12, sin COVID)."""
    g = grid.copy()
    g["score"] = (g["mape_dic"] + g["mape_mens"]) / 2
    ok = g["spec_extra_features"].astype(str).str.contains("oni_mean_anio") & (g["spec_n_years_shape"] == 2)
    ok &= ~g["spec_extra_features"].astype(str).str.contains(com.ONI_DELTA)
    best = g[ok].sort_values("score").iloc[0]
    return spec_from_row(best), g.sort_values("score")


# --------------------------------------------------------------------------------------------------
# Corrector post-proceso sin fuga
# --------------------------------------------------------------------------------------------------
def honest_corrector_eval(res: pd.DataFrame, origins_dec: list, ventana_meses: int = 6, peso_prior: float = 0.7):
    """Para cada origen ``o``: calibra los correctores (``postprocess_calibration``) SOLO con pares
    pred/real de orígenes anteriores cuyo objetivo es ``<= o`` (lo conocido en ``o``) y corrige las
    predicciones hechas en ``o``. Devuelve ``res`` con columna ``pred_corr`` y el último corrector."""
    from training.forecast_climate.postprocess_calibration import (
        apply_correctors,
        calibrate_bias_corrector,
        calibrate_seasonal_corrector,
    )

    out = res.copy()
    out["pred_corr"] = np.nan
    last = None
    for o in origins_dec:
        o = pd.Timestamp(o)
        past = res[(res["origen"] < o) & (res["objetivo"] <= o) & res["real"].notna()]
        # Un par por mes objetivo: el pronóstico más reciente (h más corto) disponible para ese mes.
        past = past.sort_values(["objetivo", "h"]).drop_duplicates("objetivo").sort_values("objetivo")
        bias = calibrate_bias_corrector(past["pred"], past["real"], ventana_meses, peso_prior)
        seas = calibrate_seasonal_corrector(past["pred"], past["real"], pd.DatetimeIndex(past["objetivo"]).month)
        cur = out["origen"] == o
        out.loc[cur, "pred_corr"] = [
            apply_correctors(p, pd.Timestamp(t).month, bias, seas)["prediccion_final"]
            for p, t in zip(out.loc[cur, "pred"], out.loc[cur, "objetivo"])
        ]
        last = (bias, seas, past)
    return out, last


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--dec-only", action="store_true")
    ap.add_argument("--select", action="store_true", help="elige el spec desde tuning_grid.csv y lo guarda")
    ap.add_argument("--final", action="store_true", help="tabla completa 2017-12..2025-12 + test 2026 (una vez)")
    ap.add_argument("--escenarios", action="store_true", help="sensibilidad ONI con los artefactos productivos")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    if args.tune:
        out = tune(args.dec_only)
        out.to_csv(OUT_DIR / "tuning_grid.csv", index=False)
        print(out[["label", "n_years_shape", "mape_dic", "sesgo_dic", "mape_mens", "sesgo_mens"]].head(40).round(3).to_string(index=False)
              if "mape_mens" in out else out[["label", "n_years_shape", "mape_dic", "sesgo_dic"]].head(40).round(3).to_string(index=False))
    if args.select or args.tune:
        spec, ranked = select_spec(pd.read_csv(OUT_DIR / "tuning_grid.csv"))
        SPEC_PATH.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
        print("spec elegido:", spec.label(), "nys", spec.n_years_shape)
        print(ranked[["label", "n_years_shape", "mape_dic", "mape_mens", "score"]].head(15).round(3).to_string(index=False))
    if args.final:
        final_report()
    if args.escenarios:
        md = scenario_report()
        (OUT_DIR / "escenarios_oni.md").write_text(md, encoding="utf-8")
        print(md)


def scenario_report() -> str:
    """Sensibilidad a escenarios ONI con los artefactos PRODUCTIVOS y el código real de inferencia
    (``DemandPredictor('api/models/climate_model/')``), corte = último año completo (2025)."""
    from api.services.forecast import DemandPredictor

    with contextlib.redirect_stdout(io.StringIO()):
        p = DemandPredictor(model_path="api/models/climate_model/")
    ha = p.climate_hist_annual
    cut = int(ha.loc[ha["n_meses"] >= 12, "year"].max())
    years = (cut + 1, cut + 2)
    real_last = float(ha.set_index("year").loc[cut, "demand_total"])

    def run(enso):
        with contextlib.redirect_stdout(io.StringIO()):
            ann, res = p.predict_future_climate(forecast_years=2, cutoff_year=cut, enso_phase_by_month=enso)
        return ann.ravel(), res.set_index("fecha")["prediccion"]

    def allm(phase, months=range(1, 13), yrs=years):
        return {f"{y}-{m:02d}": phase for y in yrs for m in months}

    scen = {
        "Auto (ONI proyectado)": None,
        "Todo NIÑA": allm("NIÑA"),
        "Todo NORMAL": allm("NORMAL"),
        "Todo NIÑO": allm("NIÑO"),
        f"NIÑO desde may-{years[0]} (resto auto)": {f"{years[0]}-{m:02d}": "NIÑO" for m in range(5, 13)},
    }
    rows, monthly = [], {}
    for name, enso in scen.items():
        ann, res = run(enso)
        monthly[name] = res
        rows.append({"escenario": name, f"total_{years[0]}_GWh": ann[0] / 1e6, f"total_{years[1]}_GWh": ann[1] / 1e6,
                     f"crec_{years[0]}_%": (ann[0] / real_last - 1) * 100})
    t = pd.DataFrame(rows)
    base = t.loc[t["escenario"] == "Todo NORMAL", f"total_{years[0]}_GWh"].iloc[0]
    t[f"vs_NORMAL_{years[0]}_%"] = (t[f"total_{years[0]}_GWh"] / base - 1) * 100
    # Sensibilidad de UN mes forzado (selector mes a mes): spread NIÑO vs NIÑA del mes.
    spreads = []
    for mth in range(1, 13):
        key = f"{years[0]}-{mth:02d}"
        v = {}
        for ph in ("NIÑA", "NORMAL", "NIÑO"):
            _, r = run({key: ph})
            v[ph] = float(r.loc[pd.Timestamp(f"{key}-01")])
        spreads.append({"mes": key, "NIÑA_GWh": v["NIÑA"] / 1e6, "NORMAL_GWh": v["NORMAL"] / 1e6, "NIÑO_GWh": v["NIÑO"] / 1e6,
                        "orden_ok": v["NIÑA"] < v["NORMAL"] < v["NIÑO"], "spread_%": (v["NIÑO"] / v["NIÑA"] - 1) * 100})
    s = pd.DataFrame(spreads)
    # Referencia 2026 (solo informativa, a posteriori): real ene-ago vs escenarios.
    act, exclusions = actual_series()
    a26 = act[(act.index.year == years[0]) & ~excluded_mask(act.index, exclusions)]  # sin meses post-quiebre
    comp = pd.DataFrame({n: (m.reindex(a26.index) / a26 - 1) * 100 for n, m in monthly.items()})
    mape = comp.abs().mean().rename("MAPE_%").to_frame()
    mape["sesgo_%"] = comp.mean()
    md = ["\n## Sensibilidad a escenarios ONI (artefactos productivos v2, código real de inferencia)\n",
          t.round(2).to_markdown(index=False), "",
          f"\nUn solo mes forzado ({years[0]}): demanda del mes por fase\n", s.round(2).to_markdown(index=False), "",
          f"\nA posteriori (informativo, NO usado para elegir nada): error vs real ene–ago {years[0]} por escenario\n",
          mape.round(2).to_markdown(), ""]
    return "\n".join(md)


def load_spec() -> com.GrowthModelSpec:
    d = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    d["extra_features"] = tuple(d["extra_features"])
    d["exclude_years"] = tuple(d["exclude_years"])
    return com.GrowthModelSpec(**d)


def final_report() -> None:
    """Evaluación final (se corre UNA vez tras congelar el spec): tablas antes/después sobre todos los
    orígenes, test 2026 y corrector sin fuga. Escribe CSV/MD en ``docs/work/04_clima/resultados/``."""
    clean = load_clean_data()
    orig = load_original_data()
    actuals, exclusions = actual_series()
    spec = load_spec()
    spec5 = com.GrowthModelSpec(**{**asdict(spec), "n_years_shape": 5})
    models = {
        "antes_artefacto_prod (in-sample <=2025)": make_fit_predict_artifact(orig, BACKUP_MODELS),
        "antes_receta_original (reentrenada por origen)": make_fit_predict_original_recipe(orig),
        "receta_original + datos_limpios": make_fit_predict_original_recipe(clean),
        "despues_v2 (nys=2, sin parche)": make_fit_predict(clean, spec),
        "despues_v2 (nys=5, con parche)": make_fit_predict(clean, spec5),
        "solo_deriva (mismo spec sin ONI)": make_fit_predict(
            clean, com.GrowthModelSpec(**{**asdict(spec), "extra_features": ()})),
        **BASELINES,
    }
    lines = ["# Resultados walk-forward modelo id=4 (generado por walkforward_clima.py --final)\n"]
    all_res = {}
    for tag, origins in (("dic", make_origins("2017-12", "2025-12", 12)), ("mens", make_origins("2017-12", "2025-12", 1))):
        res = pd.concat([walk_forward(fn, clean.series, origins, HORIZON, actuals, n) for n, fn in models.items()],
                        ignore_index=True)
        all_res[tag] = res
        res.to_csv(OUT_DIR / f"walkforward_{tag}.csv.gz", index=False)
        for scope, r in (("orígenes 2017-12..2025-12 (incluye 2026-01..08)", res),
                         ("solo objetivos <= 2025-12 (sin test)", res.assign(real=res["real"].where(res["objetivo"] <= TUNE_LAST_TARGET)))):
            tab = summary_table(r, exclusions)
            lines += [f"\n## Orígenes {tag} — {scope}\n", tab.to_markdown(index=False), ""]
    # Test final 2026: origen 2025-12, meses 2026-01..2026-08 (pre-quiebre).
    r = all_res["dic"]
    t26 = r[(r["origen"] == pd.Timestamp("2025-12-01")) & r["real"].notna() & ~excluded_mask(r["objetivo"], exclusions)]
    t26 = t26[t26["objetivo"].dt.year == 2026]
    tt = t26.groupby("modelo")["err_pct"].agg(mape=lambda e: e.abs().mean(), sesgo="mean",
                                              max_ape=lambda e: e.abs().max(), n="size").round(2)
    tot = t26.groupby("modelo")[["pred", "real"]].sum()
    tt["err_total_ene_ago"] = ((tot["pred"] / tot["real"] - 1) * 100).round(2)
    lines += ["\n## Test final 2026 (origen 2025-12, ene–ago 2026, crudo, una sola vez)\n", tt.to_markdown(), ""]
    piv = t26.pivot_table(index="objetivo", columns="modelo", values="err_pct").round(2)
    lines += ["\nError % por mes 2026:\n", piv.to_markdown(), ""]
    # Corrector sin fuga sobre el modelo elegido (decisión con objetivos <= 2025-12).
    name = "despues_v2 (nys=2, sin parche)"
    dec = all_res["dic"]
    sub = dec[dec["modelo"] == name]
    corr, last = honest_corrector_eval(sub, make_origins("2017-12", "2025-12", 12))
    ev = corr[corr["real"].notna() & ~excluded_mask(corr["objetivo"], exclusions)].copy()
    ev["err_corr"] = (ev["pred_corr"] - ev["real"]) / ev["real"] * 100
    rows = []
    for scope, e in (("objetivos <= 2025-12", ev[ev["objetivo"] <= TUNE_LAST_TARGET]),
                     ("test 2026", ev[(ev["objetivo"].dt.year == 2026) & (ev["origen"] == pd.Timestamp("2025-12-01"))])):
        rows.append({"alcance": scope, "mape_crudo": e["err_pct"].abs().mean(), "mape_corregido": e["err_corr"].abs().mean(),
                     "sesgo_crudo": e["err_pct"].mean(), "sesgo_corregido": e["err_corr"].mean(), "n": len(e)})
    lines += ["\n## Corrector post-proceso calibrado sin fuga (walk-forward)\n", pd.DataFrame(rows).round(2).to_markdown(index=False), ""]
    pd.to_pickle(last, OUT_DIR / "corrector_ultimo_origen.pkl")
    (OUT_DIR / "resultados_final.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
