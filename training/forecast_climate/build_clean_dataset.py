"""Regenera ``artifacts/dataset_mensual.csv`` y ``artifacts/dataset_anual.csv`` del modelo id=4 (clima+ONI)
con la capa de datos limpia (``training/common/data.py``).

Cambios frente al dataset original (copia en ``docs/work/backup_modelos_originales/climate_datasets/``):

* Demanda desde ``PRONOSTICO_demands`` deduplicada: 2020-Q4 ya no está deflactado (306/220/356 GWh →
  615/597/631 GWh) y 2021-09 (28/30 días) se imputa como promedio diario × días.
* Solo meses completos, definitivos y **anteriores al quiebre de mercado** (``demand_split_config``): la
  serie post-quiebre está en otra escala y no debe alimentar lags ni shares.
* Clima y ONI: los mismos valores mensuales del dataset original (congelados en
  ``data/clima_oni_mensual_v1.csv``; provienen de ``Clima_ctg_*.csv`` + NOAA ONI). El ONI se completa con el
  índice NOAA (``data/oni_index_noaa_1950_2026.csv``) y, para los meses con demanda pero sin ONI publicado,
  con la **proyección** ``features.project_oni_future`` desde el último ONI observado (lo único conocible en
  ese momento). La columna ``oni_fuente`` (observado/proyectado) lo deja trazable. El clima de meses sin
  dato queda NaN (la inferencia usa climatología y promedia ignorando NaN).

Uso::

    python -m training.forecast_climate.build_clean_dataset            # escribe artifacts/
    python -m training.forecast_climate.build_clean_dataset --dry-run  # solo imprime resumen
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from training.common.data import load_monthly_demand, target_series
from training.forecast_climate import features as climate_features

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = _THIS_DIR / "data"
ARTIFACTS_DIR = _THIS_DIR / "artifacts"
CLIMA_MENSUAL_PATH = DATA_DIR / "clima_oni_mensual_v1.csv"
CLIMA_ANUAL_PATH = DATA_DIR / "clima_oni_anual_v1.csv"
ONI_NOAA_PATH = DATA_DIR / "oni_index_noaa_1950_2026.csv"

CLIMATE_MONTHLY_COLS = [
    "temp_mean_mes",
    "temp_max_mes",
    "temp_min_mes",
    "humidity_mean_mes",
    "wind_speed_mean_mes",
    "rain_sum_mes",
    "dias_calurosos_mes",
]
CLIMATE_ANNUAL_COLS = [
    "temp_mean_anio",
    "temp_max_anio",
    "temp_min_anio",
    "humidity_mean_anio",
    "wind_speed_mean_anio",
    "rain_sum_anio",
]
MONTHLY_SCHEMA = ["year", "month", "demand_value", "demand_percentage", *CLIMATE_MONTHLY_COLS, "oni_actual", "oni_fuente"]
ANNUAL_SCHEMA = ["year", "demand_total", *CLIMATE_ANNUAL_COLS, "oni_mean_anio", "n_meses"]

#: Ventana móvil NOAA (centrada) asociada a cada mes calendario (igual que el build original).
ONI_MONTH_TO_SEASON = {
    1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM", 5: "AMJ", 6: "MJJ",
    7: "JJA", 8: "JAS", 9: "ASO", 10: "SON", 11: "OND", 12: "NDJ",
}


def load_oni_noaa(path: Path = ONI_NOAA_PATH) -> pd.DataFrame:
    """ONI NOAA (ancho, año × trimestre móvil) → largo ``year, month, oni_actual`` sin NaN."""
    wide = pd.read_csv(path)
    rows = [
        {"year": int(r["year"]), "month": m, "oni_actual": r[s]}
        for _, r in wide.iterrows()
        for m, s in ONI_MONTH_TO_SEASON.items()
    ]
    return pd.DataFrame(rows).dropna().sort_values(["year", "month"]).reset_index(drop=True)


def demand_monthly_frame(series: pd.Series) -> pd.DataFrame:
    """Serie mensual (kWh, índice inicio de mes) → ``year, month, demand_value``."""
    s = series.dropna()
    return pd.DataFrame(
        {"year": s.index.year.astype(int), "month": s.index.month.astype(int), "demand_value": s.to_numpy(float)}
    )


def build_datasets(
    demand: pd.Series,
    clima_mensual: pd.DataFrame,
    clima_anual: pd.DataFrame,
    oni_noaa: pd.DataFrame,
    n_years_clim: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une demanda limpia + clima + ONI con el esquema que espera ``predict_future_climate``.

    Args:
        demand: serie mensual limpia (kWh) **ya cortada** en el quiebre de mercado y sin meses parciales.
        clima_mensual: ``year, month`` + :data:`CLIMATE_MONTHLY_COLS` (+ ``oni_actual`` opcional, ignorado).
        clima_anual: ``year`` + :data:`CLIMATE_ANNUAL_COLS` (agregados anuales del horario original).
        oni_noaa: salida de :func:`load_oni_noaa`.
        n_years_clim: ventana de climatología usada para proyectar ONI faltante (igual que inferencia).
    """
    dm = demand_monthly_frame(demand)
    clim = clima_mensual[["year", "month", *CLIMATE_MONTHLY_COLS]]
    out = dm.merge(clim, on=["year", "month"], how="left").merge(oni_noaa, on=["year", "month"], how="left")
    out["oni_fuente"] = np.where(out["oni_actual"].notna(), "observado", None)

    # Meses con demanda pero sin ONI publicado (solo puede ser la cola): proyección al último ONI observado.
    missing = out["oni_actual"].isna()
    if missing.any():
        first_missing = out.index[missing][0]
        if not missing.loc[first_missing:].all():
            raise ValueError("hay huecos de ONI en medio del histórico; revisar el CSV NOAA")
        hist = out.loc[: first_missing - 1, ["year", "month", "oni_actual"]]
        proj = climate_features.project_oni_future(hist, int(missing.sum()), n_years_clim=n_years_clim)
        out.loc[missing, "oni_actual"] = proj["oni_proyectado"].to_numpy()
        out.loc[missing, "oni_fuente"] = "proyectado"

    totals = out.groupby("year")["demand_value"].transform("sum")
    out["demand_percentage"] = out["demand_value"] / totals
    out = out[MONTHLY_SCHEMA].sort_values(["year", "month"]).reset_index(drop=True)

    ann = out.groupby("year").agg(
        demand_total=("demand_value", "sum"), n_meses=("month", "size"), oni_mean_anio=("oni_actual", "mean")
    ).reset_index()
    clim_a = clima_anual[["year", *CLIMATE_ANNUAL_COLS]].copy()
    # Años sin agregado anual original (p. ej. año en curso): agregación desde los meses disponibles.
    from_months = (
        out.groupby("year")
        .agg(
            temp_mean_anio=("temp_mean_mes", "mean"),
            temp_max_anio=("temp_max_mes", "max"),
            temp_min_anio=("temp_min_mes", "min"),
            humidity_mean_anio=("humidity_mean_mes", "mean"),
            wind_speed_mean_anio=("wind_speed_mean_mes", "mean"),
            rain_sum_anio=("rain_sum_mes", "sum"),
        )
        .reset_index()
    )
    clim_a = pd.concat([clim_a, from_months[~from_months["year"].isin(clim_a["year"])]], ignore_index=True)
    ann = ann.merge(clim_a, on="year", how="left")[ANNUAL_SCHEMA].sort_values("year").reset_index(drop=True)
    return out, ann


def build_from_db(engine_or_url=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo desde la BD + archivos congelados de ``data/``."""
    monthly = load_monthly_demand(engine_or_url, impute_incomplete=True)
    demand = target_series(monthly, include_imputed=True, market_break=monthly.attrs["market_break"])
    return build_datasets(
        demand,
        pd.read_csv(CLIMA_MENSUAL_PATH),
        pd.read_csv(CLIMA_ANUAL_PATH),
        load_oni_noaa(),
    )


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args(argv)
    mensual, anual = build_from_db()
    print(anual.assign(demand_total=(anual["demand_total"] / 1e9).round(4)).to_string(index=False))
    print(mensual.tail(6).to_string(index=False))
    if not args.dry_run:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        mensual.to_csv(out / "dataset_mensual.csv", index=False)
        anual.to_csv(out / "dataset_anual.csv", index=False)
        print(f"escrito en {out}")


if __name__ == "__main__":
    main()
