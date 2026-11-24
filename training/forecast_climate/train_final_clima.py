"""Reentrena el modelo id=4 (clima+ONI) v2 con el spec elegido por walk-forward y escribe los artefactos
que la inferencia actual (``api/services/forecast.py``) ya sabe cargar.

Pasos:
1. Regenera ``training/forecast_climate/artifacts/dataset_{mensual,anual}.csv`` (capa limpia).
2. Ajusta el modelo de crecimiento con todos los años completos <= último año completo (2025).
3. Escribe en ``api/models/climate_model/``: ``annual_scaler_x.pkl``, ``annual_model.pkl``,
   ``annual_scaler_y.pkl``, ``metrics.pkl`` (métricas walk-forward honestas), ``model_info.pkl`` y
   ``postproc_corrector_v1.json`` (recalibrado SIN fuga con residuos walk-forward). ``monthly_model.pkl`` y
   ``oni_decay_v1.json`` se conservan (el primero no se usa en inferencia; se mantiene por contrato).

Originales respaldados en ``docs/work/backup_modelos_originales/models/climate_model`` y
``docs/work/backup_modelos_originales/climate_datasets``.

Uso::

    python -m training.forecast_climate.train_final_clima [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import pandas as pd

from training.forecast_climate import clima_oni_model as com
from training.forecast_climate import walkforward_clima as wf
from training.forecast_climate.build_clean_dataset import ARTIFACTS_DIR
from training.forecast_climate.postprocess_calibration import export_corrector_artifact

MODEL_DIR = wf.REPO_ROOT / "api" / "models" / "climate_model"


def last_full_year(anual: pd.DataFrame) -> int:
    return int(anual.loc[anual["n_meses"] >= 12, "year"].max())


def fit_final(data: wf.ClimaData, spec: com.GrowthModelSpec):
    cutoff = last_full_year(data.a)
    tab = data.training_table(cutoff, spec.oni_source)
    model = com.fit_growth_model(tab, spec)
    if com.requires_patch(model):
        raise RuntimeError("el spec elegido requiere el parche de inferencia; no se escriben artefactos")
    target = tab.loc[model.train_years, "target_log"].to_numpy(float)
    arts = com.to_sklearn_artifacts(model, model.train_X, target)
    return cutoff, tab, model, arts


def walkforward_metrics() -> dict:
    """Lee las métricas honestas del último ``--final`` (walkforward_dic.csv.gz) del modelo elegido."""
    path = wf.OUT_DIR / "walkforward_dic.csv.gz"
    res = pd.read_csv(path, parse_dates=["origen", "objetivo"])
    actuals, exclusions = wf.actual_series()
    sub = res[res["modelo"] == "despues_v2 (nys=2, sin parche)"]
    s = wf.summarize(sub, exclusions).iloc[0]
    return {
        "annual": {"mape": float(s["mape_anual"]), "model": "crecimiento_lag1_fijo+oni",
                   "protocolo": "walk-forward orígenes dic 2017-12..2025-12, total anual con 12 meses reales, sin COVID"},
        "monthly": {"mape": float(s["mape"]), "sesgo": float(s["sesgo"]), "mape_h1_12": float(s["mape_h1_12"]),
                    "mape_h13_24": float(s["mape_h13_24"]), "model": "anual x reparto historico (n_years_shape)",
                    "protocolo": "walk-forward orígenes dic 2017-12..2025-12, h=1..24, crudo, sin COVID ni post-quiebre"},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    data = wf.load_clean_data()
    spec = wf.load_spec()
    cutoff, _tab, model, (sx, lr, sy) = fit_final(data, spec)
    print("spec:", spec.label(), "| cutoff", cutoff, "| mu", round(model.mu, 5), "| coefs", model.coefs)
    print("años de ajuste (mu):", model.train_years)

    metrics = walkforward_metrics()
    info = {
        # Claves históricas (contrato de lectura):
        "annual_features": list(sx.feature_names_in_),
        "monthly_features": list(joblib.load(MODEL_DIR / "monthly_model.pkl").feature_names_in_),
        "annual_model_name": "crecimiento_lag1_fijo_oni_v2",
        "monthly_model_name": "reparto_historico (monthly_model.pkl no se usa en inferencia)",
        "monthly_target": "demand_percentage",
        "annual_target": "log1p(demand_total), scaled by annual_scaler_y",
        # Nuevas:
        "version": "v2",
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "mu": model.mu,
        "coefs_espacio_original": model.coefs,
        "formula": "log1p(D_t) = log1p(D_{t-1}) + mu + beta_oni * oni_mean_anio_t",
        "anios_ajuste_mu": model.train_years,
        "ultimo_anio_completo": cutoff,
        # nys=5 empata con 2 en el walk-forward total (4.52 vs 4.52) y es peor en los orígenes posteriores a
        # la selección (2024-01..2025-12: 4.22 vs 3.97) -> se mantiene el contrato original (2 años).
        "n_years_shape_recomendado": 2,
        "n_years_shape_inferencia_actual": 2,
        "fuente_datos": "training/forecast_climate/build_clean_dataset.py (PRONOSTICO_demands limpia, pre-quiebre)",
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in info.items() if k != "monthly_features"}, indent=1, default=str))
        print(metrics)
        return

    data.m.to_csv(ARTIFACTS_DIR / "dataset_mensual.csv", index=False)
    data.a.to_csv(ARTIFACTS_DIR / "dataset_anual.csv", index=False)
    joblib.dump(sx, MODEL_DIR / "annual_scaler_x.pkl")
    joblib.dump(lr, MODEL_DIR / "annual_model.pkl")
    joblib.dump(sy, MODEL_DIR / "annual_scaler_y.pkl")
    joblib.dump(metrics, MODEL_DIR / "metrics.pkl")
    joblib.dump(info, MODEL_DIR / "model_info.pkl")

    bias, seas, past = pd.read_pickle(wf.OUT_DIR / "corrector_ultimo_origen.pkl")
    export_corrector_artifact(
        bias, seas, str(MODEL_DIR / "postproc_corrector_v1.json"),
        extra_meta={
            "fuente_datos": "residuos walk-forward (pronóstico más reciente por mes, objetivos <= 2025-12) del modelo v2",
            "sin_fuga": True,
            "aplicado_por_defecto": False,
            "nota": "forecast.py lo calcula como prediccion_ajustada pero sirve la cruda (apply_corrector=False); "
                    "ver docs/work/04_clima/REPORTE.md",
            "n_pares": int(len(past)),
        },
    )
    print("artefactos escritos en", MODEL_DIR)


if __name__ == "__main__":
    main()
