"""Escribe los artefactos del modelo id=4 v3 en ``api/models/climate_model/``.

v3 no tiene pesos entrenados: nivel, estacionalidad, deriva y ``beta`` ENSO se re-estiman en cada llamada con el
historico anterior al inicio pedido (``clima_v3.predict_years``). Lo que se versiona es el **spec** (hiperparametros
elegidos en TUNE, ver ``walkforward_clima_v3.CHOSEN``) y las metricas walk-forward honestas.

Escribe ``climate_v3.json`` y actualiza ``metrics.pkl`` (``annual``/``monthly`` = metricas v3; la copia v2 queda en
``metrics.pkl['v2']`` y los archivos originales en ``api/models/climate_model/v2_backup/``).

Uso::

    python -m training.forecast_climate.train_final_clima_v3 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from training.common.walkforward import annual_totals, evaluable, make_origins, walk_forward
from training.forecast_climate import clima_v3 as v3
from training.forecast_climate import walkforward_clima_v3 as w

MODEL_DIR = w.REPO_ROOT / "api" / "models" / "climate_model"


def honest_metrics(spec: v3.V3Spec) -> dict:
    full = w.load_hist()
    actuals, exclusions = w.actuals_and_exclusions()
    origins = make_origins("2017-12", "2025-12", 12)
    res = walk_forward(w.make_fit_predict(full, spec), w.series_of(full), origins, w.HORIZON, actuals, "v3")
    ev = evaluable(res, exclusions, common_support=False)
    at = annual_totals(res, exclusions)
    at = at[at["h_max"] <= 12]
    return {
        "annual": {"mape": float(at["err_pct"].abs().mean()), "model": "v3 (suma de 12 meses, origen dic previo)",
                   "protocolo": "walk-forward orígenes dic 2017-12..2025-12, total anual h1-12 con 12 meses reales, sin COVID"},
        "monthly": {"mape": float(ev["err_pct"].abs().mean()), "sesgo": float(ev["err_pct"].mean()),
                    "mape_h1_12": float(ev.loc[ev["h"] <= 12, "err_pct"].abs().mean()),
                    "mape_h13_24": float(ev.loc[ev["h"] > 12, "err_pct"].abs().mean()),
                    "model": "clima_oni_v3 (regresión de nivel por ventana + deriva + beta ONI)",
                    "protocolo": "walk-forward orígenes dic 2017-12..2025-12, h=1..24, crudo, sin COVID ni post-quiebre"},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    spec = w.CHOSEN
    full = w.load_hist()
    h = full.sort_values(["year", "month"]).reset_index(drop=True)
    y = pd.Series(np.log(h["demand_value"].to_numpy(float)),
                  index=pd.DatetimeIndex(pd.to_datetime(dict(year=h["year"], month=h["month"], day=1))))
    oni = v3._oni_series(h, 1, None)
    beta = v3.estimate_beta(y, oni, spec)
    g = v3.estimate_drift(y, oni, beta, spec)
    metrics = honest_metrics(spec)
    info = {
        "version": "v3",
        "spec": asdict(spec),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "ultimo_mes_historico": f"{int(h['year'].iloc[-1])}-{int(h['month'].iloc[-1]):02d}",
        "beta_oni_por_unidad_estimado_al_ultimo_mes": beta,
        "deriva_anual_log_estimada_al_ultimo_mes": g,
        "formula": "log D_t = a + g*tau + estacionalidad_mes + beta*ONI_t (ventana 'window' meses; a, estacionalidad por MCO; "
                   "beta por MCO expansivo sobre Delta12; g = mezcla deriva reciente/larga)",
        "metricas_walkforward": metrics,
        "fuente": "docs/work/09_clima_v3/REPORTE.md; training/forecast_climate/walkforward_clima_v3.py (CHOSEN)",
    }
    print(json.dumps(info, indent=1, default=str))
    if args.dry_run:
        return
    (MODEL_DIR / "climate_v3.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    old = joblib.load(MODEL_DIR / "v2_backup" / "metrics.pkl")  # siempre desde el respaldo v2 (idempotente)
    joblib.dump({"annual": metrics["annual"], "monthly": metrics["monthly"], "v2": old}, MODEL_DIR / "metrics.pkl")
    print("escrito", MODEL_DIR / "climate_v3.json", "y metrics.pkl")


if __name__ == "__main__":
    main()
