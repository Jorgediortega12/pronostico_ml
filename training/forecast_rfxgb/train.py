"""Entrenamiento final (refit con toda la historia pre-quiebre) y serialización de los artefactos v2.

Lee la configuración elegida por validación (``selected_config.json``, escrita tras correr
``experiments.py``), entrena RF (id 1) y XGB (id 2) con todos los meses completos anteriores al quiebre de
mercado y guarda::

    api/models/rf_model/v2/{model.joblib, metadata.json}
    api/models/xgb_model/v2/{model.joblib, metadata.json}

Los .pkl de ``api/models/rf_model`` / ``xgb_model`` (modelo legado que usa ``forecast.py``) no se tocan.

Uso::

    python -m training.forecast_rfxgb.train [--until 2026-08]
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from api.services import forecast_rfxgb as F
from training.common.data import load_monthly_demand, target_series

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SELECTED = HERE / "selected_config.json"


def load_selected() -> dict:
    return json.loads(SELECTED.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=None, help="último mes de entrenamiento (YYYY-MM); por defecto todo")
    ap.add_argument("--models-root", default=str(F.MODELS_ROOT))
    args = ap.parse_args()

    monthly = load_monthly_demand(impute_incomplete=True)
    mb = monthly.attrs["market_break"]
    series = target_series(monthly, until=args.until, market_break=mb)
    sel = load_selected()
    report = {}
    report_path = REPO / "docs" / "work" / "02_rfxgb" / "walkforward_final_resumen.csv"
    if report_path.exists():
        report = pd.read_csv(report_path).set_index("modelo").to_dict(orient="index")

    for key, mid in (("rf", 1), ("xgb", 2)):
        cfg = sel[key]
        art = F.fit_model(series, mid, cfg)
        out_dir = Path(args.models_root) / ("rf_model" if mid == 1 else "xgb_model") / "v2"
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(art, out_dir / F.ARTIFACT_FILE)
        meta = {k: v for k, v in art.items() if k != "estimator"}
        meta.update(
            {
                "entrenado": datetime.now().isoformat(timespec="seconds"),
                "fuente": "BD PRONOSTICO_demands via training.common.data (dedupe 2020-Q4, 2021-09 imputado)",
                "quiebre_mercado": None if mb is None else {"fecha_limite": str(mb.fecha_limite),
                                                            "porcentaje": mb.porcentaje,
                                                            "primer_mes_post": f"{mb.first_post_month:%Y-%m}"},
                "seleccion": "walk-forward orígenes 2017-12..2023-12 (objetivos <= 2025-12); ver docs/work/02_rfxgb",
                "metricas_walkforward": report.get(sel[key].get("name", key), {}),
                "python": platform.python_version(),
                "unidad": "kWh, escala mercado completo",
            }
        )
        (out_dir / F.METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        print(f"{F.MODEL_NAMES[mid]}: {art['n_train_rows']} filas, {art['train_first_month']}..{art['train_last_month']} -> {out_dir}")


if __name__ == "__main__":
    main()
