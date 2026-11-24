"""Ajuste final del modelo elegido y serialización del artefacto (api/models/sarimax_model/sarimax_v2.json).

La elección (componentes) vive en ``training/forecast_sarimax/seleccion.json`` y se decidió SOLO con la
selección walk-forward (orígenes <= 2023-12, reales <= 2025-12). Aquí se re-estiman los parámetros con toda la
historia limpia pre-quiebre de mercado (escala completa). Los originales están respaldados en
docs/work/backup_modelos_originales/models/sarimax_model/ y NO se borran (forecast.py los sigue usando hasta
que agente-e2e integre el módulo nuevo).

Uso: ``python -m training.forecast_sarimax.train_final``
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from api.services.forecast_sarimax import ARTIFACT_FILE, MODEL_DIR, Artifact, ModelSpec, fit_spec
from training.forecast_sarimax.wf import load_data

SELECTION_FILE = Path(__file__).resolve().parent / "seleccion.json"


def main():
    sel = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    train, _, _, mb = load_data()
    comps = []
    for d in sel["components"]:
        spec = ModelSpec.from_dict(d)
        res = fit_spec(train, spec)
        comps.append({
            "spec": spec.to_dict(),
            "params": [float(x) for x in res.params],
            "param_names": list(getattr(res.model, "param_names", [])),
            "aic": float(res.aic),
        })
        print(spec.name(), dict(zip(comps[-1]["param_names"], [round(p, 4) for p in comps[-1]["params"]])))
    art = Artifact(
        components=comps,
        weights=sel.get("weights", [1.0] * len(comps)),
        train_end=f"{train.index.max():%Y-%m}",
        metadata={
            "nombre": sel["nombre"],
            "creado": datetime.now().isoformat(timespec="seconds"),
            "datos": "PRONOSTICO_demands limpia (training.common.data): dedupe 2020-Q4, 2021-09 imputado, corte en "
                     f"quiebre de mercado {mb.first_post_month:%Y-%m}" if mb else "sin quiebre",
            "escala": "mercado completo (kWh); el llamador aplica el factor de división de mercado",
            "seleccion": sel.get("criterio", ""),
            "metricas_walkforward": sel.get("metricas", {}),
        },
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    art.save(MODEL_DIR / ARTIFACT_FILE)
    print("guardado", MODEL_DIR / ARTIFACT_FILE, "train_end", art.train_end)


if __name__ == "__main__":
    main()
