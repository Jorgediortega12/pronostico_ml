"""Smoke e2e HTTP real contra uvicorn (agente-e2e).

Uso (con la API levantada en BASE, p. ej. ``python -m uvicorn api.main:app --port 8000``)::

    python docs/work/05_e2e/scripts/smoke_http.py --base http://127.0.0.1:8000 \
        --database-url postgresql+psycopg2://postgres@127.0.0.1:55432/afinia

Credenciales de servicio desde el entorno/.env (ML_SERVICE_USER / ML_SERVICE_PASSWORD).
Escribe en docs/work/05_e2e/:
  * smoke_resumen.csv      — una fila por petición (estado, MAPE con tope y crudo, meses con tope, tiempos)
  * mape_ene_ago_2026.csv  — MAPE ene-ago 2026 crudo vs con tope por modelo (start=2026-01)
  * curvas_2026_2027.csv   — curva mensual 2026-2027 de cada modelo (crudo, servido, real, error)
  * escenarios.csv         — totales 2027 por escenario (IPC id 1/2, ENSO id 4)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "work" / "05_e2e"
MODELOS = {1: "RF", 2: "XGB", 3: "SARIMAX", 4: "Clima+ONI"}
RANGOS = [("2026-01-01", "2027-12-31"), ("2025-01-01", "2025-12-31"), ("2027-01-01", "2027-12-31"),
          ("2026-03-01", "2026-03-01"), ("2026-07-01", "2028-12-31")]


def login(base: str) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    r = requests.post(f"{base}/v1/autenticacion/login",
                      data={"username": os.environ["ML_SERVICE_USER"], "password": os.environ["ML_SERVICE_PASSWORD"]},
                      timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def predict(base, token, dburl, **body):
    payload = {"user_id": 1, "session_id": 1, "database_url": dburl, **body}
    t = time.time()
    r = requests.post(f"{base}/v1/forecast/predict/", json=payload, headers={"Authorization": f"Bearer {token}"},
                      timeout=300)
    dt = time.time() - t
    return r.status_code, r.text, dt


def _mape(errs):
    v = [abs(e) for e in errs if e is not None and math.isfinite(e)]
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--database-url", default="postgresql+psycopg2://postgres@127.0.0.1:55432/afinia")
    a = ap.parse_args()
    token = login(a.base)
    resumen, curvas, mape26, esc = [], [], [], []
    fallos = 0

    def run(mid, start, end, tag="base", **extra):
        nonlocal fallos
        code, text, dt = predict(a.base, token, a.database_url, id=mid, start_date=start, end_date=end, **extra)
        row = {"modelo": MODELOS[mid], "id": mid, "start": start, "end": end, "escenario": tag, "http": code,
               "segundos": round(dt, 2)}
        try:
            body = json.loads(text)  # json estricto: NaN/Infinity no son JSON válido para requests/json
            json.dumps(body, allow_nan=False)
        except ValueError as exc:
            row["estado"] = f"JSON invalido: {exc}"
            fallos += 1
            resumen.append(row)
            return None
        if code != 200 or "monthly_predictions" not in body:
            row["estado"] = f"error: {body.get('message') or body.get('detail')}"
            fallos += 1
            resumen.append(row)
            return None
        mp = body["monthly_predictions"]
        crud = body.get("metricas_error_crudas", {})
        row.update({
            "estado": "ok", "n_meses": len(mp["fechas"]),
            "mape_con_tope": body["metricas_error"]["mape_total_ventana"],
            "mape_crudo": crud.get("mape_total_ventana"),
            "mape_crudo_pre_division": crud.get("mape_meses_pre_division"),
            "meses_con_real": crud.get("num_meses_con_datos_reales"),
            "meses_post_division_con_real": crud.get("num_meses_post_division_con_real"),
            "meses_con_tope": crud.get("num_meses_con_tope"),
            "meses_con_override": crud.get("num_meses_con_override"),
            "total_ultimo_anio_gwh": round(body["annual_predictions"][-1] / 1e6, 1),
            "min_mes_gwh": round(min(mp["valores_pronostico"]) / 1e6, 1),
            "max_mes_gwh": round(max(mp["valores_pronostico"]) / 1e6, 1),
        })
        resumen.append(row)
        return body

    for mid in MODELOS:
        for start, end in RANGOS:
            body = run(mid, start, end)
            if body is None or (start, end) != ("2026-01-01", "2027-12-31"):
                continue
            mp = body["monthly_predictions"]
            df = pd.DataFrame({k: mp[k] for k in ("fechas", "valores_pronostico", "valores_pronostico_crudos",
                                                  "valores_reales", "errores_porcentuales",
                                                  "errores_porcentuales_crudos", "tope_aplicado",
                                                  "override_aplicado", "post_division", "factor_division")})
            df.insert(0, "modelo", MODELOS[mid])
            for c in ("valores_pronostico", "valores_pronostico_crudos", "valores_reales"):
                df[c.replace("valores", "gwh")] = pd.to_numeric(df[c]) / 1e6
            curvas.append(df)
            ea = df[df["fechas"] < "2026-09-01"]
            mape26.append({
                "modelo": MODELOS[mid],
                "mape_crudo_ene_ago_2026": _mape(ea["errores_porcentuales_crudos"].tolist()),
                "mape_con_tope_y_override_ene_ago_2026": _mape(ea["errores_porcentuales"].tolist()),
                "sesgo_crudo_ene_ago_2026": float(pd.to_numeric(ea["errores_porcentuales_crudos"]).mean()),
                "meses_con_tope_ene_ago": int(ea["tope_aplicado"].sum()),
                "max_abs_err_crudo": float(pd.to_numeric(ea["errores_porcentuales_crudos"]).abs().max()),
                "total_2026_gwh": float(df.loc[df["fechas"] < "2027-01-01", "gwh_pronostico_crudos"].sum()),
                "total_2027_gwh": float(df.loc[df["fechas"] >= "2027-01-01", "gwh_pronostico_crudos"].sum()),
                "err_crudo_sep_2026_escala_front": df.loc[df["fechas"].str[:7] == "2026-09", "errores_porcentuales_crudos"].iloc[0],
                "err_crudo_oct_2026_escala_front": df.loc[df["fechas"].str[:7] == "2026-10", "errores_porcentuales_crudos"].iloc[0],
            })

    # Escenarios
    for mid in (1, 2):
        for ipc in (None, {2027: -5.0}, {2027: 5.0}):
            body = run(mid, "2027-01-01", "2027-12-31", tag=f"ipc={ipc}", **({"ipc_growth_by_year": ipc} if ipc else {}))
            if body:
                esc.append({"modelo": MODELOS[mid], "escenario": f"ipc={ipc}",
                            "total_2027_gwh": sum(body["monthly_predictions"]["valores_pronostico"]) / 1e6})
    for fase in ("NIÑA", "NORMAL", "NIÑO"):
        enso = {f"2027-{m:02d}": fase for m in range(1, 13)}
        body = run(4, "2027-01-01", "2027-12-31", tag=f"enso={fase}", enso_phase_by_month=enso)
        if body:
            esc.append({"modelo": "Clima+ONI", "escenario": f"enso={fase}",
                        "total_2027_gwh": sum(body["monthly_predictions"]["valores_pronostico"]) / 1e6})
    # Rango invertido: debe responder error controlado (no 500)
    run(1, "2027-01-01", "2026-01-01", tag="rango_invertido")
    if resumen[-1]["estado"].startswith("error") and resumen[-1]["http"] == 200:
        resumen[-1]["estado"] = "ok (error controlado esperado)"
        fallos -= 1

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resumen).to_csv(OUT / "smoke_resumen.csv", index=False)
    pd.DataFrame(mape26).to_csv(OUT / "mape_ene_ago_2026.csv", index=False)
    if curvas:
        pd.concat(curvas).to_csv(OUT / "curvas_2026_2027.csv", index=False)
    pd.DataFrame(esc).to_csv(OUT / "escenarios.csv", index=False)
    print(pd.DataFrame(resumen).to_string())
    print(pd.DataFrame(mape26).to_string())
    print(pd.DataFrame(esc).to_string())
    print(f"FALLOS: {fallos}")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
